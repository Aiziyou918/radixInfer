from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, TypeAlias

import torch

from radixinfer.utils import align_down

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


# ---------------------------------------------------------------------------
# Base types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    cached_len: int           # tokens already in cache before insertion (to be freed)
    handle: BaseCacheHandle


class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle


class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None: ...

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult: ...

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult: ...

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor: ...

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...

    @abstractmethod
    def check_integrity(self) -> None: ...


# ---------------------------------------------------------------------------
# RadixTreeNode
# ---------------------------------------------------------------------------

class RadixTreeNode:
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        self.timestamp = tic or time.monotonic_ns()
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        return self._value

    def is_root(self) -> bool:
        return self._parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        from radixinfer.kernel import fast_compare_key
        cmp_len = min(self._length, len(input_ids))
        return fast_compare_key(self._key, input_ids, cmp_len)

    def split_at(self, pos: int) -> RadixTreeNode:
        assert 0 < pos < self.length
        parent = self.parent
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)
        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        return self.timestamp < other.timestamp


# ---------------------------------------------------------------------------
# RadixCacheHandle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        node = self.node
        parts: list[torch.Tensor] = []
        while not node.is_root():
            parts.append(node.value)
            node = node.parent
        if not parts:
            return torch.empty(0, dtype=torch.int32)
        parts.reverse()
        return torch.cat(parts)


# ---------------------------------------------------------------------------
# RadixPrefixCache
# ---------------------------------------------------------------------------

def _get_key_fn(page_size: int) -> KEY_FN:
    if page_size == 1:
        return lambda x: x[0].item()
    return lambda x: tuple(x[:page_size].tolist())


class RadixPrefixCache(BasePrefixCache):
    def __init__(self, device: torch.device) -> None:
        from radixinfer.core import get_global_ctx
        ctx = get_global_ctx()
        self.device = device
        self.page_size = ctx.page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert size <= self.evictable_size, (
            f"Cannot evict {size}, only {self.evictable_size} is evictable"
        )
        leaf_nodes = self._collect_evictable_leaves()
        heapq.heapify(leaf_nodes)
        evicted: list[torch.Tensor] = []
        evicted_size = 0
        while evicted_size < size:
            assert leaf_nodes, f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leaf_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaf_nodes, parent)
        return torch.cat(evicted).to(self.device)

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        if self.root_node.ref_count != 1:
            raise RuntimeError(
                f"RadixPrefixCache integrity check failed: root ref_count={self.root_node.ref_count}, expected 1"
            )

        seen: set[int] = set()
        evictable_size = 0
        protected_size = 0
        stack = [self.root_node]
        while stack:
            node = stack.pop()
            if node.uuid in seen:
                raise RuntimeError("RadixPrefixCache integrity check failed: cycle detected")
            seen.add(node.uuid)

            if node.is_root():
                if node._parent is not None:
                    raise RuntimeError("RadixPrefixCache integrity check failed: root has parent")
            else:
                if node.length <= 0:
                    raise RuntimeError("RadixPrefixCache integrity check failed: non-root node has empty key")
                if len(node._key) != node.length or len(node._value) != node.length:
                    raise RuntimeError(
                        "RadixPrefixCache integrity check failed: key/value length mismatch"
                    )
                parent = node.parent
                if parent.children.get(self.key_fn(node._key)) is not node:
                    raise RuntimeError(
                        "RadixPrefixCache integrity check failed: parent-child linkage mismatch"
                    )
                if node.ref_count < 0:
                    raise RuntimeError(
                        f"RadixPrefixCache integrity check failed: negative ref_count={node.ref_count}"
                    )
                if node.ref_count == 0:
                    evictable_size += node.length
                else:
                    protected_size += node.length

            stack.extend(node.children.values())

        if evictable_size != self.evictable_size:
            raise RuntimeError(
                "RadixPrefixCache integrity check failed: "
                f"evictable_size={self.evictable_size}, expected {evictable_size}"
            )
        if protected_size != self.protected_size:
            raise RuntimeError(
                "RadixPrefixCache integrity check failed: "
                f"protected_size={self.protected_size}, expected {protected_size}"
            )

    def _tree_walk(self, input_ids: torch.Tensor) -> tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        tic = time.monotonic_ns()
        while prefix_len < indice_len:
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node
            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len
            if match_len != node.length:
                node = node.split_at(match_len)
                return node, prefix_len
            node.timestamp = tic
        return node, prefix_len

    def _collect_evictable_leaves(self) -> list[RadixTreeNode]:
        stack = [self.root_node]
        leaves: list[RadixTreeNode] = []
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if n.ref_count == 0 and not n.is_root():
                    leaves.append(n)
            else:
                stack.extend(n.children.values())
        return leaves
