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
    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: dict[Any, RadixTreeNode] = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.timestamp = tic or time.monotonic_ns()
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if len(key) != len(value):
            raise ValueError(
                f"RadixTreeNode key/value length mismatch: {len(key)} != {len(value)}"
            )
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
        if self._parent is None:
            raise RuntimeError("RadixTreeNode parent is not set")
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
        if not (0 < pos < self.length):
            raise ValueError(
                f"split_at expects 0 < pos < {self.length}, got {pos}"
            )
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
    """Radix-tree prefix cache with LRU eviction.

    Eviction uses a min-heap of (timestamp, node) tuples over evictable leaf
    nodes.  Heap entries are validated lazily on pop so that timestamp updates
    (from cache hits in _tree_walk) and lock/unlock transitions do not require
    O(log n) heap updates on every access — only O(log n) on eviction.

    Complexity:
      match_prefix  – O(depth * page_size)  (key_fn tuple allocation per level)
      insert_prefix – O(depth + log h)      (h = heap size)
      lock_handle   – O(depth * log h)
      evict(k)      – O(k log h)  amortised (lazy deletions bounded by inserts)
    """

    def __init__(self, device: torch.device, page_size: int) -> None:
        self.device = device
        self.page_size = page_size
        self.key_fn = _get_key_fn(self.page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.evictable_size = 0
        self.protected_size = 0
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

        # Min-heap of (timestamp_at_push, node).  Entries become stale when a
        # node is re-locked or its timestamp is refreshed by a cache hit; they
        # are discarded lazily when popped during eviction.
        self._evictable_heap: list[tuple[int, RadixTreeNode]] = []

    def _push_evictable(self, node: RadixTreeNode) -> None:
        heapq.heappush(self._evictable_heap, (node.timestamp, node))

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        if not isinstance(handle, RadixCacheHandle):
            raise TypeError(
                f"RadixPrefixCache expected RadixCacheHandle, got {type(handle).__name__}"
            )
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                if node.ref_count < 0:
                    raise RuntimeError(
                        f"RadixPrefixCache ref_count underflow on node uuid={id(node)}"
                    )
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                    if node.is_leaf():
                        self._push_evictable(node)
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                    # Stale heap entry will be discarded lazily on next eviction.
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
            self._push_evictable(new_node)
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        if size > self.evictable_size:
            raise RuntimeError(
                f"Cannot evict {size}, only {self.evictable_size} is evictable"
            )
        evicted: list[torch.Tensor] = []
        evicted_size = 0
        while evicted_size < size:
            if not self._evictable_heap:
                raise RuntimeError(
                    f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
                )
            ts, node = heapq.heappop(self._evictable_heap)
            # Lazy validity check: skip stale entries.
            # A heap entry is stale when:
            #   - the node was re-locked (ref_count > 0), or
            #   - the node's timestamp was refreshed by a cache hit (ts mismatch), or
            #   - the node gained children after insertion (no longer a leaf).
            if ts != node.timestamp or node.ref_count != 0 or not node.is_leaf() or node.is_root():
                continue

            evicted_size += node.length
            evicted.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                self._push_evictable(parent)

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
            if id(node) in seen:
                raise RuntimeError("RadixPrefixCache integrity check failed: cycle detected")
            seen.add(id(node))

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
            # Full match: refresh timestamp for LRU.  If the node is an evictable
            # leaf, push a fresh heap entry; the stale entry will be skipped lazily.
            node.timestamp = tic
            if node.ref_count == 0 and node.is_leaf():
                self._push_evictable(node)
        return node, prefix_len
