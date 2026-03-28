from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from .page_pool import PageSpan


def _normalize_prefix(tokens: list[int], page_size: int) -> tuple[int, ...]:
    prefix_len = len(tokens) - (len(tokens) % page_size)
    return tuple(tokens[:prefix_len])


def _align_down(value: int, alignment: int) -> int:
    return value - (value % alignment)


def _shared_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    matched = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        matched += 1
    return matched


def _key_fn(tokens: tuple[int, ...], page_size: int) -> tuple[int, ...] | int:
    if page_size == 1:
        return tokens[0]
    return tokens[:page_size]


@dataclass(frozen=True)
class PrefixCacheKey:
    node_id: int


@dataclass
class PrefixHit:
    matched_tokens: int
    cached_span: PageSpan | None = None
    cache_key: PrefixCacheKey | None = None


class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size


@dataclass
class RadixNode:
    node_id: int
    token_chunk: tuple[int, ...]
    full_span: PageSpan
    parent: RadixNode | None = None
    ref_count: int = 0
    timestamp_ns: int = field(default_factory=time.monotonic_ns)
    children: dict[tuple[int, ...] | int, RadixNode] = field(default_factory=dict)

    @property
    def total_length(self) -> int:
        return self.full_span.token_count

    @property
    def length(self) -> int:
        return len(self.token_chunk)

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def evictable(self) -> bool:
        return self.ref_count == 0 and not self.is_root


class PrefixStore:
    def __init__(self, capacity: int, page_size: int) -> None:
        self.capacity = capacity
        self.page_size = page_size
        self._next_node_id = 1
        self._root = RadixNode(node_id=0, token_chunk=(), full_span=PageSpan(page_ids=(), token_count=0))
        self._nodes: dict[int, RadixNode] = {self._root.node_id: self._root}
        self._evictable_size = 0
        self._protected_size = 0

    def match(self, tokens: list[int]) -> PrefixHit:
        normalized = _normalize_prefix(tokens, self.page_size)
        node, matched = self._tree_walk(normalized)
        if node.is_root:
            return PrefixHit(matched_tokens=0)
        node.timestamp_ns = time.monotonic_ns()
        return PrefixHit(
            matched_tokens=matched,
            cached_span=node.full_span,
            cache_key=PrefixCacheKey(node.node_id),
        )

    def lock(self, key: PrefixCacheKey | None) -> None:
        node = self._get_node(key)
        if node is None:
            return
        self._update_lock_state(node, unlock=False)

    def unlock(self, key: PrefixCacheKey | None) -> None:
        node = self._get_node(key)
        if node is None:
            return
        self._update_lock_state(node, unlock=True)

    def insert(self, tokens: list[int], span: PageSpan) -> tuple[PrefixCacheKey | None, list[PageSpan]]:
        normalized = _normalize_prefix(tokens, self.page_size)
        if not normalized:
            return None, []

        node, prefix_len = self._tree_walk(normalized)
        if prefix_len == len(normalized) and node.total_length == len(normalized):
            node.timestamp_ns = time.monotonic_ns()
            return PrefixCacheKey(node.node_id), [span]

        if prefix_len != len(normalized):
            suffix_tokens = normalized[prefix_len:]
            suffix_pages = span.page_ids[prefix_len // self.page_size :]
            node = self._create_child(
                parent=node,
                token_chunk=suffix_tokens,
                full_span=PageSpan(page_ids=tuple(span.page_ids), token_count=len(normalized)),
            )

        node.timestamp_ns = time.monotonic_ns()
        evicted = self._evict_over_capacity()
        return PrefixCacheKey(node.node_id), evicted

    def entry_ref_count(self, key: PrefixCacheKey) -> int:
        return self._nodes[key.node_id].ref_count

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self._evictable_size,
            protected_size=self._protected_size,
        )

    def evict(self, size: int) -> list[PageSpan]:
        if size <= 0:
            return []
        if size > self._evictable_size:
            raise ValueError(
                f"cannot evict {size} tokens, only {self._evictable_size} tokens are evictable"
            )
        evicted: list[PageSpan] = []
        evicted_size = 0
        while evicted_size < size:
            candidate = self._find_evictable_leaf()
            if candidate is None:
                raise RuntimeError(
                    f"failed to evict enough tokens: need {size}, evicted {evicted_size}"
                )
            evicted.append(candidate.full_span)
            evicted_size += candidate.length
            self._remove_leaf(candidate)
        return evicted

    def _get_node(self, key: PrefixCacheKey | None) -> RadixNode | None:
        if key is None:
            return None
        return self._nodes.get(key.node_id)

    def _tree_walk(self, tokens: tuple[int, ...]) -> tuple[RadixNode, int]:
        matched = 0
        node = self._root
        now = time.monotonic_ns()
        while matched < len(tokens):
            child = node.children.get(_key_fn(tokens[matched:], self.page_size))
            if child is None:
                return node, matched
            chunk_match = _shared_prefix_len(child.token_chunk, tokens[matched:])
            chunk_match = _align_down(chunk_match, self.page_size)
            if chunk_match == 0:
                return node, matched
            matched += chunk_match
            if chunk_match != len(child.token_chunk):
                return self._split_node(child, chunk_match), matched
            child.timestamp_ns = now
            node = child
        return node, matched

    def _split_node(self, node: RadixNode, prefix_len: int) -> RadixNode:
        if prefix_len <= 0 or prefix_len >= len(node.token_chunk):
            raise ValueError("invalid split length")
        parent = node.parent
        if parent is None:
            raise ValueError("cannot split root")
        split_total = node.total_length - len(node.token_chunk) + prefix_len
        split_pages = split_total // self.page_size
        split_node = RadixNode(
            node_id=self._allocate_node_id(),
            token_chunk=node.token_chunk[:prefix_len],
            full_span=PageSpan(
                page_ids=node.full_span.page_ids[:split_pages],
                token_count=split_total,
            ),
            parent=parent,
            ref_count=node.ref_count,
            timestamp_ns=node.timestamp_ns,
        )
        if split_node.ref_count == 0:
            self._evictable_size += split_node.length
        else:
            self._protected_size += split_node.length
        parent.children[_key_fn(split_node.token_chunk, self.page_size)] = split_node
        node.parent = split_node
        original_length = node.length
        node.token_chunk = node.token_chunk[prefix_len:]
        if node.ref_count == 0:
            self._evictable_size -= prefix_len
        else:
            self._protected_size -= prefix_len
        assert node.length == original_length - prefix_len
        split_node.children[_key_fn(node.token_chunk, self.page_size)] = node
        self._nodes[split_node.node_id] = split_node
        return split_node

    def _create_child(self, parent: RadixNode, token_chunk: tuple[int, ...], full_span: PageSpan) -> RadixNode:
        node = RadixNode(
            node_id=self._allocate_node_id(),
            token_chunk=token_chunk,
            full_span=full_span,
            parent=parent,
        )
        parent.children[_key_fn(token_chunk, self.page_size)] = node
        self._nodes[node.node_id] = node
        self._evictable_size += node.length
        return node

    def _evict_over_capacity(self) -> list[PageSpan]:
        evicted: list[PageSpan] = []
        while len(self._nodes) - 1 > self.capacity:
            candidate = self._find_evictable_leaf()
            if candidate is None:
                break
            evicted.append(candidate.full_span)
            self._remove_leaf(candidate)
        return evicted

    def _find_evictable_leaf(self) -> RadixNode | None:
        best: RadixNode | None = None
        for node in self._nodes.values():
            if not node.evictable or not node.is_leaf:
                continue
            if best is None or node.timestamp_ns < best.timestamp_ns:
                best = node
        return best

    def _remove_leaf(self, node: RadixNode) -> None:
        parent = node.parent
        if parent is None:
            raise ValueError("cannot remove root")
        del parent.children[_key_fn(node.token_chunk, self.page_size)]
        if node.ref_count == 0:
            self._evictable_size -= node.length
        else:
            self._protected_size -= node.length
        del self._nodes[node.node_id]

    def _allocate_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _update_lock_state(self, node: RadixNode, *, unlock: bool) -> None:
        current = node
        now = time.monotonic_ns()
        while not current.is_root:
            if unlock:
                if current.ref_count <= 0:
                    raise ValueError("prefix entry ref_count underflow")
                current.ref_count -= 1
                if current.ref_count == 0:
                    self._protected_size -= current.length
                    self._evictable_size += current.length
            else:
                if current.ref_count == 0:
                    self._evictable_size -= current.length
                    self._protected_size += current.length
                current.ref_count += 1
            current.timestamp_ns = now
            assert current.parent is not None
            current = current.parent

    def check_integrity(self) -> None:
        evictable = 0
        protected = 0
        for node in self._nodes.values():
            if node.is_root:
                continue
            if node.ref_count == 0:
                evictable += node.length
            else:
                protected += node.length
        if evictable != self._evictable_size or protected != self._protected_size:
            raise RuntimeError(
                "PrefixStore accounting mismatch: "
                f"expected evictable={evictable}, protected={protected}, "
                f"got evictable={self._evictable_size}, protected={self._protected_size}"
            )


# ---------------------------------------------------------------------------
# mini-sglang compatible prefix cache interface
# ---------------------------------------------------------------------------

class BaseCacheHandle(ABC):
    """Abstract handle returned by prefix cache match/insert operations."""

    @property
    @abstractmethod
    def cached_len(self) -> int: ...

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...


class InsertResult(NamedTuple):
    cached_len: int           # tokens already cached before this insert (to be freed)
    handle: BaseCacheHandle   # handle pointing to the newly inserted node


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

    @abstractmethod
    def reset(self) -> None: ...

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo: ...

    @abstractmethod
    def check_integrity(self) -> None: ...


# ---------------------------------------------------------------------------
# RadixTreeNode — torch-tensor-backed radix tree node
# ---------------------------------------------------------------------------

class RadixTreeNode:
    _counter: int = 0

    def __init__(self, key_fn, tic: int | None = None) -> None:
        self.key_fn = key_fn
        self.children: dict = {}
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0
        self.uuid: int = RadixTreeNode._counter
        RadixTreeNode._counter += 1
        self.timestamp: int = tic if tic is not None else time.monotonic_ns()
        # Set later via set_key_value
        self._key: torch.Tensor
        self._value: torch.Tensor
        self._length: int = 0

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
    def value(self) -> torch.Tensor:
        return self._value

    @property
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None
        return self._parent

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
    _cached_len: int
    node: RadixTreeNode

    @property
    def cached_len(self) -> int:
        return self._cached_len

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
# RadixPrefixCache — mini-sglang compatible BasePrefixCache implementation
# ---------------------------------------------------------------------------

def _get_key_fn(page_size: int):
    if page_size == 1:
        return lambda x: int(x[0].item())
    return lambda x: tuple(x[:page_size].tolist())


class RadixPrefixCache(BasePrefixCache):
    """Radix-tree prefix cache using torch tensors for keys/values.

    Implements the same interface as mini-sglang's RadixPrefixCache so that
    CacheManager can use it without modification.
    """

    def __init__(self, device: torch.device) -> None:
        from radixinfer.core import get_global_ctx
        from radixinfer.utils import align_down as _align_down

        ctx = get_global_ctx()
        self.device = device
        self.page_size = ctx.page_size
        self._key_fn = _get_key_fn(self.page_size)
        self._align_down = _align_down
        self._empty = torch.empty(0, dtype=torch.int32, device=device)
        self._evictable_size = 0
        self._protected_size = 0
        self._root = RadixTreeNode(self._key_fn)
        self._root.ref_count = 1  # root is always protected

    # ------------------------------------------------------------------
    # BasePrefixCache interface
    # ------------------------------------------------------------------

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                if node.ref_count == 0:
                    self._evictable_size += node.length
                    self._protected_size -= node.length
                node = node.parent
        else:
            while not node.is_root():
                if node.ref_count == 0:
                    self._evictable_size -= node.length
                    self._protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        ids = input_ids.cpu().to(torch.int32)
        node, prefix_len = self._tree_walk(ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = self._align_down(len(input_ids), self.page_size)
        input_ids = input_ids[:insert_len].cpu().to(torch.int32)
        indices = indices[:insert_len].cpu().to(torch.int32)

        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self._key_fn)
            new_node.set_key_value(
                input_ids[prefix_len:].clone(),
                indices[prefix_len:].clone(),
            )
            new_node.set_parent(node)
            self._evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self._empty
        assert size <= self._evictable_size, (
            f"Cannot evict {size}, only {self._evictable_size} is evictable"
        )

        leaf_nodes = self._collect_evictable_leaves()
        heapq.heapify(leaf_nodes)
        evicted: list[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert leaf_nodes, (
                f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            )
            node = heapq.heappop(leaf_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted.append(node.value)
            self._evictable_size -= node.length
            parent = node.parent
            del parent.children[self._key_fn(node._key)]
            if parent.is_leaf() and parent.ref_count == 0 and not parent.is_root():
                heapq.heappush(leaf_nodes, parent)

        return torch.cat(evicted).to(self.device)

    def reset(self) -> None:
        raise NotImplementedError("RadixPrefixCache.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(
            evictable_size=self._evictable_size,
            protected_size=self._protected_size,
        )

    def check_integrity(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tree_walk(self, input_ids: torch.Tensor) -> tuple[RadixTreeNode, int]:
        prefix_len = 0
        indice_len = len(input_ids)
        node = self._root
        tic = time.monotonic_ns()

        while prefix_len < indice_len:
            child_node = node.children.get(self._key_fn(input_ids[prefix_len:]))
            if child_node is None:
                return node, prefix_len
            node = child_node

            match_len = node.get_match_len(input_ids[prefix_len:])
            match_len = self._align_down(match_len, self.page_size)
            prefix_len += match_len

            if match_len != node.length:
                node = node.split_at(match_len)
                return node, prefix_len

            node.timestamp = tic

        return node, prefix_len

    def _collect_evictable_leaves(self) -> list[RadixTreeNode]:
        stack = [self._root]
        leaves: list[RadixTreeNode] = []
        while stack:
            n = stack.pop()
            if n.is_leaf():
                if n.ref_count == 0 and not n.is_root():
                    leaves.append(n)
            else:
                stack.extend(n.children.values())
        return leaves
