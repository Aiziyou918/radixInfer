from __future__ import annotations

import time
from dataclasses import dataclass, field

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
        node.ref_count += 1
        node.timestamp_ns = time.monotonic_ns()

    def unlock(self, key: PrefixCacheKey | None) -> None:
        node = self._get_node(key)
        if node is None:
            return
        if node.ref_count <= 0:
            raise ValueError("prefix entry ref_count underflow")
        node.ref_count -= 1
        node.timestamp_ns = time.monotonic_ns()

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
        parent.children[_key_fn(split_node.token_chunk, self.page_size)] = split_node
        node.parent = split_node
        node.token_chunk = node.token_chunk[prefix_len:]
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
        del self._nodes[node.node_id]

    def _allocate_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id
