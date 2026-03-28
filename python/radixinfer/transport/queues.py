from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Generic, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Legacy mp.Queue helpers (still used by tokenizer_worker simple path)
# ---------------------------------------------------------------------------

@dataclass
class QueuePair(Generic[T]):
    ingress: mp.Queue
    egress: mp.Queue


def make_queue() -> mp.Queue:
    return mp.Queue()


def drain_queue(queue: mp.Queue, limit: int) -> list:
    items = []
    for _ in range(limit):
        try:
            items.append(queue.get_nowait())
        except Empty:
            break
    return items


# ---------------------------------------------------------------------------
# ZMQ-backed queue factories (with mp.Queue fallback when zmq unavailable)
# ---------------------------------------------------------------------------

def make_zmq_push(addr: str, *, create: bool = True, encoder=None):
    """Return a push queue bound/connected to *addr*."""
    from radixinfer.utils.mp import ZmqPushQueue, pickle_encode

    return ZmqPushQueue(addr, create=create, encoder=encoder or pickle_encode)


def make_zmq_pull(addr: str, *, create: bool = True, decoder=None):
    """Return a pull queue bound/connected to *addr*."""
    from radixinfer.utils.mp import ZmqPullQueue, pickle_decode

    return ZmqPullQueue(addr, create=create, decoder=decoder or pickle_decode)


def make_zmq_pair(addr: str):
    """Return (push_queue, pull_queue) sharing *addr* — push binds, pull connects."""
    push = make_zmq_push(addr, create=True)
    pull = make_zmq_pull(addr, create=False)
    return push, pull


def make_zmq_pub(addr: str, *, create: bool = True, encoder=None):
    """Return a PUB socket bound/connected to *addr* (for multi-rank broadcast)."""
    from radixinfer.utils.mp import ZmqPubQueue, pickle_encode

    return ZmqPubQueue(addr, create=create, encoder=encoder or pickle_encode)


def make_zmq_sub(addr: str, *, create: bool = False, decoder=None):
    """Return a SUB socket connected to *addr* (for multi-rank broadcast)."""
    from radixinfer.utils.mp import ZmqSubQueue, pickle_decode

    return ZmqSubQueue(addr, create=create, decoder=decoder or pickle_decode)
