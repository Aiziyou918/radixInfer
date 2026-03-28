from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class QueuePair(Generic[T]):
    ingress: mp.Queue[T]
    egress: mp.Queue[T]


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
