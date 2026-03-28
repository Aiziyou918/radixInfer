"""ZMQ-based queue primitives for inter-process communication.

Provides push/pull and pub/sub patterns with optional async variants.
Falls back to multiprocessing.Queue when zmq is not installed.

NOTE: The _FallbackQueue wraps multiprocessing.Queue so it is safe to share
across processes (e.g. passed via Process(args=...)).  Do NOT use queue.Queue
here — that only works within a single process.
"""
from __future__ import annotations

import multiprocessing as mp
import threading
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class _FallbackQueue(Generic[T]):
    """multiprocessing.Queue fallback when ZMQ is unavailable.

    Uses multiprocessing.Queue so it can be safely shared across processes.
    """

    def __init__(self, _addr: str, *, create: bool = True, encoder=None, decoder=None):
        self._q: mp.Queue = mp.Queue()
        self._encoder: Callable = encoder or (lambda x: x)
        self._decoder: Callable = decoder or (lambda x: x)

    def put(self, item: T) -> None:
        self._q.put(self._encoder(item))

    def put_raw(self, raw: bytes) -> None:
        self._q.put(raw)

    def get(self) -> T:
        return self._decoder(self._q.get())

    def get_raw(self) -> bytes:
        return self._q.get()

    def get_nowait(self) -> T:
        import queue as _q

        return self._decoder(self._q.get_nowait())

    def decode(self, raw: bytes) -> T:
        return self._decoder(raw)

    def empty(self) -> bool:
        return self._q.empty()


def _make_queue(addr: str, *, create: bool, encoder=None, decoder=None, socket_type: str):
    try:
        import zmq

        ctx = zmq.Context.instance()
        if socket_type == "push":
            sock = ctx.socket(zmq.PUSH)
            if create:
                sock.bind(addr)
            else:
                sock.connect(addr)
        elif socket_type == "pull":
            sock = ctx.socket(zmq.PULL)
            if create:
                sock.bind(addr)
            else:
                sock.connect(addr)
        elif socket_type == "pub":
            sock = ctx.socket(zmq.PUB)
            if create:
                sock.bind(addr)
            else:
                sock.connect(addr)
        elif socket_type == "sub":
            sock = ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            if create:
                sock.bind(addr)
            else:
                sock.connect(addr)
        else:
            raise ValueError(f"Unknown socket_type: {socket_type}")

        return _ZmqQueue(sock, encoder=encoder, decoder=decoder)

    except ImportError:
        return _FallbackQueue(addr, create=create, encoder=encoder, decoder=decoder)


class _ZmqQueue(Generic[T]):
    def __init__(self, socket, *, encoder=None, decoder=None):
        self._sock = socket
        self._encoder: Callable = encoder or (lambda x: x)
        self._decoder: Callable = decoder or (lambda x: x)

    def put(self, item: T) -> None:
        self._sock.send(self._encoder(item))

    def put_raw(self, raw: bytes) -> None:
        self._sock.send(raw)

    def get(self) -> T:
        return self._decoder(self._sock.recv())

    def get_nowait(self) -> T:
        import zmq

        if not self._sock.poll(0, zmq.POLLIN):
            import queue as _q
            raise _q.Empty
        return self._decoder(self._sock.recv(zmq.NOBLOCK))

    def get_raw(self) -> bytes:
        return self._sock.recv()

    def decode(self, raw: bytes) -> T:
        return self._decoder(raw)

    def empty(self) -> bool:
        import zmq

        return not self._sock.poll(0, zmq.POLLIN)


def ZmqPushQueue(addr: str, *, create: bool = True, encoder=None):
    return _make_queue(addr, create=create, encoder=encoder, socket_type="push")


def ZmqPullQueue(addr: str, *, create: bool = True, decoder=None):
    return _make_queue(addr, create=create, decoder=decoder, socket_type="pull")


def ZmqPubQueue(addr: str, *, create: bool = True, encoder=None):
    return _make_queue(addr, create=create, encoder=encoder, socket_type="pub")


def ZmqSubQueue(addr: str, *, create: bool = False, decoder=None):
    return _make_queue(addr, create=create, decoder=decoder, socket_type="sub")
