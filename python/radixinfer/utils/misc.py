from __future__ import annotations


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers, asserting exact division.
    If allow_replicate=True, allows b > a when b % a == 0, returning 1.
    """
    if allow_replicate and b > a:
        assert b % a == 0, f"{b=} must be divisible by {a=} for KV head replication"
        return 1
    assert a % b == 0, f"{a=} must be divisible by {b=}"
    return a // b


def div_ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    return (a // b) * b


class Unset:
    pass


UNSET = Unset()
