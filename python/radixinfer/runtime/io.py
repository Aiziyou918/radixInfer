from __future__ import annotations

import queue
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from radixinfer.runtime.scheduler_config import SchedulerConfig


class SchedulerIOMixin:
    """Mixin that sets up the scheduler's external message I/O."""

    def __init__(self, config: SchedulerConfig, tp_cpu_group=None) -> None:
        self._tp_cpu_group = tp_cpu_group
        tp_info = getattr(config, "tp_info", None)
        tp_rank = tp_info.rank if tp_info else 0
        tp_size = tp_info.size if tp_info else 1
        is_primary = tp_rank == 0

        if getattr(config, "offline_mode", False):
            # Tests / inline API — caller injects queues via _in_queue / _out_queue
            self._in_queue: queue.Queue = queue.Queue()
            self._out_queue: queue.Queue = queue.Queue()
            self.receive_msg = self._offline_receive_msg
            self.send_result = self._offline_send_result
            return

        from radixinfer.transport.queues import (
            make_zmq_pub,
            make_zmq_pull,
            make_zmq_push,
            make_zmq_sub,
        )

        if is_primary:
            self._recv_from_tokenizer = make_zmq_pull(
                config.zmq_backend_addr, create=True
            )
            self._send_to_detokenizer = make_zmq_push(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
            )

        if tp_size > 1:
            if is_primary:
                self._send_into_ranks = make_zmq_pub(
                    config.zmq_scheduler_broadcast_addr, create=True
                )
                self.receive_msg = self._recv_msg_multi_rank0
            else:
                self._recv_from_rank0 = make_zmq_sub(
                    config.zmq_scheduler_broadcast_addr, create=False
                )
                self.receive_msg = self._recv_msg_multi_rank1
                self.send_result = self._send_result_rank1
                return
        else:
            self.receive_msg = self._recv_msg_single_rank

        self.send_result = self._send_result_rank0

    # ------------------------------------------------------------------
    # Offline (queue.Queue) implementations
    # ------------------------------------------------------------------

    def _offline_receive_msg(self, blocking: bool = False) -> list:
        msgs = []
        if blocking:
            self.run_when_idle()
            try:
                msgs.append(self._in_queue.get(timeout=0.5))
            except queue.Empty:
                return msgs
        while not self._in_queue.empty():
            try:
                msgs.append(self._in_queue.get_nowait())
            except queue.Empty:
                break
        return msgs

    def _offline_send_result(self, results: list) -> None:
        for r in results:
            self._out_queue.put_nowait(r)

    # ------------------------------------------------------------------
    # ZMQ single-rank implementations
    # ------------------------------------------------------------------

    def _recv_msg_single_rank(self, blocking: bool = False) -> list:
        msgs = []
        if blocking:
            self.run_when_idle()
            msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            msgs.append(self._recv_from_tokenizer.get())
        return msgs

    def _send_result_rank0(self, results: list) -> None:
        if not results:
            return
        from radixinfer.transport.protocol import BatchDetokenizeRequest

        if len(results) == 1:
            self._send_to_detokenizer.put(results[0])
        else:
            self._send_to_detokenizer.put(BatchDetokenizeRequest(requests=results))

    # ------------------------------------------------------------------
    # ZMQ multi-rank implementations
    # ------------------------------------------------------------------

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> list:
        import torch

        msgs = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()
            self._send_into_ranks.put_raw(raw)
            msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw: list = []
        while not self._recv_from_tokenizer.empty():
            pending_raw.append(self._recv_from_tokenizer.get_raw())

        # Broadcast count so other ranks know how many messages to receive
        count_tensor = torch.tensor(len(pending_raw))
        if self._tp_cpu_group is not None:
            self._tp_cpu_group.broadcast(count_tensor, root=0).wait()

        for raw in pending_raw:
            self._send_into_ranks.put_raw(raw)
            msgs.append(self._recv_from_tokenizer.decode(raw))
        return msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> list:
        import torch

        msgs = []
        if blocking:
            self.run_when_idle()
            msgs.append(self._recv_from_rank0.get())

        dst_tensor = torch.tensor(-1)
        if self._tp_cpu_group is not None:
            self._tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        count = int(dst_tensor.item())

        for _ in range(count):
            msgs.append(self._recv_from_rank0.get())
        return msgs

    def _send_result_rank1(self, results: list) -> None:
        pass  # non-primary ranks do not send results to the detokenizer

    # ------------------------------------------------------------------
    # Synchronisation helpers
    # ------------------------------------------------------------------

    def sync_all_ranks(self) -> None:
        """CPU-side barrier across all TP ranks."""
        if self._tp_cpu_group is not None:
            self._tp_cpu_group.barrier().wait()

    def run_when_idle(self) -> None:
        """Called while blocking-waiting for the next message (hook for subclasses)."""
