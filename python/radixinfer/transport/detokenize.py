"""Incremental detokenization with per-request state management.

Handles UTF-8 boundaries and CJK characters so streaming output never emits
incomplete multi-byte sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from radixinfer.transport.protocol import DetokenizeRequest


def _is_chinese_char(cp: int) -> bool:
    """Return True if the codepoint is in a CJK Unicode block."""
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


def find_printable_text(text: str) -> str:
    """Return the longest printable prefix that ends on a word boundary."""
    if text.endswith("\n"):
        return text
    if len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    if len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    return text[: text.rfind(" ") + 1]


@dataclass
class DecodeStatus:
    decoded_ids: List[int]
    decoded_str: str
    read_offset: int   # index up to which ids have been fully decoded
    surr_offset: int   # index up to which ids are in the surrogate window
    sent_offset: int   # byte offset of text already sent to the client


class DetokenizeManager:
    """Stateful incremental detokenizer — one instance per tokenizer worker.

    Maintains per-request decode state so that each new token is decoded in
    context (needed for correct multi-byte / BPE token boundaries).
    """

    def __init__(self, tokenizer) -> None:
        # tokenizer must expose batch_decode(list[list[int]]) -> list[str]
        self.decode_map: Dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def detokenize(self, msgs: List[DetokenizeRequest]) -> List[str]:
        """Process a batch of DetokenizeRequests and return incremental text chunks."""
        read_ids: List[List[int]] = []
        surr_ids: List[List[int]] = []

        for msg in msgs:
            if msg.request_id not in self.decode_map:
                self.decode_map[msg.request_id] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.request_id]
            # Append the new token unless it's a trailing EOS on a finished request
            if msg.emit_text and not (msg.finished and msg.token_id == self.eos_token_id):
                s.decoded_ids.append(msg.token_id)
            read_ids.append(s.decoded_ids[s.surr_offset:])
            surr_ids.append(s.decoded_ids[s.surr_offset: s.read_offset])

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map.get(msg.request_id)
            if s is None:
                # Duplicate finished message (overlap scheduling can emit the same
                # request_id in two consecutive batches; the first finished=True
                # already cleaned up this entry — safely skip the stale duplicate).
                incremental_strs.append("")
                continue
            new_text = read_str[len(surr_str):]

            if len(new_text) > 0 and not new_text.endswith("\ufffd"):
                # No surrogate / replacement character — commit the new text
                output_str = s.decoded_str + new_text
                s.decoded_str = output_str
                s.surr_offset = s.read_offset
                s.read_offset = len(s.decoded_ids)
            else:
                # Partial multi-byte sequence — only emit up to a safe boundary
                new_text = find_printable_text(new_text)
                output_str = s.decoded_str + new_text

            incremental_output = output_str[s.sent_offset:] if msg.emit_text else ""
            s.sent_offset = len(output_str)
            incremental_strs.append(incremental_output)

            if msg.finished:
                del self.decode_map[msg.request_id]

        return incremental_strs
