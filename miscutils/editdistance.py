from collections.abc import Sequence
import typing
import Levenshtein

import torch

from . import parutil

T = typing.TypeVar("T")


class ErrorCounts(typing.NamedTuple):
    ref_count: int
    sub_count: int
    del_count: int
    ins_count: int

    @classmethod
    def zero(cls):
        return cls(ref_count=0, sub_count=0, del_count=0, ins_count=0)

    @classmethod
    def compute(cls, ref: Sequence[T], hyp: Sequence[T]):
        sub_count = 0
        del_count = 0
        ins_count = 0
        for opname, src_begin, src_end, dst_begin, dst_end in Levenshtein.opcodes(
            ref, hyp
        ):
            if opname == "equal":
                continue
            elif opname == "delete":
                del_count += src_end - src_begin
            elif opname == "insert":
                ins_count += dst_end - dst_begin
            elif opname == "replace":
                sub_count += src_end - src_begin
            else:
                assert False, "Unknown opcode emitted from `Levenshtein.opcodes`"
        return cls(
            ref_count=len(ref),
            sub_count=sub_count,
            del_count=del_count,
            ins_count=ins_count,
        )

    @property
    def error_rate(self) -> float:
        return (self.sub_count + self.del_count + self.ins_count) / self.ref_count

    def add(self, other: "ErrorCounts") -> "ErrorCounts":
        return type(self)(
            ref_count=self.ref_count + other.ref_count,
            sub_count=self.sub_count + other.sub_count,
            del_count=self.del_count + other.del_count,
            ins_count=self.ins_count + other.ins_count,
        )

    def all_reduce_torch(self) -> "ErrorCounts":
        counts = torch.tensor(
            [self.ref_count, self.sub_count, self.del_count, self.ins_count],
            dtype=torch.int64,
        )
        counts = parutil.all_reduce(counts, "sum")
        return type(self)(
            ref_count=counts[0].item(),  # type: ignore
            sub_count=counts[1].item(),  # type: ignore
            del_count=counts[2].item(),  # type: ignore
            ins_count=counts[3].item(),  # type: ignore
        )
