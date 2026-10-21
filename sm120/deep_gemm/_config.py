"""Process-wide configuration knobs of DeepGEMM (vendor csrc/apis/config.hpp + layout runtime)."""

from __future__ import annotations

from . import _env

_STATE = {
    "num_sms": None,
    "tc_util": 100,
    "pdl": False,
    "deterministic": False,
    "ignore_compile_dims": False,
    "block_size_multiple_of": 1,
    "mk_alignment": None,
}


def _device_sms() -> int:
    import torch

    if not torch.cuda.is_available():
        return 1
    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)


def init(path: str | None = None) -> None:  # noqa: ARG001
    pass


def shutdown() -> None:
    pass


def set_num_sms(num_sms: int) -> None:
    n = int(num_sms)
    if n <= 0:
        raise ValueError(f"set_num_sms: {n} must be positive")
    _STATE["num_sms"] = min(n, _device_sms())


def get_num_sms() -> int:
    if _STATE["num_sms"] is None:
        _STATE["num_sms"] = _device_sms()
    return int(_STATE["num_sms"])


def set_tc_util(v: int) -> None:
    _STATE["tc_util"] = int(v)


def get_tc_util() -> int:
    return int(_STATE["tc_util"])


def set_ignore_compile_dims(v: bool) -> None:
    _STATE["ignore_compile_dims"] = bool(v)


def set_block_size_multiple_of(v: int) -> None:
    _STATE["block_size_multiple_of"] = int(v)


def set_pdl(v: bool) -> None:
    _STATE["pdl"] = bool(v)


def get_pdl() -> bool:
    return bool(_STATE["pdl"])


def use_deterministic_algorithms(v: bool = True) -> None:
    # Every shim path is deterministic for a fixed shape (cuBLAS default algorithms / b12x
    # non-split-K atomics are not used for the linear path's default config).
    _STATE["deterministic"] = bool(v)


def is_deterministic() -> bool:
    return bool(_STATE["deterministic"])


def set_mk_alignment_for_contiguous_layout(v: int) -> None:
    _STATE["mk_alignment"] = int(v)


def get_mk_alignment_for_contiguous_layout() -> int:
    v = _STATE["mk_alignment"]
    return int(v) if v is not None else _env.mk_alignment()


def get_theoretical_mk_alignment_for_contiguous_layout(expected_m=None) -> int:
    del expected_m
    return _env.mk_alignment()  # DeepGEMM: 128 on every non-SM100 arch (kLegacyMKAlignment...)
