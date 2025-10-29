from collections.abc import Iterable
import math

import numpy as np
import numpy.typing as npt

_POS_DOTS = [
    " ⢀⢠⢰⢸",
    "⡀⣀⣠⣰⣸",
    "⡄⣄⣤⣴⣼",
    "⡆⣆⣦⣶⣾",
    "⡇⣇⣧⣷⣿",
]

_NEG_DOTS = [
    " ⠈⠘⠸⢸",
    "⠁⠉⠙⠹⢹",
    "⠃⠋⠛⠻⢻",
    "⠇⠏⠟⠿⢿",
    "⡇⡏⡟⡿⣿",
]

_ZERO_CHAR = " "
_NUM_LEVELS = 5


def _qints_to_dots(qs: Iterable[int], dotchars: list[str]) -> str:
    qs = list(qs)
    if len(qs) % 2 != 0:
        qs = qs + [0]

    ret = ""
    for left, right in zip(qs[0::2], qs[1::2]):
        ret += dotchars[left][right]
    return ret


def _quantize_real01(
    xs: npt.NDArray[np.float32], num_levels: int
) -> npt.NDArray[np.int32]:
    return (xs * num_levels).astype(np.int32).clip(min=0, max=num_levels - 1)


def _frame(x: npt.NDArray[np.float32], target_len: int) -> npt.NDArray[np.float32]:
    (wave_len,) = x.shape

    padded_len = int(math.ceil(wave_len / target_len)) * target_len
    x = np.pad(x, (0, padded_len - wave_len))

    return x.reshape((target_len, padded_len // target_len))


def plot_nonnegatives(
    x: npt.NDArray[np.float32],
    max_abs_value: float | None = None,
    num_cols: int = 80,
    bottom: float | None = 0.0,
    top: float | None = None,
) -> str:
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-D array.")
    if (x < 0.0).any():
        raise ValueError("Negative values are passed to plot_nonnegatives")

    if bottom is None:
        bottom = x.min()
    assert bottom is not None
    if top is None:
        top = x.max()
    assert top is not None

    x = (x - bottom) / (top - bottom)
    framed_x = _frame(x, num_cols * 2).mean(axis=-1)
    qs = _quantize_real01(framed_x, _NUM_LEVELS)
    return f"""
{top:.2f}
{_qints_to_dots(qs, _POS_DOTS)}
{bottom:.2f}
""".strip()


def plot_wave(
    x: npt.NDArray[np.float32],
    max_abs_value: float | None = None,
    num_cols: int = 80,
) -> str:
    """
    Plots waveform
    """
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-D array.")

    if max_abs_value is None:
        max_abs_value = np.abs(x).max()
    assert max_abs_value is not None
    x = x / max_abs_value

    framed_x = _frame(x, num_cols * 2)
    upper_envelope_qs = _quantize_real01(
        framed_x.max(axis=-1).clip(min=0.0), _NUM_LEVELS
    )
    lower_envelope_qs = _quantize_real01(
        -framed_x.min(axis=-1).clip(max=0.0), _NUM_LEVELS
    )

    return f"""
+{max_abs_value:.2f}
{_qints_to_dots(upper_envelope_qs, _POS_DOTS)}
{_qints_to_dots(lower_envelope_qs, _NEG_DOTS)}
-{max_abs_value:.2f}
""".strip()
