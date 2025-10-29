import math

import numpy as np
import numpy.typing as npt

_NEG_CHARS = [
    " ",
    "⠉",
    "⠛",
    "⠿",
    "⣿",
]
_POS_CHARS = [
    " ",
    "⣀",
    "⣤",
    "⣶",
    "⣿",
]
_ZERO_CHAR = " "
_NUM_LEVELS = 5


def plot_wave(
    x: npt.NDArray[np.float32],
    max_abs_value: float | None = None,
    num_cols: int = 80,
):
    """
    Plots waveform
    """
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-D array.")

    if max_abs_value is None:
        max_abs_value = np.abs(x).max()

    x = x / max_abs_value
    (wave_len,) = x.shape

    padded_len = int(math.ceil(wave_len / num_cols)) * num_cols
    x = np.pad(x, (0, padded_len - wave_len))

    framed_x = x.reshape((num_cols, padded_len // num_cols))
    upper_env = framed_x.max(axis=-1)
    lower_env = framed_x.min(axis=-1)

    pos_zero_plot = ""
    neg_plot = ""
    for ub, lb in zip(upper_env, lower_env):
        q_up = int(ub * _NUM_LEVELS)
        q_down = -int(lb * _NUM_LEVELS)
        if q_up > 0:
            pos_zero_plot += _POS_CHARS[min(_NUM_LEVELS - 1, q_up)]
        else:
            pos_zero_plot += _ZERO_CHAR
        if q_down > 0:
            neg_plot += _NEG_CHARS[min(_NUM_LEVELS - 1, q_down)]
        else:
            neg_plot += _ZERO_CHAR

    return f"""
+{max_abs_value:.2f}
{pos_zero_plot}
{neg_plot}
-{max_abs_value:.2f}
""".strip()


_AMP_CHARS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
_AMP_LEVELS = 8


def plot_amplitudes(
    x: npt.NDArray[np.float32],
    max_abs_value: float | None = None,
    num_cols: int = 80,
):
    """
    Plots (root-mean-squared) amplitudes of waveforms.
    """
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("`x` must be a 1-D array.")
    (wave_len,) = x.shape

    padded_len = int(math.ceil(wave_len / num_cols)) * num_cols
    x = np.pad(x, (0, padded_len - wave_len))

    sampled_x = (x.reshape((num_cols, padded_len // num_cols)) ** 2).mean(
        axis=-1
    ) ** 0.5

    if max_abs_value is None:
        max_abs_value = sampled_x.max()
    sampled_x = sampled_x / max_abs_value

    plot = ""
    for x in sampled_x:
        qx = min(int(abs(x) * _AMP_LEVELS), _AMP_LEVELS - 1)
        plot += _AMP_CHARS[qx]

    return f"""
{max_abs_value:.2f}
{plot}
""".strip()
