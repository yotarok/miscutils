import numpy as np
from miscutils import aaplot


def test_plot_sin():
    t = np.linspace(0, 2.0 * np.pi, 500)
    x = np.sin(t)

    s = aaplot.plot_wave(x, num_cols=25)
    print(s)
    expected = """
+1.00
⣀⣤⣶⣿⣿⣿⣿⣿⣿⣶⣤⣀             
             ⠉⠛⠿⣿⣿⣿⣿⣿⣿⠿⠛⠉
-1.00
""".strip()
    assert s == expected

    t = np.linspace(0, 2.0 * np.pi, 500)
    x = np.sin(t) * np.cos(t * 100)

    s = aaplot.plot_wave(x, num_cols=25)
    print(s)
    expected = """
+0.95
 ⣤⣶⣿⣿⣿⣿⣿⣿⣶⣤⣀ ⣀⣤⣶⣿⣿⣿⣿⣿⣶⣶⣤ 
 ⠛⠿⠿⣿⣿⣿⣿⣿⠿⠛⠉ ⠉⠛⠿⣿⣿⣿⣿⣿⣿⠿⠛ 
-0.95
""".strip()
    assert s == expected


def test_plot_sin_amplitude():
    t = np.linspace(0, 2.0 * np.pi, 500)
    x = np.sin(t)

    s = aaplot.plot_amplitudes(x, num_cols=25)
    expected = """
1.00
▂▃▅▇████▇▆▄▃▁▃▄▆▇████▇▅▃▂
""".strip()
    assert s == expected
