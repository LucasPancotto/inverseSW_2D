import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from types import SimpleNamespace


def u_latex_sci_notation(x, _, int_=False):
    if x == 0:
        return r"$0$"
    exponent = int(np.floor(np.log10(abs(x))))
    base = x / 10**exponent
    if int_:
        base = int(x / 10**exponent)
        return r"${:d} \times 10^{{{:d}}}$".format(base, exponent)
    else:
        return r"${:.2f} \times 10^{{{:d}}}$".format(base, exponent) # return float base

def h_latex_sci_notation(y, _, int_=False):
    if abs(y - self.h0) < 1e-12:
        return r"$0$"
    val = y - self.h0
    exponent = int(np.floor(np.log10(abs(val))))
    base = val / 10**exponent
    if int_:
        base = int(val / 10**exponent)
        return r"${:d} \times 10^{{{:d}}}$".format(base, exponent)
    else:
        return r"${:.2f} \times 10^{{{:d}}}$".format(base, exponent) # return float base

def h_offset_latex_sci_notation(int_=False):
    if abs(self.h0) < 1e-12:
        return r"$0$"
    val = self.h0
    exponent = int(np.floor(np.log10(abs(val))))

    base = val / 10**exponent
    if int_:
        base = int(val / 10**exponent)
        return r"$\,+\, {:d} \times 10^{{{:d}}}$".format(base, exponent)
    else:
        return r"$\,+\, {:.2f} \times 10^{{{:d}}}$".format(base, exponent) # return float base
