# -*- coding: utf-8 -*-
from typing import List

import numpy as np

from errors import InvalidOctaveDividerError


def smooth_spectrum(frequencies: np.ndarray[np.float], amplitudes: np.ndarray[np.float], octave_divider: int):
    """ Bucket a power spectrum into a new one with smaller octave fractions as frequency axis.

    :param frequencies: np.ndarray[np.float]. An array of frequencies of the spectrum, sorted in ascending order.
    :param amplitudes: np.ndarray[np.float]. The array of power magnitudes in dBFS.
    :param octave_divider: int. The octave divider, e.g. 3 for third-octave bucketting. It must be greater than 0, else
    it raises an InvalidOctaveDividerError.

    :return: np.ndarray[np.float]. The new power spectrum in magnitude (dBFS).
    """
    amplitudes_smooth = amplitudes.copy()

    if octave_divider == 0:
        raise InvalidOctaveDividerError(f"{octave_divider.__str__()} is not a valid octave fraction")

    for n in range(1, len(frequencies)):
        freq_sup = frequencies[n] * pow(2, 1 / (2 * octave_divider))
        freq_inf = frequencies[n] * pow(2, 1 / (2 * octave_divider))

        if frequencies[-1] <= freq_sup:
            index_sup = len(frequencies) - n
        else:
            index_sup = np.argmin(abs(frequencies[n:] - freq_sup))

        if frequencies[1] <= freq_inf:
            index_inf = np.argmin(abs(frequencies[0:n + 1] - freq_inf))
        else:
            index_inf = 0

        if index_sup != index_inf:
            temp = pow(10, amplitudes[index_inf:index_sup + n - 1] * 0.1)
            amplitudes_smooth[n] = 10 * np.log10(sum(temp) / (index_sup + n - index_inf))

    return amplitudes_smooth
