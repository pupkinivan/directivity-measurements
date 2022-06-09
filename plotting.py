import matplotlib.pyplot as plt
from measurements import OCTAVE_BANDS, Sensitivity
import numpy as np


def average_directivity_per_octave(directivity_values: np.ndarray[np.float],
                                   directivity_frequencies: np.ndarray[np.float]):
    average_per_octave = np.zeros(len(OCTAVE_BANDS))

    for octave_center in OCTAVE_BANDS:
        freq_inf = octave_center / np.sqrt(2)
        index_freq_inf = np.argmin(np.abs(directivity_frequencies - freq_inf))
        freq_sup = octave_center * np.sqrt(2)
        index_freq_sup = np.argmin(np.abs(directivity_frequencies - freq_sup))

        average_per_octave[np.where(OCTAVE_BANDS == octave_center)[0][0]] = \
            np.mean(directivity_values[index_freq_inf: index_freq_sup])

    return average_per_octave


def plot_sensitivity(sensitivity: Sensitivity, frequencies: np.ndarray[np.float]):
    fontsize_axes = 18

    _ = plt.figure(1, [10, 5])
    plt.semilogx(sensitivity.get_frequencies(), smooth_spectrum(sens.get_frequencies(), sens.get_for_pink(), 3))
    plt.grid()

    x_label1 = [r"$63$", r"$125$", r"$250$", r"$500$", r"$1 k$", r"$2 k$", r"$4 k$", r"$8 k$"]
    plt.xticks(OCTAVE_BANDS, x_label1, fontsize=fontsize_axes)
    plt.yticks(fontsize=fontsize_axes)
    plt.xlim([44, 12000])
    plt.xlabel('Frecuencia [Hz]', fontsize=fontsize_axes)
    plt.ylabel('Nivel de presión [dB SPL]', fontsize=fontsize_axes)
    plt.title('Sensibilidad', fontsize=20)

    plt.savefig('Sensibilidad.png')


def plot_sonogram(directivity_matrix, frequencies):
    fig2 = plt.figure(2, [10, 5])
    ax2 = plt.subplot(111)
    ax2.set_xscale("log")
    plot = plt.contourf(
        frequencies,
        np.arange(13),
        (directivity_matrix.astype('float64')))

    x_label1 = [r"$63$", r"$125$", r"$250$", r"$500$", r"$1 k$", r"$2 k$", r"$4 k$", r"$8 k$"]
    plt.xticks(np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000]), x_label1, fontsize=18)
    plt.xlim([44, 12000])
    y_label = [r"$90^o$", r"$45^o$", r"$0^o$", r"$45^o$", r"$90^o$"]
    plt.yticks(np.array([0, 3, 6, 9, 12]), y_label, fontsize=18)
    plt.xlabel('Frecuencia [Hz]', fontsize=18)
    plt.ylabel('Ángulo [$^o$]', fontsize=18)
    cbar = fig2.colorbar(plot)
    cbar.ax.set_ylabel('Nivel relativo [$dB_{re}$]', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    # plt.title('Sonograma',fontsize=20)
    plt.savefig('Sonograma.png')