import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_spectrum


OCTAVE_BANDS = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
DIRECTIVITY_ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

np.seterr(divide='ignore')


class Sensitivity:
    def __init__(self, frequencies: np.ndarray[np.float], pink: np.ndarray, f_100_500, f_171):
        self._freq = frequencies
        self._pink = pink
        self._f_100_500 = f_100_500
        self._f_171 = f_171

    def get_frequencies(self): return self._freq

    def get_for_pink(self): return self._pink

    def set_pink(self, new_values: np.ndarray[np.float]): self._pink = new_values

    def get_for_f_100_500(self): return self._f_100_500

    def set_f_100_500(self, new_values: np.ndarray): self._f_100_500 = new_values

    def get_for_f_171(self): return self._f_171

    def set_f_171(self, new_values: np.ndarray[np.float]): self._f_171 = new_values


class InvalidDirectivityDictionaryError(ValueError):
    def __init__(self, message: str):
        self._message = message

    def __str__(self): return self._message


class Directivity:
    def __init__(self, frequencies: np.ndarray[np.float], directivity_dict: dict, coherence: dict):
        self._freq = frequencies
        self._directivity_dict = directivity_dict
        self._coherence = coherence
        try:
            self._directivity_array = np.ndarray([self._directivity_dict[str(angle)] for angle in DIRECTIVITY_ANGLES])
        except object as e:
            raise InvalidDirectivityDictionaryError(
                "Could not instantiate Directivity because of a problem in the directivity dictionary")

    def get_frequencies(self):
        return self._freq

    def get_directivity_at_angle(self, angle: int):
        return self._directivity_dict[str(angle)]

    def get_directivity_array(self):
        return self._directivity_array

    def set_directivity_at_angle(self, angle: int, new_directivity_value: np.float | float):
        self._directivity_dict[str(angle)] = new_directivity_value


class Reference:
    pink = 84
    f_171 = 94.8
    f_100_500 = 87.3


class SmaartMeasurements:
    def __init__(self, sensitivity: Sensitivity = None, directivity: Directivity = None, reference: Reference = None):
        self._sensitivity = sensitivity
        self._directivity = directivity
        self._reference = reference

    def plot_sensitivity(self, sensitivity: Sensitivity, frequencies: np.ndarray[np.float]):
        fig = plt.figure(1, [10, 5])
        sens_suav = smooth_spectrum(sens.get_frequencies(), sens.get_for_pink(), 3)
        plt.semilogx(sensitivity.get_frequencies(), sens_suav)
        plt.grid()
        plt.xlabel('Frecuencia [Hz]', fontsize=18)
        x_label1 = [r"$63$", r"$125$", r"$250$", r"$500$", r"$1 k$", r"$2 k$", r"$4 k$", r"$8 k$"]
        plt.xticks(OCTAVE_BANDS, x_label1, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlim([44, 12000])
        plt.ylabel('Nivel de presión [dB SPL]', fontsize=18)
        plt.title('Sensibilidad', fontsize=20)
        plt.savefig('Sensibilidad.png')

    def average_directivity_per_octave(self):
        average_per_octave = np.zeros(len(OCTAVE_BANDS))

        for octave_center in OCTAVE_BANDS:
            freq_inf = octave_center / np.sqrt(2)
            index_freq_inf = np.argmin(np.abs(self._directivity.get_frequencies() - freq_inf))
            freq_sup = octave_center * np.sqrt(2)
            index_freq_sup = np.argmin(np.abs(self._directivity.get_frequencies() - freq_sup))

            average_per_octave[np.where(OCTAVE_BANDS == octave_center)[0][0]] = \
                np.mean(self._directivity.get_directivity_array()[index_freq_inf: index_freq_sup])

        return average_per_octave

    def plot_sonogram(self, directivity_matrix, frequencies):
        fig = plt.figure(2, [10, 5])
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

        cbar = fig.colorbar(plot)
        cbar.ax.set_ylabel('Nivel relativo [$dB_{re}$]', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.title('Sonograma', fontsize=20)
        plt.savefig('Sonograma.png')
