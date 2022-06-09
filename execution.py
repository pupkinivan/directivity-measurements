# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from measurements import average_directivity_per_octave, Directivity, plot_sensitivity, plot_sonogram, Reference, \
    Sensitivity, SmaartMeasurements

# Read the XLSX files
sens_df = pd.read_excel('Sensibilidad y Directividad - Curvas Smaart.xlsx', sheet_name='Sensibilidad')
sens_values = sens_df.to_numpy()

direc_df = pd.read_excel('Sensibilidad y Directividad - Curvas Smaart.xlsx', sheet_name='Directividad')
direc_values = direc_df.to_numpy()

# Instantiate classes
sens = Sensitivity(sens_values[2:, 0], sens_values[2:, 1], sens_values[2:, 4], sens_values[2:, 7])
directivity_values_dict = {
    "0": direc_values[1:, 1],
    "15": direc_values[1:, 6],
    "30": direc_values[1:, 11],
    "45": direc_values[1:, 16],
    "60": direc_values[1:, 21],
    "75": direc_values[1:, 26],
    "90": direc_values[1:, 31],
}
coherence_values_dict = {
    "0": direc_values[1:, 3],
    "15": direc_values[1:, 8],
    "30": direc_values[1:, 13],
    "45": direc_values[1:, 18],
    "60": direc_values[1:, 23],
    "75": direc_values[1:, 28],
    "90": direc_values[1:, 33],
}
direc = Directivity(direc_values[1:, 0], directivity_values_dict, coherence_values_dict)
ref = Reference()
measurements = SmaartMeasurements(sensitivity=sens, directivity=direc, reference=ref)

# =============================================================================
# CORRECCION DE LAS CURVAS DE SENSIBILIDAD y DIRECTIVIDAD
# =============================================================================

# Valor más cercano en frecuencia a 171 Hz. TODO por qué 171 Hz?!
dif = np.abs(sens.get_frequencies() - 171)
index_closest_171 = dif.argmin(dif)

# Diferencia entre la referencia y la curva del Smaart
dif_dBSPL = ref.f_171 - sens.get_for_f_171()[index_closest_171]

# Corrección de sensibilidad tomando como referencia los dBSPL a 171 Hz
# Correción de calibración
sens.set_f_171(sens.get_for_f_171() + dif_dBSPL)
sens.set_f_100_500(sens.get_for_f_100_500() + dif_dBSPL)
sens.set_pink(sens.get_for_pink() + dif_dBSPL)

# Correción energética
dif_sens_pink = sens.get_for_f_171()[index_closest_171] - sens.get_for_pink()[index_closest_171]
sens.set_pink(sens.get_for_pink() + dif_sens_pink)

dif_sens_100_500 = sens.get_for_f_171()[index_closest_171] - sens.get_for_f_100_500()[index_closest_171]
sens.set_f_100_500(sens.get_for_f_100_500() + dif_sens_100_500)

# Corrección de directividad tomando como referencia los dBSPL a 171 Hz
direc.set_directivity_at_angle(0, direc.get_directivity_at_angle(0) + dif_dBSPL)
# Normalización con respecto a la directividad en 0º
direc.set_directivity_at_angle(0,
                               direc.get_directivity_at_angle(0) - direc.get_directivity_at_angle(0))
direc.set_directivity_at_angle(15, (
        direc.get_directivity_at_angle(15) + dif_dBSPL) - direc.get_directivity_at_angle(0))
direc.set_directivity_at_angle(30, (
        direc.get_directivity_at_angle(30) + dif_dBSPL) - direc.get_directivity_at_angle(0))
direc.set_directivity_at_angle(45, (
        direc.get_directivity_at_angle(45) + dif_dBSPL) - direc.get_directivity_at_angle(0))
direc.set_directivity_at_angle(60, (
        direc.get_directivity_at_angle(60) + dif_dBSPL) - direc.get_directivity_at_angle(0))
direc.set_directivity_at_angle(75, (
        direc.get_directivity_at_angle(75) + dif_dBSPL) - direc.get_directivity_at_angle(0))
direc.set_directivity_at_angle(90, (
        direc.get_directivity_at_angle(90) + dif_dBSPL) - direc.get_directivity_at_angle(0))

# Sonograma
direc_matrix = np.array([direc.get_directivity_at_angle(90),
                         direc.get_directivity_at_angle(75),
                         direc.get_directivity_at_angle(60),
                         direc.get_directivity_at_angle(45),
                         direc.get_directivity_at_angle(30),
                         direc.get_directivity_at_angle(15),
                         direc.get_directivity_at_angle(0),
                         direc.get_directivity_at_angle(15),
                         direc.get_directivity_at_angle(30),
                         direc.get_directivity_at_angle(45),
                         direc.get_directivity_at_angle(60),
                         direc.get_directivity_at_angle(75),
                         direc.get_directivity_at_angle(90)])

direc_oct_0 = average_directivity_per_octave(direc.get_directivity_at_angle(0), direc.get_frequencies())
direc_oct_15 = average_directivity_per_octave(direc.get_directivity_at_angle(15), direc.get_frequencies())
direc_oct_30 = average_directivity_per_octave(direc.get_directivity_at_angle(30), direc.get_frequencies())
direc_oct_45 = average_directivity_per_octave(direc.get_directivity_at_angle(45), direc.get_frequencies())
direc_oct_60 = average_directivity_per_octave(direc.get_directivity_at_angle(60), direc.get_frequencies())
direc_oct_75 = average_directivity_per_octave(direc.get_directivity_at_angle(75), direc.get_frequencies())
direc_oct_90 = average_directivity_per_octave(direc.get_directivity_at_angle(90), direc.get_frequencies())

direc_matrix_oct = np.array([direc_oct_90,
                             direc_oct_75,
                             direc_oct_60,
                             direc_oct_45,
                             direc_oct_30,
                             direc_oct_15,
                             direc_oct_0,
                             direc_oct_15,
                             direc_oct_30,
                             direc_oct_45,
                             direc_oct_60,
                             direc_oct_75,
                             direc_oct_90])

plot_sensitivity(sensitivity=sens, frequencies=sens.get_frequencies())

index_100 = np.argmin(np.abs(100 - sens.get_frequencies()))
index_500 = np.argmin(np.abs(500 - sens.get_frequencies()))
sens_100_500 = np.mean(sens.get_for_pink()[index_100:index_500])
sens_pink = np.mean(sens.get_for_pink())

plot_sonogram(directivity_matrix=direc_matrix, frequencies=direc.get_frequencies())

# =============================================================================
# PATRÓN POLAR
# =============================================================================

# Datos a plotear    
polar_63 = direc_matrix_oct[:, 0]
polar_125 = direc_matrix_oct[:, 1]
polar_250 = direc_matrix_oct[:, 2]
polar_500 = direc_matrix_oct[:, 3]
polar_1k = direc_matrix_oct[:, 4]
polar_2k = direc_matrix_oct[:, 5]
polar_4k = direc_matrix_oct[:, 6]
polar_8k = direc_matrix_oct[:, 7]

# Directividad para 63, 250, 1k, 4k
s = pd.Series(np.arange(1))
theta = np.arange(-np.pi / 2, np.pi / 2 + 0.1, np.pi / 12)

fig3 = plt.figure(figsize=(16, 8))
ax3 = plt.subplot(111, projection='polar')
ax3.plot(theta, polar_63, linestyle='--', lw=2, label='63 Hz')
ax3.plot(theta, polar_250, linestyle='--', lw=2, label='250 Hz')
ax3.plot(theta, polar_1k, linestyle='--', lw=2, label='1 kHz')
ax3.plot(theta, polar_4k, linestyle='--', lw=2, label='4 kHz')

# Configuración del ploteo
ax3.set_theta_zero_location('N')
ax3.set_rorigin(-30)
ax3.set_thetamin(-90)
ax3.set_thetamax(90)

gr_label = [r"$90^o$", r"$30^o$", r"$60^o$", r"$0^o$", r"$60^o$", r"$30^o$", r"$90^o$"]
plt.xticks(np.array([-np.pi / 2, -np.pi / 6, -np.pi / 3, 0, np.pi / 3, np.pi / 6, np.pi / 2]), gr_label, fontsize=14)
dB_label = [r"$-25.0$", r"$-20.0$", r"$-15.0$", r"$-10.0$", r"$-5.0$", r"$0.0$"]
plt.yticks(np.array([-25, -20, -15, -10, -5, 0]), dB_label, fontsize=18)
plt.legend(loc='upper right', fontsize=14)
# plt.title('Patrón polar - 63 Hz, 250 Hz, 1 kHz, 4 kHz',fontsize=20)
plt.savefig('Patrón polar - 63 Hz, 250 Hz, 1 kHz, 4 kHz.png')

# Directividad para 125, 500, 2k, 8k
theta = np.arange(-np.pi / 2, np.pi / 2 + 0.1, np.pi / 12)

fig4 = plt.figure(figsize=(16, 8))
ax4 = plt.subplot(111, projection='polar')
ax4.plot(theta, polar_125, linestyle='--', lw=2, label='125 Hz')
ax4.plot(theta, polar_500, linestyle='--', lw=2, label='500 Hz')
ax4.plot(theta, polar_2k, linestyle='--', lw=2, label='2 kHz')
ax4.plot(theta, polar_8k, linestyle='--', lw=2, label='8 kHz')

# Configuración del ploteo
ax4.set_theta_zero_location('N')
ax4.set_rorigin(-30)
ax4.set_thetamin(-90)
ax4.set_thetamax(90)

gr_label = [r"$90^o$", r"$30^o$", r"$60^o$", r"$0^o$", r"$60^o$", r"$30^o$", r"$90^o$"]
plt.xticks(np.array([-np.pi / 2, -np.pi / 6, -np.pi / 3, 0, np.pi / 3, np.pi / 6, np.pi / 2]), gr_label, fontsize=14)
dB_label = [r"$-25.0$", r"$-20.0$", r"$-15.0$", r"$-10.0$", r"$-5.0$", r"$0.0$"]
plt.yticks(np.array([-25, -20, -15, -10, -5, 0]), dB_label, fontsize=18)
plt.legend(loc='upper right', fontsize=14)
# plt.title('Patrón polar - 125 Hz, 500 Hz, 2 kHz, 8 kHz',fontsize=20)
plt.savefig('Patrón polar - 125 Hz, 500 Hz, 2 kHz, 8 kHz.png')
