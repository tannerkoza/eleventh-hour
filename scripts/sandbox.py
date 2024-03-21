import numpy as np
from navtools.signals import gps_l1ca_prn_generator
from navtools.dsp import carrier_from_frequency, parcorr, upsample_sequence

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("notebook")

prn = gps_l1ca_prn_generator(prn=23)
prn2 = np.roll(prn, 512)
acorr = parcorr(prn2, prn)

chips = np.arange(-512, 511)
plt.plot(chips, np.sqrt(acorr), color="#D4586D")
plt.ylabel("$R_{C/A}(\\delta \\tau)$")
plt.xlabel("$\\delta \\tau$ [chips]")


Fs = 4 * 1575.42e6
Fc = 1575.42e6
# carr = carrier_from_frequency(1575.42e6, 10 * 1575.42e6, 0.001)
carr = carrier_from_frequency(Fc, Fs, 0.001)
prn_up = upsample_sequence(prn, carr.size, 2 * 1575.42e6, 1.023e6)

sig = carr * prn_up
PSD_carr = np.abs(np.fft.fft(carr)) ** 2 / (carr.size * Fs)
PSD_log_carr = 10.0 * np.log10(PSD_carr)
PSD_shifted_carr = np.fft.fftshift(PSD_log_carr)

PSD = np.abs(np.fft.fft(sig)) ** 2 / (sig.size * Fs)
PSD_log = 10.0 * np.log10(PSD)
PSD_shifted = np.fft.fftshift(PSD_log)

center_freq = 0  # frequency we tuned our SDR to
f = np.arange(
    Fs / -2.0, Fs / 2.0, Fs / carr.size
)  # start, stop, step.  centered around 0 Hz
f += center_freq  # now add center frequency
plt.figure()
plt.plot(f, PSD_shifted, color="#8CBCE8", label="spread signal")
plt.plot(f, PSD_shifted_carr, color="#D4586D", label="carrier")
plt.xlim(1575.42e6 - 10e6, 1575.42e6 + 10e6)
plt.ylabel("Magnitude [dB]")
plt.xlabel("Frequency [Hz]")
ax = plt.gca()
axins = ax.inset_axes(
    [0.65, 0.425, 0.325, 0.325],
    xlim=(1575.42e6 - 5e4, 1575.42e6 + 5e4),
    ylim=(-300, -20),
)
axins.plot(f, PSD_shifted, color="#8CBCE8")
axins.plot(f, PSD_shifted_carr, color="#D4586D")
axins.set_xticks([Fc])
ax.legend(loc="upper left")

plt.show()
