# import numpy as np
# from navtools.signals import gps_l1ca_prn_generator
# from navtools.dsp import carrier_from_frequency, parcorr, upsample_sequence

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set_context("notebook")

# prn = gps_l1ca_prn_generator(prn=23)
# prn2 = np.roll(prn, 512)
# acorr = parcorr(prn2, prn)

# chips = np.arange(-512, 511)
# plt.plot(chips, np.sqrt(acorr), color="#D4586D")
# plt.ylabel("$R_{C/A}(\\delta \\tau)$")
# plt.xlabel("$\\delta \\tau$ [chips]")


# Fs = 4 * 1575.42e6
# Fc = 1575.42e6
# # carr = carrier_from_frequency(1575.42e6, 10 * 1575.42e6, 0.001)
# carr = carrier_from_frequency(Fc, Fs, 0.001)
# prn_up = upsample_sequence(prn, carr.size, 2 * 1575.42e6, 1.023e6)

# sig = carr * prn_up
# PSD_carr = np.abs(np.fft.fft(carr)) ** 2 / (carr.size * Fs)
# PSD_log_carr = 10.0 * np.log10(PSD_carr)
# PSD_shifted_carr = np.fft.fftshift(PSD_log_carr)

# PSD = np.abs(np.fft.fft(sig)) ** 2 / (sig.size * Fs)
# PSD_log = 10.0 * np.log10(PSD)
# PSD_shifted = np.fft.fftshift(PSD_log)

# center_freq = 0  # frequency we tuned our SDR to
# f = np.arange(
#     Fs / -2.0, Fs / 2.0, Fs / carr.size
# )  # start, stop, step.  centered around 0 Hz
# f += center_freq  # now add center frequency
# plt.figure()
# plt.plot(f, PSD_shifted, color="#8CBCE8", label="spread signal")
# plt.plot(f, PSD_shifted_carr, color="#D4586D", label="carrier")
# plt.xlim(1575.42e6 - 10e6, 1575.42e6 + 10e6)
# plt.ylabel("Magnitude [dB]")
# plt.xlabel("Frequency [Hz]")
# ax = plt.gca()
# axins = ax.inset_axes(
#     [0.65, 0.425, 0.325, 0.325],
#     xlim=(1575.42e6 - 5e4, 1575.42e6 + 5e4),
#     ylim=(-300, -20),
# )
# axins.plot(f, PSD_shifted, color="#8CBCE8")
# axins.plot(f, PSD_shifted_carr, color="#D4586D")
# axins.set_xticks([Fc])
# ax.legend(loc="upper left")

# plt.show()

import numpy as np
import pandas as pd
from scipy.io import loadmat
import datetime
import matplotlib.pyplot as plt
from navsim.emitters import SatelliteEmitters
from navsim.emitters import GPSTime
from scipy.interpolate import interp1d

FILE_PATH = "/home/tannerkoza/devel/eleventh-hour/data/stl2600-2.csv"
RX_POS = np.array([-1520424, -5083133, 3530701])


def gpst2utc(week: int, tow: float, leap_seconds: int = 18):
    GPS_TIME_EPOCH = datetime.datetime(1980, 1, 6, 0, 0, 0, 0, None)

    utc = (
        GPS_TIME_EPOCH
        + datetime.timedelta(weeks=int(week))
        + datetime.timedelta(seconds=tow - leap_seconds)
    )

    return utc


def main():
    emitters = SatelliteEmitters(
        constellations="iridium-next", mask_angle=-5, disable_progress=True
    )

    jl_df = pd.read_csv(FILE_PATH, header=2)
    jl_df.columns = jl_df.columns.str.replace(" ", "")

    grouped_df = jl_df.groupby(jl_df["sv_data[0].prn"])

    for prn in grouped_df.groups.keys():
        df = grouped_df.get_group(prn)

        week = df["sv_data[0].sv_data_time.week_number"].to_numpy()
        tow = df["sv_data[0].sv_data_time.seconds_of_week"].to_numpy()
        datetimes = np.array([gpst2utc(week=w, tow=sec) for (w, sec) in zip(week, tow)])
        elapsed_time = np.array(
            [(dt - datetimes[0]).total_seconds() for dt in datetimes]
        )

        states = emitters.from_datetimes(datetimes=datetimes, rx_pos=RX_POS)

        prn_pos = df[
            ["sv_data[0].sv_pos[0]", "sv_data[0].sv_pos[1]", "sv_data[0].sv_pos[2]"]
        ].to_numpy()
        prn_vel = df[
            ["sv_data[0].sv_vel[0]", "sv_data[0].sv_vel[1]", "sv_data[0].sv_vel[2]"]
        ].to_numpy()

        sv_pos0 = np.array([sv.pos for sv in states[0].values()])

        matched_sv_index = np.argmin(np.linalg.norm(prn_pos[0] - sv_pos0, axis=1))
        matched_sv = list(states[0].keys())[matched_sv_index]

        matched_pos = np.array([epoch[matched_sv].pos for epoch in states])
        matched_vel = np.array([epoch[matched_sv].vel for epoch in states])

        pos_error = prn_pos - matched_pos
        norm_pos_error = np.linalg.norm(pos_error, axis=1)
        unbiased_norm_pos_error = norm_pos_error - norm_pos_error[0]

        vel_error = prn_vel - matched_vel
        norm_vel_error = np.linalg.norm(vel_error, axis=1)
        unbiased_norm_vel_error = norm_vel_error - norm_vel_error[0]

        plt.plot(elapsed_time, norm_pos_error, ".")

    plt.show()


if __name__ == "__main__":
    main()


# f = interp1d(dt_t, sim_pos, kind="cubic", bounds_error=False, fill_value="extrapolate")
# sim_pos = f(t)

# sv_pos = df[
#     ["sv_data[0].sv_pos[0]", "sv_data[0].sv_pos[1]", "sv_data[0].sv_pos[2]"]
# ].to_numpy()
# sv_vel = df[
#     ["sv_data[0].sv_vel[0]", "sv_data[0].sv_vel[1]", "sv_data[0].sv_vel[2]"]
# ].to_numpy()


# plt.figure()
# plt.plot(t, sim_pos[0], ".")
# # plt.plot(t, sv_pos[:, 0])

# plt.figure()
# plt.plot(t, sim_pos[1])
# plt.plot(t, sv_pos[:, 1])

# plt.figure()
# plt.plot(t, sim_pos[2])
# plt.plot(t, sv_pos[:, 2])

# plt.figure()
# error = sim_pos.T - sv_pos
# plt.plot(t, np.linalg.norm(error, axis=1))
# plt.show()
# print()
