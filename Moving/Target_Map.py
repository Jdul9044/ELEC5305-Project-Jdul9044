import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft, spectrogram
from pathlib import Path
from scipy.signal import welch
from scipy.io import wavfile, loadmat
import pandas as pd


fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

theta = np.linspace(0, 360, 361)
ax.plot(
    [np.deg2rad(60), np.deg2rad(60)],
    [0, 1],label="Target 1 - 250Hz tone (original position)", linewidth=4, color="C0", alpha=0.4
)

ax.plot(
    [np.deg2rad(90), np.deg2rad(90)],
    [0, 1],label="Target 1 - 250Hz tone (moved)", linewidth=4, color="C0"
)

ax.plot(
    [np.deg2rad(10), np.deg2rad(10)],
    [0, 1], label="Target 2 - 300Hz tone and broadband (200-350Hz)", linewidth=4, color="C1"
)


r = ax.get_rmax()

ax.plot(
    np.deg2rad(60),
    1,
    'o',
    color='red',
    markersize=8
)

ax.plot(
    np.deg2rad(90),
    1,
    'o',
    color='red',
    markersize=8
)

ax.plot(
    np.deg2rad(10),
    1,
    'o',
    color='red',
    markersize=8
)

ax.set_theta_zero_location('N')   # 0° = North / top
ax.set_theta_direction(-1)        # clockwise
ax.set_title('Simulation Moving Target Map')
ax.set_yticklabels([])
# ax.set_ylim(0, 1)
ax.legend()