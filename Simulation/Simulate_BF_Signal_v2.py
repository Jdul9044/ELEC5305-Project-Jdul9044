import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

# Setup ########################################
fs = 48e3 #Hz
c = 1500 #m/s
mic_num = 8
# spacing = 1.875 #m [c/(2d) = max freq -> 343 with 0.5m is 343 Hz] -> 400
spacing = 1.875 #m [c/(2d) = max freq -> 343 with 0.5m is 343 Hz] -> 400
duration = 30 #seconds
noise_level = 0.1 # amplitude
azi_deg = 60 #deg
ele_deg = 0 #deg
mic_pos = np.zeros((mic_num,3))
mic_pos[:, 0] = np.arange(mic_num) * spacing
################################################
azi = np.deg2rad(azi_deg) #rad
ele = np.deg2rad(ele_deg) #rad

N = int(fs * duration)
n = np.arange(N)
t = n/fs

# Unit vector
u = np.array([
    np.cos(ele) * np.cos(azi),
    np.cos(ele) * np.sin(azi),
    np.sin(ele)
    ])

tau = np.dot(mic_pos, u) / c

# Broadband
freq_range = 50,250
broadband_signal = np.fft.rfft(np.random.rand(N))
freqs = np.fft.rfftfreq(N, d=1/fs)
freq_band = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
broadband_signal[~freq_band] = 0



audio = np.zeros((N, mic_num))
    
for m in range(mic_num):
    
    audio[:, m]= (
        0.1 * np.sin(2 * np.pi * 300 * (t - tau[m]))
        +
        0.1 * np.sin(2 * np.pi * 301 * (t - tau[m]))
    ) + np.fft.irfft(broadband_signal * np.exp(-1j * 2 * np.pi * freqs * tau[m]), n=N)
        
audio += (noise_level*np.random.randn(N, mic_num))
audio = audio / np.max(np.abs(audio))

for i in range(mic_num):
    audio_channel = audio[:,i]
    audio_pcm = np.int16(audio_channel * (2**15-1))
    audio_pcm.tofile("microphone" + str(i) + ".pcm")
    
