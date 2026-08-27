import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, istft

# Setup ########################################
fs = 48e3 #Hz
c = 343 #m/s
mic_num = 8
spacing = 0.5 #m [c/(2d) = max freq -> 343 with 0.5m is 343 Hz]
duration = 100 #seconds
noise_level = 1 # amplitude
azi_deg = 30 #deg
azi = np.deg2rad(azi_deg) #rad
ele_deg = 0 #deg
ele = np.deg2rad(ele_deg) #rad
mic_pos = np.zeros((mic_num,3))
mic_pos[:, 0] = np.arange(mic_num) * spacing
################################################

N = int(fs * duration)
n = np.arange(N)
t = n/fs

# Unit vector
u = np.array([
    np.cos(ele) * np.cos(azi),
    np.cos(ele) * np.sin(azi),
    np.sin(ele)
    ])

source = (
    0.1 * np.sin(2 * np.pi * 200 * t)
    +
    0.1 * np.sin(2 * np.pi * 201 * t)
)

tau = np.dot(mic_pos,u) / c

audio = np.zeros((N, mic_num))


for m in range(mic_num):

    delayed_time = t - tau[m]
    audio[:, m] = np.interp(delayed_time, t, source, left=0.0, right=0.0)
        
audio += (noise_level*np.random.randn(N, mic_num))
audio = np.clip(audio,-1,1)

for i in range(mic_num):
    audio[:,i]
    audio_pcm = np.int16(audio * 2**15-1)
    audio_pcm.tofile("microphone" + str(i) + ".pcm")
    
    
