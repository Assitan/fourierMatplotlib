import numpy as np
from matplotlib import pyplot as plt
from scipy.io.wavfile import write
from scipy.fft import fft, fftfreq, rfft, rfftfreq, irfft

SAMPLE_RATE = 44100  # Hertz
DURATION = 5  # Seconds


# generate a sine wave
def generate_sine_wave(freq, sample_rate, duration):
    xd = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = xd * freq
    # 2pi because np.sin takes radians
    yd = np.sin((2 * np.pi) * frequencies)
    return xd, yd


# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
plt.plot(x, y)
plt.show()

# generate the signals
_, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
_, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
noise_tone = noise_tone * 0.3

mixed_tone = nice_tone + noise_tone

# normalization
normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767)

plt.plot(normalized_tone[:1000])
plt.show()

# store in an audio format

# Remember SAMPLE_RATE = 44100 Hz is our playback rate
write("audios/mysinewave.wav", SAMPLE_RATE, normalized_tone)

# Fast Fourier Transform
# Number of samples in normalized_tone
N = SAMPLE_RATE * DURATION

yf = fft(normalized_tone)
xf = fftfreq(N, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
plt.show()

# calculating the Fourier transform
yf = fft(normalized_tone)
xf = fftfreq(N, 1 / SAMPLE_RATE)

# plot the values
plt.plot(xf, np.abs(yf))
plt.show()

# Making It Faster
yf = fft(normalized_tone)
xf = fftfreq(N, 1 / SAMPLE_RATE)

# Note the extra 'r' at the front
yf = rfft(normalized_tone)
xf = rfftfreq(N, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
plt.show()

# Filtering the Signal (fixing)
# The maximum frequency is half the sample rate
points_per_freq = len(xf) / (SAMPLE_RATE / 2)

# Our target frequency is 4000 Hz
target_idx = int(points_per_freq * 4000)

yf[target_idx - 1 : target_idx + 2] = 0

plt.plot(xf, np.abs(yf))
plt.show()

# Applying the Inverse FFT

new_sig = irfft(yf)

plt.plot(new_sig[:1000])
plt.show()

norm_new_sig = np.int16(new_sig * (32767 / new_sig.max()))

write("audios/clean.wav", SAMPLE_RATE, norm_new_sig)

