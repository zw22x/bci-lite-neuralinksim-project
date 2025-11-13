""" Synthetic EEG Generator
    Zac Westbrook"""
import numpy as np

# Class labels "rest", "left", "right", "feet" 

class SyntheticEEG:
    def __init__(self, fs: int = 250, duration: float = 4.0, channels: int= 8):
        self.fs = fs
        self.duration = duration
        self.n_samples = int(fs * duration)
        self.t = np.linspace(0, duration, self.n_samples, endpoint=False)
        self.channels = channels 

    def _band_limited_noise(self, low: float, high: float, amplitude: float = 1.0) -> np.array:
        w = np.fft.rfftfreq(self.n_samples, 1/self.fs)
        mask = (w >= low) & (w <= high)
        psd = np.zeros_like(w)
        psd[mask] = 1
        psd /= np.sqrt(np.sum(psd**2))
        phase = np.random.uniform(0, 2*np.pi, len(w))
        signal_fft = np.sqrt(psd) * np.exp(1j * phase)
        signal = np.fft.irfft(signal_fft, n=self.n_samples)
        return amplitude * signal / np.std(signal)
    
    def generate_trial(self, label: str) -> np.ndarray:
        signal = np.zeros((self.channels, self.n_samples))

        if label == "left":
            signal[0, :] += self._band_limited_noise(8, 12, amplitude=2.0)
            signal[0, :] += self._band_limited_noise(18, 25, amplitude=1.0)
        elif label == "right":
            signal[2, :] += self._band_limited_noise(8, 12, amplitude=2.0)
            signal[2, :] += self._band_limited_noise(18, 25, amplitude=1.0)
        elif label == "feet":
            signal [1, :] += self._band_limited_noise(10, 13, amplitude=1.5)

        background = self._band_limited_noise(1, 40, amplitude=0.5)
        for ch in range(self.channels):
            signal[ch, :] += background
        noise = np.random.normal(0, 0.3, signal.shape)
        signal += noise

        return signal
    
                            
                            