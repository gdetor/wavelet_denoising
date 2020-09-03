import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pywt import wavedec, dwt_max_level, Wavelet, threshold, waverec


class wavelet_denoising(object):
    def __init__(self, normalize=False, wavelet='haar', level=None,
                 mode='soft', method='universal', resolution=100,
                 energy_perc=[0.9]):
        self.wavelet = wavelet
        self.level = level
        self.method = method
        self.resolution = resolution
        self.mode = mode
        self.energy_perc = energy_perc
        self.normalize = normalize
        if level is None:
            self.nlevel = 0
        else:
            self.nlevel = level
        self.normalized_data = 0

    def transform(self, signal):
        tmp_signal = signal.copy()
        if self.normalize is True:
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            tmp_signal = scaler.fit_transform(tmp_signal.reshape(-1, 1))[:, 0]
            self.normalized_data = tmp_signal.copy()
        coeffs = self.wav_transform(tmp_signal)
        denoised_signal = self.denoise(coeffs)
        return denoised_signal

    def universal_threshold(self, signal, *args):
        m = signal.shape[0]
        sigma = self.mad(signal)
        thres = sigma * np.sqrt((2*np.log(m)) / m)
        # thres = sigma * np.sqrt(2 * np.log(m))
        return thres

    def SURE(self, signal, thr):
        n = signal.shape[0]
        sigma = self.mad(signal)
        g_fun = self.soft_threshold(signal, thr=thr) - signal
        norm_g = self.euclidean_norm(g_fun)
        grad_sum = sum([self.grad_g_fun(x, thr=thr) for x in signal])
        return sigma**2 + (norm_g**2 + 2*sigma**2*grad_sum) / n

    def sure_threshold(self, signal, *args):
        t = np.linspace(1e-1, 2*np.log(signal.shape[0]), self.resolution)
        thr_values = []
        for i in range(self.resolution):
            thr_values.append(self.SURE(signal, thr=t[i]))
        thr_values = np.array(thr_values)
        opt_thr = t[np.argmin(thr_values)]
        return opt_thr

    def energy_threshold(self, signal, *args):
        perc = args[0]
        tmp_signal = np.sort(np.abs(signal))[::-1]
        energy_thr = perc * self.energy(tmp_signal)
        energy_tmp = 0
        for sig in tmp_signal:
            energy_tmp += sig**2
            if energy_tmp >= energy_thr:
                thr = sig
                break
        return thr

    def wav_transform(self, signal):
        size = self.nearest_even_int(signal.shape[0])
        filter_ = Wavelet(self.wavelet)
        if self.level is None:
            level = dwt_max_level(signal.shape[0], filter_len=filter_.dec_len)
            self.nlevel = level
        coeffs = wavedec(signal[:size], filter_, level=level)
        return coeffs

    def denoise(self, signal):
        filter_ = Wavelet(self.wavelet)

        nl = self.nlevel
        ne = len(self.energy_perc)
        if nl > ne:
            diff = nl - ne + 1
            self.energy_perc = (self.energy_perc+[self.energy_perc[-1]]
                                * diff)

        if self.method == 'universal':
            method_fun = self.universal_threshold
        elif self.method == 'sure':
            method_fun = self.sure_threshold
        else:
            method_fun = self.energy_threshold

        denoised_coeffs = []
        for i, c in enumerate(signal):
            thres = method_fun(c, self.energy_perc[i])
            c = threshold(c, thres, mode=self.mode, substitute=0)
            denoised_coeffs.append(c)
        denoised_signal = waverec(denoised_coeffs, filter_, mode='smooth')
        return denoised_signal

    def nearest_even_int(self, n):
        if n % 2 == 0:
            return n
        else:
            return n-1

    def soft_threshold(self, x, thr=0):
        return ((x >= thr) * (x - thr) + (x <= -thr) * (x + thr)
                + (np.abs(x) <= thr) * 0)

    def grad_g_fun(self, x, thr=1):
        return (x >= thr) * 1 + (x <= -thr) * 1 + (np.abs(x) <= thr) * 0

    def energy(self, x):
        return np.dot(x, x)

    def euclidean_norm(self, x):
        return np.sqrt((x**2).sum())

    def mad(self, x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))

    def transform_mean_var(signal, x, mu=None, sigma=None):
        mu1 = x.mean()
        sigma1 = signal.std()
        if mu is None:
            mu2 = mu1
        else:
            mu2 = mu
        if sigma is None:
            sigma2 = sigma1
        else:
            sigma2 = sigma
        return mu2 + (x - mu1) * sigma2 / sigma1
