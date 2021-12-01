import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pywt import wavedec, dwt_max_level, Wavelet, threshold, waverec

# =====================================================================
# Auxiliary functions
# =====================================================================


def energy(x):
    return np.dot(x, x)


def euclidean_norm(x):
    return np.sqrt((x**2).sum())


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def grad_g_fun(x, thr=1):
    return (x >= thr) * 1 + (x <= -thr) * 1 + (np.abs(x) <= thr) * 0


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


def nearest_even_int(n):
    if n % 2 == 0:
        res = n
    else:
        res = n-1
    return res


def binary_length(x):
    m = x.shape[0]
    j = np.ceil(np.log(m) / np.log(2.)).astype('i')
    return m, j


def soft_hard_thresholding(x, thr=1, is_hard_on=False):
    if is_hard_on:
        res = x * (np.abs(x) > thr)
    else:
        res = ((x >= thr) * (x - thr) + (x <= -thr) * (x + thr)
               + (np.abs(x) <= thr) * 0)
    return res
# =====================================================================
# Main wavelets denoising class
# =====================================================================


class wavelet_denoising(object):
    def __init__(self,
                 normalize=False,
                 wavelet='haar',
                 level=None,
                 mode='soft',
                 method='universal',
                 resolution=100,
                 energy_perc=[0.9],
                 is_mad_on=False):
        self.wavelet = wavelet
        self.level = level
        self.method = method
        self.resolution = resolution
        self.mode = mode
        self.energy_perc = energy_perc
        self.normalize = normalize
        self.mad = is_mad_on
        if level is None:
            self.nlevel = 0
        else:
            self.nlevel = level
        self.normalized_data = 0

    def fit(self, signal):
        tmp_signal = signal.copy()
        if self.normalize is True:
            scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            tmp_signal = scaler.fit_transform(tmp_signal.reshape(-1, 1))[:, 0]
            self.normalized_data = tmp_signal.copy()
        coeffs = self.wav_transform(tmp_signal)
        denoised_signal = self.denoise(coeffs)
        return denoised_signal

    # ********************************************************************
    # Multilevel wavelet decomposition
    def wav_transform(self, signal):
        size = nearest_even_int(signal.shape[0])
        filter_ = Wavelet(self.wavelet)
        if self.level is None:
            level = dwt_max_level(signal.shape[0], filter_len=filter_.dec_len)
            self.nlevel = level
        coeffs = wavedec(signal[:size], filter_, level=level)
        return coeffs

    # ********************************************************************
    # Main denoising method
    def denoise(self, signal):
        filter_ = Wavelet(self.wavelet)

        nl = self.nlevel
        ne = len(self.energy_perc)
        if nl >= ne:
            diff = nl - ne + 1
            self.energy_perc = (self.energy_perc+[self.energy_perc[-1]]
                                * diff)
        denoised_coeffs = []
        for i, coeff in enumerate(signal):
            thr = self.determine_threshold(coeff,
                                           self.energy_perc[i],
                                           is_mad_on=False)
            tmp_coeff = threshold(coeff,
                                  thr,
                                  mode=self.mode,
                                  substitute=0)
            denoised_coeffs.append(tmp_coeff)
        denoised_signal = waverec(denoised_coeffs, filter_, mode='smooth')
        return denoised_signal
    # ********************************************************************
    # Thresholding methods

    def determine_threshold(self, signal, energy_perc, is_mad_on=False):
        thr = 0.0
        if self.method == 'universal':
            thr = self.universal_thresholding(signal, is_mad_on=is_mad_on)
        elif self.method == 'minmax':
            thr = self.minmaxi_thresholding(signal)
        elif self.method == 'heursure':
            thr = self.heursure_thresholding(signal)
        elif self.method == 'sqtwolog':
            thr = self.sqtwolog_thresholding(signal)
        elif self.method == 'rigsure':
            thr = self.sure_thresholding(signal)
        elif self.method == 'energy':
            thr = self.energy_thresholding(signal, perc=energy_perc)
        elif self.method == 'fullsure':
            thr = self.full_sure_thresholding(signal)
        else:
            print("No such method detected!")
            print("Set back to default (universal thresholding)!")
            thr = self.universal_thresholding(signal, is_mad_on=is_mad_on)
        return thr

    def universal_thresholding(self, signal, is_mad_on=False):
        """ Universal thresholding """
        m = signal.shape[0]
        if is_mad_on:
            sigma = mad(signal)
        else:
            sigma = 1.0
        thr = sigma * np.sqrt((2*np.log(m)) / m)
        # thres = sigma * np.sqrt(2 * np.log(m))
        return thr

    def minmaxi_thresholding(self, signal):
        lamlist = [0, 0, 0, 0, 0, 1.27, 1.474, 1.669, 1.860, 2.048, 2.232,
                   2.414, 2.594, 2.773, 2.952, 3.131, 3.310, 3.49, 3.67, 3.85,
                   4.03, 4.21]
        _, j = binary_length(signal)
        if j <= len(lamlist):
            thr = lamlist[j-1]
        else:
            thr = 4.21 + (j - len(lamlist)) * 0.18
        return thr

    def aux_sure(self, signal, thr):
        m = signal.shape[0]
        sigma = mad(signal)
        g_fun = soft_hard_thresholding(signal, thr=thr) - signal
        norm_g = euclidean_norm(g_fun)
        grad_sum = sum([grad_g_fun(x, thr=thr) for x in signal])
        return sigma**2 + (norm_g**2 + 2*sigma**2*grad_sum) / m

    def full_sure_thresholding(self, signal):
        t = np.linspace(1e-1, 2*np.log(signal.shape[0]), self.resolution)
        thr_values = []
        for i in range(self.resolution):
            thr_values.append(self.aux_sure(signal, thr=t[i]))
        thr_values = np.array(thr_values)
        opt_thr = t[np.argmin(thr_values)]
        return opt_thr

    def sure_thresholding(self, signal):
        m = signal.shape[0]
        sorted_signal = np.sort(np.abs(signal))**2
        c = np.linspace(m-1, 0, m)
        s = np.cumsum(sorted_signal) + c * sorted_signal
        risk = (m - (2.0 * np.arange(m)) + s) / m
        ibest = np.argmin(risk)
        thr = np.sqrt(sorted_signal[ibest])
        return thr

    def heursure_thresholding(self, signal):
        m, j = binary_length(signal)
        magic = np.sqrt(2 * np.log(m))
        eta = (np.linalg.norm(signal)**2 - m) / m
        critical = j**(1.5)/np.sqrt(m)
        if eta < critical:
            thr = magic
        else:
            thr = np.min((self.sure_thresholding(signal), magic))
        return thr

    def sqtwolog_thresholding(self, signal):
        m, _ = binary_length(signal)
        thr = np.sqrt(2.0 * np.log(m))
        return thr

    def energy_thresholding(self, signal, perc=0.1):
        """ Energy-based thresholding """
        tmp_signal = np.sort(np.abs(signal))[::-1]
        energy_thr = perc * energy(tmp_signal)
        energy_tmp = 0
        for sig in tmp_signal:
            energy_tmp += sig**2
            if energy_tmp >= energy_thr:
                thr = sig
                break
        return thr
