import numpy as np

from scipy.signal import detrend

from sklearn.preprocessing import MinMaxScaler

from pywt import wavedec, dwt_max_level, Wavelet, threshold, waverec


# =====================================================================
# Auxiliary functions
# =====================================================================
def Energy(x):
    """! Computes the energy of a signal. The energy is essentially the
    magnitude of the signal (inner product of x with itself).

    @param x Input signal as numpy array (1D)

    @return The energy of the input signal x
    """
    return np.dot(x, x)


def EuclideanNorm(x):
    """! Computes the Euclidean norm (p-norm with p=2) of the input
    1D vector (signal) x.

    @param x The input signal (numpy float 1D ndarray)

    @return The norm of the input signal as a float scaler
    """
    return np.linalg.norm(x)


def mad(x):
    return 1.482579 * np.median(np.abs(x - np.median(x)))


def meanad(x):
    return 1.482579 * np.mean(np.abs(x - np.mean(x)))


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


def NearestEvenInteger(n):
    """! Returns the nearest even integer to number n.

    @param n Input number for which one requires the nearest even integer

    @return The even nearest integer to the input number
    """
    if n % 2 == 0:
        res = n
    else:
        res = n-1
    return res


def DyadicLength(x):
    """! Returns the length and the dyadic length of the input 1D array x.

    @param x The input signal (float 1D ndarray)

    @return Returns the length m and the least power of 2 greater than m

    @note This function has been taken from the pyYAWT package
    (https://pyyawt.readthedocs.io/_modules/pyyawt/denoising.html).
    """
    m = x.shape[0]
    j = np.ceil(np.log(m) / np.log(2.)).astype('i')
    return m, j


def SoftHardThresholding(x, thr=1, method='s'):
    """! Performs either a soft or a hard thresholding on the input signal x.

    @param x The 1D input signal
    @param thr The threshold value (float, default=1)
    @param method A string that indicates if either the soft or the hard
    thresholding is being used (default=soft, s for soft, h for hard)

    @return Returns the thresholded signal
    """
    if method.lower() == 'h':
        res = x * (np.abs(x) > thr)
    elif method.lower() == 's':
        res = ((x >= thr) * (x - thr) + (x <= -thr) * (x + thr)
               + (np.abs(x) <= thr) * 0)
    else:
        print("Thresholding method not found! Choose s (soft) or h (hard)")
        res = None
    return res


# =====================================================================
# Main wavelets denoising class
# =====================================================================
class WaveletDenoising(object):
    def __init__(self,
                 normalize=False,
                 wavelet='haar',
                 level=None,
                 mode='soft',
                 method='universal',
                 resolution=100,
                 energy_perc=0.9):
        self.wavelet = wavelet
        self.level = level
        self.method = method
        self.resolution = resolution
        self.mode = mode
        self.energy_perc = energy_perc
        self.normalize = normalize

        self.filter_ = Wavelet(self.wavelet)

        if level is None:
            self.nlevel = 0
        else:
            self.nlevel = level
        self.normalized_data = None

    def fit(self, signal):
        tmp_signal = signal.copy()
        tmp_signal = self.Preprocess(tmp_signal)
        coeffs = self.WavTransform(tmp_signal)
        denoised_signal = self.Denoise(tmp_signal, coeffs)
        return denoised_signal

    # ********************************************************************
    # Preprocessing methods
    def Preprocess(self, signal, normalize=False):
        # Remove all the unnecessary trends (DC, etc)
        xhat = detrend(signal)
        # Normalize the data (bring them into [0, 1])
        if self.normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
            xhat = self.scaler.fit_transform(xhat.reshape(-1, 1))[:, 0]
            self.normalized_data = xhat.copy()
        return xhat

    # ********************************************************************
    # Multilevel wavelet decomposition
    def WavTransform(self, signal):
        # Find the nearest even integer to input signal's length
        size = NearestEvenInteger(signal.shape[0])
        # Check if a WAVEDEC level has been provided otherwise infer one
        if self.nlevel == 0:
            level = dwt_max_level(signal.shape[0],
                                  filter_len=self.filter_.dec_len)
            self.nlevel = level
        # Compute the Wavelet coefficients using WAVEDEC
        coeffs = wavedec(signal[:size], self.filter_, level=self.nlevel)
        return coeffs

    # ********************************************************************
    # Main denoising method
    def Denoise(self, signal, coeffs):
        # Determine the threshold for the coefficients based on the level of
        # WAVEDEC
        thr = self.DetermineThreshold(coeffs[-self.nlevel], self.energy_perc)

        # Apply the threshold to all the coefficients
        coeffs[1:] = [threshold(c, value=thr, mode=self.mode)
                      for c in coeffs[1:]]

        # Apply the WAVEREC to reconstruct the signal
        denoised_signal = waverec(coeffs, self.filter_, mode='smooth')

        # Renormalize in case the input signal was normalized to [0, 1]
        if self.normalize:
            denoised_signal = self.scaler.inverse_transform(
                    denoised_signal.reshape(-1, 1))[:, 0]
        return denoised_signal

    # ********************************************************************
    # Thresholding methods
    def DetermineThreshold(self, signal, energy_perc):
        thr = 0.0
        if self.method == 'universal':
            thr = self.UniversalThreshold(signal)
        elif self.method == 'heursure':
            thr = self.heursure_thresholding(signal)
        elif self.method == 'sqtwolog':
            thr = self.SquareRootLogThreshold(signal)
        elif self.method == 'rigsure':
            thr = self.sure_thresholding(signal)
        elif self.method == 'energy':
            thr = self.EnergyThreshold(signal, perc=energy_perc)
        elif self.method == 'fullsure':
            thr = self.SURETheoreticThreshold(signal)
        else:
            print("No such method detected!")
            print("Set back to default (universal thresholding)!")
            thr = self.UniversalThreshold(signal)
        return thr

    def UniversalThreshold(self, signal):
        """! Universal threshold
        @param signal Input signal (1D ndarray of floats)

        @return A float scaler representing the threshold value
        """
        m = signal.shape[0]
        sigma = mad(signal)
        # sigma = meanad(signal)
        thr = sigma * np.sqrt((2*np.log(m)) / m)
        # thres = sigma * np.sqrt(2 * np.log(m))
        return thr

    def SURE_auxiliary(self, signal, thr):
        m = signal.shape[0]
        sigma = mad(signal)
        g_fun = threshold(signal, value=thr, mode=self.mode) - signal
        norm_g = EuclideanNorm(g_fun)
        grad_sum = sum([grad_g_fun(x, thr=thr) for x in signal])
        return sigma**2 + (norm_g**2 + 2*sigma**2*grad_sum) / m

    def SURETheoreticThreshold(self, signal):
        t = np.linspace(1e-1, 2*np.log(signal.shape[0]), self.resolution)
        thr_values = []
        for i in range(self.resolution):
            thr_values.append(self.SURE_auxiliary(signal, thr=t[i]))
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
        m, j = DyadicLength(signal)
        magic = np.sqrt(2 * np.log(m))
        eta = (np.linalg.norm(signal)**2 - m) / m
        critical = j**(1.5)/np.sqrt(m)
        if eta < critical:
            thr = magic
        else:
            thr = np.min((self.sure_thresholding(signal), magic))
        return thr

    def SquareRootLogThreshold(self, signal):
        """! It computes a threshold for the input signal. The threshold value
        is given by the squared-root of 2 x log(m), where m is the length of
        the input signal.

        Square-root of 2log threshold
        @param signal Input signal (1D ndarray of floats)

        @return A float scaler representing the threshold value
        """
        m = len(signal)
        thr = np.sqrt(2.0 * np.log(m))
        return thr

    def EnergyThreshold(self, signal, perc=0.1):
        """! Energy-based threshold method. It estimates a threshold value for
        the input signal based on signal's energy.

        @param signal Input signal (1D ndarray of floats)
        @param perc   Energy retained percentage (flaot scaler)

        @return A float scaler representing the threshold value
        """
        tmp_signal = np.sort(np.abs(signal))[::-1]
        energy_thr = perc * Energy(tmp_signal)
        energy_tmp = 0
        for sig in tmp_signal:
            energy_tmp += sig**2
            if energy_tmp >= energy_thr:
                thr = sig
                break
        return thr
