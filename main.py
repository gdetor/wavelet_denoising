import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from scipy.signal import butter, filtfilt
from scipy.signal import spectrogram
# from scipy.signal import periodogram
from ml_decomposition import wavelet_denoising


def plot_coeffs_distribution(coeffs):
    fig = plt.figure()
    size_ = int(len(coeffs) // 2) + 1
    if size_ % 2 != 0:
        size_ = size_+1

    for i in range(len(coeffs)):
        ax = fig.add_subplot(size_, 2, i+1)
        ax.hist(coeffs[i], bins=50)


def run_experiment(data, presented_data_len=100):
    method = 'universal'
    wd = wavelet_denoising(normalize=False, wavelet='bior4.4', level=None,
                           mode='soft', method=method, resolution=100,
                           energy_perc=[0.99, 0.97, 0.85])
    denoised_data_univ = wd.transform(data)
    wd.method = 'sure'
    denoised_data_sure = wd.transform(data)
    wd.method = 'energy'
    denoised_data_lala = wd.transform(data)

    fig = plt.figure(figsize=(13, 13))
    K = presented_data_len
    ax = fig.add_subplot(421)
    ax.plot(data[:K], 'b')
    ax.set_title("Original Data")
    ax = fig.add_subplot(422)
    f, t, Sxx = spectrogram(data, fs=fs)
    ax.pcolormesh(t, f, Sxx)
    # ax.psd(data, Fs=fs)
    ax = fig.add_subplot(423)
    ax.plot(denoised_data_univ[:K], 'm')
    ax.set_title("Universal Method")
    ax = fig.add_subplot(424)
    f, t, Sxx = spectrogram(denoised_data_univ, fs=fs)
    ax.pcolormesh(t, f, Sxx)
    # ax.psd(denoised_data_univ[:K], Fs=fs)
    ax = fig.add_subplot(425)
    ax.plot(denoised_data_sure[:K], 'k')
    ax.set_title("SURE Method")
    ax = fig.add_subplot(426)
    f, t, Sxx = spectrogram(denoised_data_sure, fs=fs)
    ax.pcolormesh(t, f, Sxx)
    # ax.psd(denoised_data_sure[:K], Fs=fs)
    ax = fig.add_subplot(427)
    ax.plot(denoised_data_lala[:K], 'c')
    ax.set_title("Energy Method")
    ax = fig.add_subplot(428)
    f, t, Sxx = spectrogram(denoised_data_lala, fs=fs)
    ax.pcolormesh(t, f, Sxx)
    # ax.psd(denoised_data_lala[:K], Fs=fs)


if __name__ == '__main__':
    # EKG Data
    fs = 100
    raw_data = pd.read_pickle("data/apnea_ecg.pkl")
    N = int(len(raw_data) // 1000)
    data = raw_data[:N].values
    data = data[:, 0]
    run_experiment(data)

    # data = np.zeros((128,))
    # data[:16] = 4
    # data += np.random.normal(0, 1, (128,))
    # run_experiment(data, presented_data_len=16)

    raw_data = np.genfromtxt("./data/Z001.txt")
    fc = 40
    fs = 173.61
    w = fc / (fs / 2)
    b, a = butter(5, w, 'low')
    data = filtfilt(b, a, raw_data)
    run_experiment(data)
    plt.show()
