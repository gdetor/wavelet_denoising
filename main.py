import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from scipy.signal import butter, filtfilt
from scipy.signal import spectrogram

from sklearn.preprocessing import MinMaxScaler

from ml_decomposition import wavelet_denoising


def plot_coeffs_distribution(coeffs):
    fig = plt.figure()
    size_ = int(len(coeffs) // 2) + 1
    if size_ % 2 != 0:
        size_ = size_+1

    for i in range(len(coeffs)):
        ax = fig.add_subplot(size_, 2, i+1)
        ax.hist(coeffs[i], bins=50)


def pretty_plot(data, titles, palet, fs=1, length=100, nperseg=256):
    fig = plt.figure(figsize=(13, 13))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    index = 1
    for i, d in enumerate(data):
        ax = fig.add_subplot(8, 2, index)
        ax.plot(d[:length], color=palet[i])
        ax.set_title(titles[i])
        ax = fig.add_subplot(8, 2, index+1)
        f, t, Sxx = spectrogram(d, fs=fs, nperseg=nperseg)
        ax.pcolormesh(t, f, Sxx)
        index += 2


def run_experiment(data, fs=1, nperseg=256, length=100):
    titles = ['Original data',
              'Universal Method',
              'SURE Method',
              'SURE Method (theoretical)',
              'Energy Method',
              'MinMax Method',
              'SQTWOLOG Method',
              'Heursure Method']

    experiment = ['universal',
                  'rigsure',
                  'fullsure',
                  'energy',
                  'minmax',
                  'sqtwolog',
                  'heursure']

    wd = wavelet_denoising(normalize=False,
                           wavelet='bior4.4',
                           level=None,
                           mode='soft',
                           method="universal",
                           resolution=100,
                           energy_perc=[0.99, 0.97, 0.85])
    res = [data]
    for i, e in enumerate(experiment):
        wd.method = experiment[i]
        res.append(wd.fit(data))
    palet = ['r', 'b', 'k', 'm', 'c', 'orange', 'g', 'y']
    pretty_plot(res, titles, palet, fs=fs, length=length, nperseg=nperseg)


if __name__ == '__main__':
    # ECG Data
    fs = 100
    raw_data = pd.read_pickle("data/apnea_ecg.pkl")
    N = int(len(raw_data) // 1000)
    data = raw_data[:N].values
    data = data[:, 0]
    run_experiment(data, fs=fs)

    data = np.zeros((128,))
    data[:16] = 4
    data += np.random.normal(0, 1, (128,))
    run_experiment(data, length=100, nperseg=32)

    raw_data = np.genfromtxt("./data/Z001.txt")
    fc = 40
    fs = 173.61
    w = fc / (fs / 2)
    b, a = butter(5, w, 'low')
    data = filtfilt(b, a, raw_data)
    run_experiment(data, fs=fs)
    plt.show()
