import numpy as np
# import pandas as pd
import matplotlib.pylab as plt

# from scipy.signal import butter, filtfilt
from scipy.signal import spectrogram

from denoising import WaveletDenoising


def plot_coeffs_distribution(coeffs):
    """! Plots all the wavelet decomposition's coefficients. """
    fig = plt.figure()
    size_ = int(len(coeffs) // 2) + 1
    if size_ % 2 != 0:
        size_ = size_+1

    for i in range(len(coeffs)):
        ax = fig.add_subplot(size_, 2, i+1)
        ax.hist(coeffs[i], bins=50)


def pretty_plot(data, titles, palet, fs=1, length=100, nperseg=256):
    """! Plots the contents of the list data. """
    fig = plt.figure(figsize=(13, 13))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    index = 1
    for i, d in enumerate(data):
        ax = fig.add_subplot(8, 2, index)
        ax.plot(d[:length], color=palet[i])
        ax.set_title(titles[i])
        ax = fig.add_subplot(8, 2, index+1)
        f, t, Sxx = spectrogram(d, fs=fs, nperseg=nperseg)
        ax.pcolormesh(t, f, Sxx, shading='auto')
        index += 2


def run_experiment(data, level=2, fs=1, nperseg=256, length=100):
    """! Run the wavelet denoising over the input data for each threshold
    method.
    """

    # Experiments titles / thresholding methods
    titles = ['Original data',
              'Universal Method',
              'SURE Method',
              'Energy Method',
              'SQTWOLOG Method',
              'Heursure Method']

    # Theshold methods
    experiment = ['universal',
                  'stein',
                  'energy',
                  'sqtwolog',
                  'heurstein']

    # WaveletDenoising class instance
    wd = WaveletDenoising(normalize=False,
                          wavelet='db3',
                          level=level,
                          thr_mode='soft',
                          selected_level=level,
                          method="universal",
                          energy_perc=0.90)

    # Run all the experiments, first element in res is the original data
    res = [data]
    for i, e in enumerate(experiment):
        wd.method = experiment[i]
        res.append(wd.fit(data))

    # Plot all the results for comparison
    palet = ['r', 'b', 'k', 'm', 'c', 'orange', 'g', 'y']
    pretty_plot(res,
                titles,
                palet,
                fs=fs,
                length=length,
                nperseg=nperseg)


if __name__ == '__main__':
    # ECG Data
    import pandas as pd
    fs = 100
    raw_data = pd.read_pickle("./data/apnea_ecg.pkl")
    N = int(len(raw_data) // 1000)
    data = raw_data[:N].values
    data = data[:, 0]
    run_experiment(data, level=3, fs=fs)

    # EEG Data
    # raw_data = np.genfromtxt("./data/Z001.txt")
    # fc = 40
    # fs = 173.61
    # w = fc / (fs / 2)
    # b, a = butter(5, w, 'low')
    # data = filtfilt(b, a, raw_data)
    # run_experiment(data, level=4, fs=fs)
    plt.show()
