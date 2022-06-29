# Signal denoising with Wavelets

This repository contains a Python class for signal denoising using the Wavelet's
multilevel decomposition. The current implementation is based on the Python's 
package [PyWavelets](https://pywavelets.readthedocs.io/en/latest/). However,
there is already a denoising method provided by [PyYAWT](https://pyyawt.readthedocs.io/)
package.


## Main algorithm

In this class the main method cleans up a signal by implementing the following
algorithm:
  1. Preprocess the input signal S by removing any trends (like DC currents),
  and normalize it into the interval [0, 1]. The normalization is optional and
  it doesn't happen automatically. 
  2. Apply a multilevel wavelet decomposition on signal S using the method 
  **wavedec** of PyWavelets. 
  3. Use the detail coefficients cD from step 2 and determine the appropriate
  threshold value. To determine the threshold six different methods can be 
  used (see Threshold methods bellow).
  4. Use the determined threshold value to apply either a soft or a hard 
  thresholding on the detail coefficients. At this stage the user can decide to
  scale the coefficients. This might be necessary if input signal's noise is 
  not normally distributed with mean 0 and std 1.
  5. Reconstruct the signal from the thresholded detail coefficients using the
  function *waverec* of PyWavelets. In case
  a normalization took place on input signal S at the preprocessing stage, 
  renormalize.

> In general, it might be good to scale the input signal before you apply any
denoising method and avoid using normalization.


## Threshold methods supported

There are six methods for determining the threshold so far. These methods are:
  * *Universal*
  * *sqtwolog*
  * *energy*
  * *SURE*
  * *STEIN*
  * *HEURSTEIN*

## Example usage

Here you can find a very simple example of how to instantiate the **WaveletDenoising**
class and how to call its main method **fit()**.

```python
import numpy as np
import matplotlib.pylab as plt

from denoising import WaveletDenoising


t = np.linspace(0, 1, 1000)
freq = 15
y = np.sin(2.0 * np.pi * t * freq) + np.random.normal(0, 1, (1000, ))


wd = WaveletDenoising(normalize=False,
                      wavelet='db3',
                      level=3,
                      thr_mode='soft',
                      selected_level=None,
                      method="universal",
                      resolution=100,
                      energy_perc=0.90)

denoised_y = wd.fit(y)


fig = plt.figure()
ax = fig.add_subplot(121)
ax.plot(t, y)
ax = fig.add_subplot(122)
ax.plot(t, denoised_y)
plt.show()
```


## Dependencies

**wavelet_denoising** requires the following packages:
  * Numpy
  * PyWavelets
  * Scipy
  * Sklearn
  * Matplotlib

You can install all the dependencies by typing the following in your terminal:
```bash
$ pip (or pip3) install -r requirements.txt
```

## Report bugs

In case you would like to report a bug or you experience any problems with the
current repository please open an issue using the [Github Issue Tracker](https://github.com/gdetor/wavelet_denoising/issues)
