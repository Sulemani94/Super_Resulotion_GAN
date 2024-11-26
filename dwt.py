"""
Created on Sat Jun 17 19:16:40 2023
@author: KH_Sulemani
"""
# import pywt
import cv2
# import numpy as np
# import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import skimage
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
from DeepNet_arch import psnr

train_dir='results'
img='runway22_tst.png'
original  = cv2.imread(train_dir + "/" + img)

#####################################==============
original=skimage.img_as_float(original)
sigma = 0.001
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 5),
                        sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, multichannel=True , average_sigmas=True)
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
# print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

im_bayes = denoise_wavelet(noisy, convert2ycbcr=True,multichannel = True,wavelet='coif5',
                            method='BayesShrink', mode='soft',wavelet_levels=3,
                            rescale_sigma=True)
im_visushrink = denoise_wavelet(noisy,  convert2ycbcr=True, multichannel = True,
                                method='VisuShrink', mode='soft',
                   wavelet_levels=5,  wavelet='coif5',           sigma=sigma_est/3, rescale_sigma=True)

# VisuShrink is designed to eliminate noise with high probability, but this
# results in a visually over-smooth appearance.

# Compute PSNR as an indication of image quality
psnr_noisy = peak_signal_noise_ratio(original, noisy)
psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
psnr_bayes=psnr(psnr_bayes)
psnr_visushrink = peak_signal_noise_ratio(original, im_visushrink)


ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title(f'Noisy\nPSNR={psnr_noisy:0.4g}')
ax[0, 1].imshow(im_bayes)
ax[0, 1].axis('off')
ax[0, 1].set_title(
    f'Wavelet denoising\n(BayesShrink)\nPSNR={psnr_bayes:0.4g}')
ax[1, 0].imshow(im_visushrink)
ax[1, 0].axis('off')
ax[1, 0].set_title(
    'Wavelet denoising\n(VisuShrink, $\\sigma=\\sigma_{est}$)\n'
      'PSNR=%0.4g' % psnr_visushrink)
ax[1, 1].imshow(original)
ax[1, 1].axis('off')
ax[1, 1].set_title('Original')

fig.tight_layout()

plt.show()
