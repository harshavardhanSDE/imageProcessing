import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread("sampleImage.jpg", cv2.IMREAD_GRAYSCALE)

floatImage = np.float32(image)
transformedImage = np.fft.fft2(floatImage)
shiftedTransformedImage = np.fft.fftshift(transformedImage)
magnitudeSpectrum = np.log(np.abs(magnitudeSpectrum))

plt.imshow(magnitudeSpectrum)
plt.title("Magnitude Spectrum")
plt.show()