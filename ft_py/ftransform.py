import cv2
import numpy as np
import matplotlib.pyplot as plt 

image = cv2.imread("sampleImage.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error in loading the image.")
    exit()

floatImage = np.float32(image)

transformedImage = np.fft.fft2(floatImage)

shiftedTransform = np.fft.fftshift(transformedImage)

magnitudeSpectrum = np.log(np.abs(shiftedTransform))

plt.imshow(magnitudeSpectrum)
plt.title("Magnitude spectrum")
plt.show()