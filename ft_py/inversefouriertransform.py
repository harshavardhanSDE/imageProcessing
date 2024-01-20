import cv2
import numpy as np 
import matplotlib.pyplot as plt 

image = cv2.imread("sampleImage.jpg", cv2.IMREAD_GRAYSCALE)

floatImage = np.float32(image)
transformedImage = np.fft.fft2(floatImage)
shiftedTransformedImage = np.fft.fftshift(transformedImage)

inverseTransformedImage = np.fft.ifft2(np.fft.ifftshift(shiftedTransformedImage))
inverseTransformedImage = np.abs(inverseTransformedImage).astype(np.uint8)


plt.imshow(inverseTransformedImage)
plt.title("Reconstructed Image")
plt.show()