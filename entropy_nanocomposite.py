import cv2
import numpy as np

# Read the image and convert it to grayscale
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Get the shape of the image
nx, ny = image.shape

# Initialize the probability array to 0
probability = np.zeros(256)

# Loop over all pixels in the image
for i in range(nx):
  for j in range(ny):
    # Get the intensity value of the current pixel
    intensity = image[i, j]
    
    # Increment the count for the current intensity value
    probability[intensity] += 1

# Normalize the probability values by dividing by the total number of pixels
probability /= nx * ny

# Initialize the entropy to 0
entropy = 0

# Loop over all intensity values
for intensity in range(256):
  # Skip intensity values with 0 probability
  if probability[intensity] == 0:
    continue
    
  # Add the contribution of the current intensity value to the entropy
  entropy -= probability[intensity] * np.log2(probability[intensity])

# Print the entropy value
print('Entropy of the image:', entropy)
