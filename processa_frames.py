# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 00:45:52 2023

@author: yanbw
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 18:27:18 2023

@author: yanbw
"""
from PIL import Image
import numpy as np
import os
import imageio

# Define the directory where your images are located
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Define the directory where your images are located
directory = 'espuma'

# Sort the image file names based on the frame number
image_files = sorted([file for file in os.listdir(directory) if file.endswith('.png')], key=lambda x: int(x.split('.')[0]))
concentration_matrices = []

# Iterate over each image file
for filename in image_files:
    # Load the image
    image_path = os.path.join(directory, filename)
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Apply thresholding to separate the dye from the background
    _, thresholded = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)




    
    concentration_matrix = np.zeros_like(thresholded, dtype=np.float32)
    concentration_matrix = enhanced
    concentration_matrices.append(concentration_matrix)
    

    print(concentration_matrix)
    plt.imshow(concentration_matrix)
    plt.show()
    # Continue with the next image
frames = []
k=0
for concentration_matrix in concentration_matrices:
    # Create a plot of the concentration matrix
    plt.imshow(concentration_matrix)
    plt.axis('off')
    plt.savefig('a.png')
    k=k+1
    
    img = Image.open('a.png')
    frames.append(np.array(img))
        
        # Remove the saved image file
    img.close()
        
    
    # Save the plot images as a GIF
gif_path = str(directory)+'.gif'
imageio.mimsave(gif_path, frames, duration=0.2)