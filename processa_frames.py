# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 23:10:23 2023

@author: yanbw
"""

# Import necessary libraries
import numpy as np
import os
import imageio
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv
from scipy import interpolate

# Define the directory where your images are located
directory = 'ar'

# Sort the image file names based on the frame number
image_files = sorted([file for file in os.listdir(directory) if file.endswith('.png')], key=lambda x: int(x.split('.')[0]))

# Assuming your original matrix is named 'original_matrix'
# Define the new dimensions for the reduced matrix
new_rows = 100  # Desired number of rows
new_cols = 100  # Desired number of columns

# Create grid indices for the original matrix

##AJEITAR PARA CADA CASO
rows, cols = 412, 424
x = np.linspace(0, cols - 1, cols)
y = np.linspace(0, rows - 1, rows)

X, Y = np.meshgrid(x, y)

# Create grid indices for the reduced matrix
new_x = np.linspace(0, cols - 1, new_cols)
new_y = np.linspace(0, rows - 1, new_rows)
new_X, new_Y = np.meshgrid(new_x, new_y)

# Use interpolation to create the reduced matrix
interpolation_method = 'cubic'  # You can choose other methods like 'linear' or 'nearest'

concentration_matrices = []
mask = [0, 0]
b = True

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
    c, thresholded = cv2.threshold(enhanced, 120, 255, cv2.THRESH_BINARY)
    print(np.shape(enhanced))
    
    ##INTERPOLAÇÃO, COMO VOCE TINHA FEITO, FICA BOM NA AGUA E ESPUMA, NO AR NÃO!!
  #  R = reduced_matrix = interpolate.griddata((X.flatten(), Y.flatten()), enhanced.flatten(), (new_X, new_Y), method=interpolation_method)
    R=enhanced
    R = (255 - R) / 255
    if(b == True):
        mask = R
        print("a", mask)
        b = False

    matrix = mask - R
    concentration_matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix))
    
    concentration_matrices.append(concentration_matrix)
    
    # Save concentration_matrix as CSV for each time step
    csv_path = os.path.join(directory, f'concentration_{filename.split(".")[0]}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in concentration_matrix:
            csvwriter.writerow(row)

    # Display the concentration_matrix
    plt.imshow(concentration_matrix, cmap='coolwarm', interpolation='nearest', vmin=0, vmax=1)  # Adjust vmin and vmax
    plt.axis('off')
    plt.colorbar() 
    plt.show()

# Continue with the GIF creation as before
frames = []
for concentration_matrix in concentration_matrices:
    plt.imshow(concentration_matrix, cmap='coolwarm', interpolation='nearest', vmin=0 , vmax=1)  # Adjust vmin and vmax
    plt.axis('off')
    plt.savefig('a.png')
    
    img = Image.open('a.png')
    frames.append(np.array(img))
    
    # Remove the saved image file
    img.close()

# Save the plot images as a GIF
gif_path = os.path.join(directory, 'animation.gif')
imageio.mimsave(gif_path, frames, duration=0.2)
