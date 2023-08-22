import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sys import argv
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit, prange
import imageio
import cv2
import Poisson


def poisson(dl, L,PRE):
    # Define the problem parameters
    Lx = L  # Length of the domain in x-direction
    Ly = L  # Length of the domain in y-direction
    dx = dy = dl
    Nx = Ny = int(L / dl)

    # Define the boundary conditions
    u_top = 0.0
    u_bottom = 0.0
    u_left = 0.0
    u_right = 0.0

    # Create the grid
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Initialize the u matrix
    u = np.zeros((Ny, Nx))
    u_new = np.copy(u)

    # Apply the boundary conditions
    u[0, :] = u[1, :] - dx * u_top  # Top boundary (Neumann condition)
    u[-1, :] = u[-2, :] + dx * u_bottom  # Bottom boundary (Neumann condition)
    u[:, 0] = u[:, 1] - dy * u_left  # Left boundary (Neumann condition)
    u[:, -1] = u[:, -2] + dy * u_right  # Right boundary (Neumann condition)


    # Source boundary condition
    u[Nx - 2 : , 0:2] = -PRE
    u[Nx - 2 : , 0:2] = -PRE

    u[0: 2, -2:] = PRE
    u[0 : 2, -2:] = PRE

    # Solve the Poisson equation
    max_iter = 1000
    tolerance = 1e-4
    for it in range(max_iter):
        u_new[1:-1, 1:-1] = (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]) / 4

        # Update the boundary conditions at each iteration
        u_new[0, :] = u_new[1, :] - dx * u_top
        u_new[-1, :] = u_new[-2, :] + dx * u_bottom
        u_new[:, 0] = u_new[:, 1] - dy * u_left
        u_new[:, -1] = u_new[:, -2] + dy * u_right

        # Check for convergence
        if np.max(np.abs(u_new - u)) < tolerance:
            break

        u = np.copy(u_new)

    # Calculate the gradients
    grad_y, grad_x= np.gradient(u)
    
    grad_y, grad_x=grad_y/np.linalg.norm(grad_y),    grad_x/np.linalg.norm(grad_x)
    
    # Plot the solution and gradients
    PRE=PRE/500
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    im1 = ax[0, 0].imshow(u, cmap='coolwarm', origin='lower', extent=[0, L, 0, L], vmin=-PRE, vmax=PRE)
    ax[0, 0].set_title('Scalar Field')
    ax[0, 0].set_xlabel('x')
    ax[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=ax[0, 0])

    im2 = ax[0, 1].imshow(grad_x, cmap='coolwarm', origin='lower', extent=[0, L, 0, L], vmin=-PRE, vmax=PRE)
    ax[0, 1].set_title('Gradient in X-direction')
    ax[0, 1].set_xlabel('x')
    ax[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=ax[0, 1])

    im3 = ax[1, 0].imshow(grad_y, cmap='coolwarm', origin='lower', extent=[0, L, 0, L], vmin=-PRE, vmax=PRE)
    ax[1, 0].set_title('Gradient in Y-direction')
    ax[1, 0].set_xlabel('x')
    ax[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=ax[1, 0])

    ax[1, 1].quiver(X, Y, grad_x, grad_y)
    ax[1, 1].set_title('Gradient Vectors')
    ax[1, 1].set_xlabel('x')
    ax[1, 1].set_ylabel('y')

    plt.tight_layout()
    plt.show()
    
    plt.quiver(X,Y,grad_x,grad_y)
    plt.show()
    return grad_x, grad_y

poisson(0.1,1,1)