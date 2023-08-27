import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from solver import solve_poisson


def poisson(dl, L, q, mi_w, K):
    u=solve_poisson(dl, L, q, mi_w, K)
    Nx=Ny=int(L/dl)
    Lx=Ly=L
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    grad_y, grad_x= np.gradient(u)
        #grad_y, grad_x= grad_y/np.linalg.norm(grad_y),  grad_x/np.linalg.norm(grad_x)
            
    if(True==True):
        # Plot the solution and gradients
            print(np.sum(u),np.mean(u),np.max(u))
            PRE=100
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
    


