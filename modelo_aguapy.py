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




####MODELO######
vw = 0.44671545
phi = 0.30975736
mi_w = 1
mi_g = 0.0172
MRF = 1.0

def PermEff(S):
        lamb = 5.0 #mobilidade total*
        krg = 1.0 #permeabilidade efetiva (gás)
        krw = 0.75 #permeabilidade efetiva (água)
        Swc = 0.99 #Saturação da água*
        Sgr = 0.000 #Saturação do gás*
        Swe = (S-Swc)/(1-S-Sgr)  #equação 7
        #print(Swe)
        k_w = krw*Swe**lamb #equação 8
        k_g = krg*(1-Swe)**(3+(2/5.0)) #equação 9
        #if(S!=0):
            #print(S,k_w,k_g)
        return k_w,k_g


def f(Sw):
   krw1,krg1=PermEff(Sw)
   lw=  (krw1 * 0.2 / mi_w)
   lg=       ((krg1 * 0.00114667) / (mi_g * MRF))
   lt=lw+lg
   #print(lw)
   #return Sw
   return lw/lt

def init(L,dl):
    nl=int(L/dl)
    r= np.zeros((nl,nl))
    return r

import numpy as np
import matplotlib.pyplot as plt

def Solve2d(L: float, dl: float, t: float, dt: float, Sw0: float, times: list) -> list:
    nt, nl = int(t/dt), int(L/dl)

    rw = (vw*dt)/(dl*phi)
    W, U = Poisson.poisson(dl, L)  # Assuming you have Poisson's solver
    W, U = W/np.linalg.norm(W), U/np.linalg.norm(U)

    Sw = np.zeros((nt+1, nl, nl))
    Sw[0] = init(L, dl)

    Sw_list = []

    Sw[0, nl-2:nl, 0] = 0.99
    for k in range(1, nt+1):
        print("done ",k/nt * 100)

        if k*dt in times:
            Sw_list.append(np.copy(Sw[k-1]))

        for i in range(1, nl-1):
            for j in range(1, nl-1):
                C = 10
                v = C*U[i, j]
                w = C*W[i, j]

                if v <= 0:
                    Fy = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i+1, j]))
                else:
                    Fy = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i-1, j]))
                if w <= 0:
                    Fx = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i, j+1]))
                else:
                    Fx = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i, j-1]))

                Sw[k, i, j] = Sw[k-1, i, j] - dt*(np.abs(v)*Fy + np.abs(w)*Fx)
                if Sw[k, i, j] > 1:
                    return Sw_list

        Sw[k, nl-2:nl, 0] = 0.99

    return Sw_list


L = 1
dl = 0.01
t = 10
dt = 0.01
Sw0 = 0
times_of_interest = [1,2,3,4,5,6,7,8,9]  # Adjust this list with the specific times you want

solutions = Solve2d(L, dl, t, dt, Sw0, times_of_interest)

directory = 'agua'  # Replace with your subfolder path
for idx, time in enumerate(times_of_interest):
    ref_csv_filename = f'concentration_{idx+2}.csv'
    ref_csv_path = os.path.join(directory, ref_csv_filename)
    ref_matrix = np.loadtxt(ref_csv_path, delimiter=',')

    calculated_matrix = solutions[idx]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(calculated_matrix, cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Sw')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Calculated Sw at time ' + str(time))

    plt.subplot(1, 2, 2)
    plt.imshow(np.rot90(ref_matrix), cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Sw')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reference Sw at time ' + str(time))

    plt.tight_layout()
    plt.show()
    plt.savefig(directory+"/ajuste_at_"+str(time)+".png")


