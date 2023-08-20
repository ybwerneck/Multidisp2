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

def Solve2d(L: float, dl: float, t: float, dt: float, Sw0: float) -> np.ndarray:
   #dx=dy=dl, X=Y=L
  nt,nl=int(t/dt),int(L/dl)

  rw = (vw*dt)/(dl*phi)
  W,U= Poisson.poisson(dl,L)
  W,U=W/np.linalg.norm(W),U/np.linalg.norm(U)

  Sw=np.zeros((nt+1,nl,nl))
  Sw[0]=init(L,dl)


  ##SOURCE E SINK
  Sw[0,nl - 2 : nl,0]=0.99
  #Sw[k,:,0]=0.99
  for k in range(1,nt):


    ##SOURCE E SINK

    #Sw[k,:,0]=0.99
    print("Done ",k/nt,"%")


    for i in range (1,nl-1):
      for j in range (1,nl-1):
        C=10
        v=C*U[i,j]
        w=C*W[i,j]
        #v=-0.1
        #w=0.5
        #print(v,w)

        if(v<=0):
            Fy=(1/dl)*(f(Sw[k-1,i,j])-f(Sw[k-1,i+1,j]))
        else:
            Fy=(1/dl)*(f(Sw[k-1,i,j])-f(Sw[k-1,i-1,j]))
        if(w<=0):
              Fx=(1/dl)*(f(Sw[k-1,i,j])-f(Sw[k-1,i,j+1]))
        else:
              Fx=(1/dl)*(f(Sw[k-1,i,j])-f(Sw[k-1,i,j-1]))






        Sw[k,i,j]= Sw[k-1,i,j]- dt*(np.abs(v)*Fy + np.abs(w)*Fx)
        if( Sw[k,i,j]>1):
            return Sw

    Sw[k,nl- 2 : nl,0]=0.99


    if(k%100000==0):
        plt.imshow(Sw[k], cmap='viridis', origin='lower', vmin=0, vmax=1)
        plt.colorbar(label='Sw')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Matrix Sw')
        plt.show()
  # border


  return Sw



s=Solve2d(0.5,0.01,3,0.001,0)
print(s)



frames=[]

for k in range(len(s)):
    if(k%25==0):
        plt.imshow(s[k], cmap='viridis', origin='lower', vmin=0, vmax=1)
        plt.colorbar(label='Sw')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Sw t='+str(k*0.001))

        # Save the current figure as an image
        plt.savefig(f'framea_{k}.png')

        # Clear the figure
        plt.clf()

        # Open the saved image and append it to the frames list
        img = Image.open(f'framea_{k}.png')
        frames.append(np.array(img))

        # Remove the saved image file
        img.close()

