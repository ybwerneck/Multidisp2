import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import Poisson




#Variaveis fixas glabais 
vw = 0.024 #velocidade de Darcy para injeção em meio poroso
phi = 0.32142857142 #porosidade
mi_w = 0.001 #viscosidade da água 20 graus 
mi_g = 0.00018 #viscosidade do ar 20 graus (ar)
Swc = 0.99 #Saturação da água* (rever)
Sgr = 0.01 #Gás residual dominio
q = 0.2998155 #velocidade de Darcy (calculada)


def PermEff(S,P):
        lamb = P[0] #mobilidade total*
        krg = P[1] #permeabilidade efetiva (gás)
        krw = P[2] #permeabilidade efetiva (água)

        Swe = (S-Swc)/(1-S-Sgr)  #equação 7
        k_w = krw*Swe**lamb #equação 8
        k_g = krg*(1-Swe)**(3+(2/lamb)) #equação 9
        #if(S!=0):
            #print(S,k_w,k_g)
        k_w = k_w
        k_g = k_g
        return k_w,k_g


def f(Sw,P):
   return Sw
   krw1,krg1=PermEff(Sw, P)
   MRF = P[3]
   if(Sw!=0):   
       print(Sw,krw1,krg1)

   lw = krw1/mi_w
   lg = krg1/ (mi_g * MRF)
   lt=lw+lg
   #print(lw)
   #return Sw
   return lw/lt

def init(L,dl):
    nl=int(L/dl)
    r= np.zeros((nl,nl))
    return r

def Solve2d(L: float, dl: float, t: float, dt: float, Sw0: float, times: list, P: list,Fw:float) -> list:
    nt, nl = int(t/dt), int(L/dl)
    rw = (vw*dt)/(dl*phi)
    krw = P[2]
    W, U = Poisson.poisson(0.1, 10, q, mi_w, krw)  


    Sw = np.zeros((nt+1, nl, nl))
    Sw[0] = init(L, dl)
    D=0.000
    Sw_list = []
    dx=dy=dl
    print(nl)
    Sw[0,   nl-3:nl,0:3] = Sw0
    for k in range(1, nt+1):
        print("done ",round(k/nt * 100, 1))

        if k*dt in times:
            Sw_list.append(np.copy(Sw[k-1]))

        for j in range(1, nl-1):
            for i in range(1, nl-1):
                C = 1#VAZAO/AREA * Q 
                v = C*U[i, j]
                w = C*W[i, j]
                if v <= 0:
                    Fy = (1)*(f(Sw[k-1, i, j], P)-f(Sw[k-1, i+1, j], P))
                else:
                    Fy = (1)*(f(Sw[k-1, i, j], P)-f(Sw[k-1, i-1, j], P))
                if w <= 0:
                    Fx = (1)*(f(Sw[k-1, i, j], P)-f(Sw[k-1, i, j+1], P))
                else:
                    Fx = (1)*(f(Sw[k-1, i, j], P)-f(Sw[k-1, i, j-1], P))
                    
                diffusion_term = D * (
                    (Sw[k-1, i+1, j] - 2 * Sw[k-1, i, j] + Sw[k-1, i-1, j]) / dx**2 +
                    (Sw[k-1, i, j+1] - 2 * Sw[k-1, i, j] + Sw[k-1, i, j-1]) / dy**2
                )
                #print(Fy,Fx)
                Sw[k, i, j] = Sw[k-1, i, j] + dt*(  diffusion_term+ v*Fy + w*Fx)
                if Sw[k, i, j] > 10:
                    return Sw_list

        Sw[k,   nl-3:nl,0:3] = Sw0

    return Sw_list


Area=0.1

L = 1
dl = 0.05 
t = 7
dt = 0.01
Sw0 = 0.99
times_of_interest = [1,3,5,7]  # 

#parametros a serem ajustados
lamb = 3 #mobilidade total*
krg = 10**(-8) #permeabilidade efetiva (gás)
krw = 10**(-9)#permeabilidade efetiva (água)
MRF = 1.5

#-------------------------------
P = [lamb, krg, krw, MRF]
solutions = Solve2d(L, dl, t, dt, Sw0, times_of_interest, P,Fw=1)

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
    plt.imshow(1-np.rot90(ref_matrix), cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Sw')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reference Sw at time ' + str(time))

    plt.tight_layout()
    plt.show()
    plt.savefig(directory+"/ajuste_at_"+str(time)+".png")
