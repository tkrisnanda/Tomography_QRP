# %%
import os
os.chdir('/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2')
import time
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qutip import *
from TK_basics import *
start_time = time.time()  # checking how long the code takes

D = 2  # dimension to read
n_dis = D**2 - 1 # no of displacement points

# list of generated optimized displacements for different truncation dimensions D
if D == 6:
    # for D6
    AL = np.array(
        [
            -0.785123 + 1.48657418j,
            -1.67098242 + 0.22666372j,
            0.23477004 + 0.53155278j,
            -0.98121628 - 0.73382913j,
            1.30034526 - 0.19658333j,
            0.91057304 - 0.14736838j,
            0.65327309 - 1.15538318j,
            -0.1952535 - 0.0984287j,
            -0.36345514 - 1.25492752j,
            -0.86678666 - 0.27228837j,
            0.50962262 + 0.81069781j,
            0.69155442 - 0.69888758j,
            -0.77472546 + 0.48025564j,
            -1.27414151 - 0.11396373j,
            -0.04185742 + 0.91628513j,
            0.25407875 - 1.70695007j,
            0.00196135 - 0.56262436j,
            0.00304715 + 0.18467356j,
            0.19371779 - 0.92074751j,
            -1.2096629 - 1.16284482j,
            -1.159518 + 0.56868829j,
            0.59088723 + 0.14120895j,
            0.84473373 + 0.45523965j,
            0.49191403 - 0.32179364j,
            -0.45441726 - 0.37142953j,
            -0.58965859 + 0.10145214j,
            1.39193867 - 0.97823193j,
            0.96166038 + 0.86471223j,
            -0.24254956 + 0.54631778j,
            1.67299726 + 0.31620016j,
            -0.67203676 + 1.00506992j,
            0.78077124 + 1.52604695j,
            0.12879202 - 0.167258j,
            -0.3002807 - 0.8619771j,
            0.01810127 + 1.30174108j,
        ]
    )
elif D == 5:
    # for D5
    AL = np.array(
        [
            0.24786904 - 0.62486787j,
            -0.65319438 - 0.09095379j,
            0.06631603 - 0.21158938j,
            0.10886298 + 0.65616421j,
            0.55692114 + 0.37281024j,
            0.56994938 + 0.87208238j,
            0.17100201 + 1.45527183j,
            0.17610646 + 0.14161006j,
            -0.35520102 + 0.97711767j,
            1.02708071 + 0.08556895j,
            -0.19244869 + 0.06274509j,
            0.5351312 - 1.34498428j,
            0.00403814 - 1.04398832j,
            -0.64709337 - 1.31903105j,
            -1.44068577 - 0.28722536j,
            1.40487164 - 0.46299343j,
            -0.81763363 - 0.63269653j,
            -1.00999804 + 0.29107477j,
            0.56738559 - 0.1778157j,
            -1.04361651 + 1.03409932j,
            0.81907933 - 0.64184197j,
            -0.48991989 + 0.44367362j,
            -0.27304522 - 0.60725384j,
            1.2172588 + 0.77913231j,
        ]
    )
elif D == 4:
    # for D4
    AL = np.array(
        [
            -1.10109796 + 0.4076309j,
            0.72841022 + 0.12844961j,
            0.46186563 - 0.53473316j,
            0.90345855 + 0.74884819j,
            -0.23118571 - 0.69918282j,
            0.25423305 + 0.05591663j,
            -0.18912888 + 0.18674846j,
            0.25303334 + 0.63805021j,
            1.1057626 - 0.39203478j,
            -0.69865491 - 0.09812627j,
            -0.87461519 - 0.77616706j,
            0.1648075 - 1.15951834j,
            -0.23149483 + 1.150327j,
            -0.07839235 - 0.25912466j,
            -0.45166399 + 0.5841133j,
        ]
    )
elif D == 3:
    # for D3
    AL = np.array(
        [
            -0.83044623 - 0.06700296j,
            0.1161914 - 0.2761754j,
            -0.29281785 + 0.02531395j,
            -0.30062738 + 0.73010012j,
            0.6721621 - 0.48074157j,
            0.18508371 + 0.22372654j,
            -0.19124917 - 0.8107844j,
            0.64486044 + 0.52234013j,
        ]
    )
elif D == 2:
    # for D2
    AL = np.array(
        [0.35792036 - 0.25075055j, -0.39611707 - 0.18459237j, 0.03819648 + 0.43534413j]
    )

AL = -AL # we applied the reverse in experiments, the map has the same condition number (robustness)
nD = D**2 - 1  # no of parameters for general states
Ntr = D**2  # no of training for obtaining the map, at least D^2

cdim = 30  # truncation for simulation
a = destroy(cdim).full()  # annihilation for cavity
adag = a.T.conj()
P = expm(1j * np.pi * adag @ a) # parity operator 

# displacement operator
def Dis(alpha):
    Di = expm(alpha * adag - np.conj(alpha) * a)
    return Di

# this part is for obtaining the map
X_r = np.zeros([1 + n_dis, Ntr])  # store readouts
X_r[0, :] = np.ones([1, Ntr])  # setting the ones
Y_r = np.zeros([nD, Ntr])  # store the targets
for j in np.arange(0, Ntr):
    # qudit mixed state embedded in the cavity mode
    rd1 = np.zeros([cdim, cdim], dtype=np.complex_)
    u_rand = rand_ket(D)
    r_rand = (u_rand * u_rand.dag()).full()
    rd1[0:D, 0:D] = r_rand  # randRho(D)

    # assign targets
    cw = 0
    # diagonal elements
    for j1 in np.arange(0, D - 1):
        Y_r[cw, j] = rd1[j1, j1].real
        cw += 1
    # off-diagonal elements
    for j1 in np.arange(0, D - 1):
        for j2 in np.arange(j1 + 1, D):
            Y_r[cw, j] = rd1[j1, j2].real
            cw += 1
            Y_r[cw, j] = rd1[j1, j2].imag
            cw += 1

    w = 0
    for v in np.arange(0, n_dis):
        Di = Dis(AL[w])
        rt = Di.T.conj() @ rd1 @ Di
        X_r[w + 1, j] = np.trace(rt @ P).real 
        w += 1

# ridge regression
lamb = 0

# training, now to obtain the map
X_R = np.zeros([1 + nD, Ntr])  # will contain the parameters
X_R[0, :] = np.ones([1, Ntr])  # setting the ones
Y_R = np.zeros([n_dis, Ntr])  # will contain the obs

# re-defining variables
X_R[1 : nD + 1, :] = Y_r
Y_R[:, :] = X_r[1 : n_dis + 1, :]

Error, beta = QN_regression(X_R, Y_R, lamb) # beta here is the process map

M = beta[:, 1 : nD + 1]  
W = np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), np.transpose(M))
CN = np.linalg.norm(M, 2) * np.linalg.norm(W, 2)
print(f"Condition number is {CN}")

np.savez(f"map_ideal_D{D}.npz", 
         M = M, W = W, beta = beta, CN = CN)

print("")
print("--- %s seconds ---" % (time.time() - start_time))
