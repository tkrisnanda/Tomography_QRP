#%%The code retrieves optimised displacements, will be stored in 'final_disps'
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
from scipy.linalg import expm
from scipy.optimize import minimize
from IPython.display import display, clear_output
from qutip import *
from TK_basics import *
import time

# Start timing
start_time = time.time()

# Constants and initializations
FD = 6  # Dimension to read
n_disps = FD**2 - 1  # Number of displacement amplitudes
st = 0.5 # Strength of random initial amplitudes
cdim = 40  # Cavity dimension

# Operators
a = destroy(cdim).full()
adag = a.T.conj()
P = expm(1j * np.pi * adag @ a) # Parity

NN = 5 # Number of different initial random seeds
dx = 0.001 # Infinitesimal for derivatives

# Getting the K matrix, rho = K * Y + Theta
n_para = FD**2 - 1
Ntr = FD**2
X_R = np.zeros([1 + n_para, Ntr], dtype=np.complex128)
X_R[0, :] = np.ones([1, Ntr])
Y_R = np.zeros([FD**2, Ntr], dtype=np.complex128)
for j in np.arange(0, Ntr):
    rd1 = np.zeros([cdim, cdim], dtype=np.complex128)
    u_rand = rand_ket(FD)
    r_rand = (u_rand * u_rand.dag()).full()
    rd1[0:FD, 0:FD] = r_rand

    cw = 1
    for j1 in np.arange(0, FD - 1):
        X_R[cw, j] = rd1[j1, j1].real
        cw += 1
    for j1 in np.arange(0, FD - 1):
        for j2 in np.arange(j1 + 1, FD):
            X_R[cw, j] = rd1[j1, j2].real
            cw += 1
            X_R[cw, j] = rd1[j1, j2].imag
            cw += 1

    Y_R[:, j] = np.transpose((np.transpose(r_rand)).reshape((FD**2, 1)))
Error, beta = QN_regression(X_R, Y_R, 0)
print(f'Error is {Error}')
K = beta[:, 1:n_para + 1]

def wigner_mat(disps):
    ND = len(disps)
    M = np.zeros([ND, FD**2], dtype=complex)
    for k in np.arange(0, ND):
        al = disps[k]
        U = expm(al * adag - np.conj(al) * a)
        MS = U @ P @ np.transpose(np.conj(U))
        Mst = MS[0:FD, 0:FD]
        M[k, :] = Mst.reshape((1, FD**2))
    M_new = np.matmul(M, K)
    return M_new

def wigner_mat_and_grad(disps):
    ND = len(disps)
    wig_tens = wigner_mat(disps)
    grad_mat_r_a = np.zeros((ND, FD**2 - 1), dtype=complex)
    grad_mat_i_a = np.zeros((ND, FD**2 - 1), dtype=complex)

    disps_dr_a = disps + dx
    disps_di_a = disps + dx * 1j

    wig_tens_dr_a = wigner_mat(disps_dr_a)
    wig_tens_di_a = wigner_mat(disps_di_a)

    grad_mat_r_a = (wig_tens_dr_a - wig_tens) / dx
    grad_mat_i_a = (wig_tens_di_a - wig_tens) / dx

    return (wig_tens, grad_mat_r_a, grad_mat_i_a)

def cost_and_grad(r_disps):
    Nn = len(r_disps)
    c_disps = r_disps[:int(Nn / 2)] + 1j * r_disps[int(Nn / 2):]
    M, dM_rs_a, dM_is_a = wigner_mat_and_grad(c_disps)
    U, S, Vd = svd(M)
    NS = len(Vd)
    cn = S[0] / S[-1]
    dS_r_a = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_rs_a, Vd.conj().T).real
    dS_i_a = np.einsum('ij,jk,ki->ij', U.conj().T[:NS], dM_is_a, Vd.conj().T).real

    grad_cn_r_a = (dS_r_a[0] * S[-1] - S[0] * dS_r_a[-1]) / (S[-1]**2)
    grad_cn_i_a = (dS_i_a[0] * S[-1] - S[0] * dS_i_a[-1]) / (S[-1]**2)
    
    return cn, np.concatenate((grad_cn_r_a, grad_cn_i_a))

best_cost = float('inf')

def wrap_cost(disps):
    global best_cost
    cost, grad = cost_and_grad(disps)
    best_cost = min(cost, best_cost)
    return cost, grad

CD = np.zeros(NN)
DIS = np.zeros([NN, n_disps], dtype=np.complex128)
for vv in np.arange(0, NN):
    init_disps = np.random.normal(0, 0.5 * st, 2 * n_disps)
    init_disps[0] = init_disps[n_disps] = 0
    ret = minimize(wrap_cost, init_disps, method='L-BFGS-B', jac=True, options=dict(ftol=1e-6))
    new_disps = ret.x[:n_disps] + 1j * ret.x[n_disps:]

    CD[vv] = ret.fun
    DIS[vv, :] = new_disps

sorted_index = np.argsort(CD)
final_disps = DIS[sorted_index[0], :]
final_CD = CD[sorted_index[0]]

# Plotting the optimum displacements
fig = plt.figure(figsize=(3, 3))
for k in np.arange(0, len(final_disps)):
    plt.plot(final_disps[k].real, final_disps[k].imag, 'ok')
plt.xlabel('Re' + r'$(\alpha)$')
plt.ylabel('Im' + r'$(\alpha)$')
plt.title('CD = %.4f' % (final_CD))
plt.grid()

cabs = np.zeros(len(final_disps))
for j in range(len(final_disps)):
    cabs[j] = np.abs(final_disps[j])
print(f'max dis amplitude is {np.max(cabs)}')

print("")
print("--- %s seconds ---" % (time.time() - start_time))
