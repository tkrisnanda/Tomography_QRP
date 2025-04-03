#%%
"""
This file requires GRAPE pulses and simulates the expected target states 
"""
import os
#working in the current dir
os.chdir('/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2')
import numpy as np
from qutip import *
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

simulate = True
save_simulated_state = True

# Directories
grape_directory = '/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/GRAPE_pulses_GH'
states_directory = '/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/GRAPE_states_GH'

# For checking if the pulses are out of range
cav_scale_factor = 73.7
qubit_scale_factor = 81.97
cav_max_amp = 0.2 / cav_scale_factor
qubit_max_amp = 0.2 / qubit_scale_factor

# Simulation Dimensions
cdim = 30
qdim = 3
al = 1# amplitude for cat states

# Define  mean thermal excitation
nbar_cav = 0.03
nbar_qb = 0#0.009  # we use pre-selection

# Hamiltonian Parameters in GHz
chi = 1.423e-3
chi_prime = 16e-6
Kerr = 6e-6
alpha = 175.3e-3

# Coherences in ns
T1 = 85e3
T2 = 14e3
Tphi = 1 / (1 / T2 - 0.5 / T1)
cavT1 = 0.992e6

# Mode Operators
q = destroy(qdim)
c = destroy(cdim)
qd, cd = q.dag(), c.dag()

Q = tensor(q, qeye(cdim))
C = tensor(qeye(qdim), c)
Qd, Cd = Q.dag(), C.dag()

# Collect states
state_list = []
for file in Path(grape_directory).glob("*"):
    state_list.append(file.stem)

# Sorting helper methods
def find_avg_photon_number(state):
        if len(state) == 5:  # fock0, fock1, fock2...
            return int(state[-1])

        elif len(state) == 6:  # fock01, fock02, fock34...
            return np.mean([int(state[-2]), int(state[-1])])

        elif len(state) == 7:  # fock0i1, fock0i2, fock3i4...
            return np.mean([int(state[-3]), int(state[-1])])
        
        elif len(state) == 8:  #fock0-i3, fock1-i3, ...
            return np.mean([int(state[-4]), int(state[-1])])
        
        elif state == 'cat-eve-1':  
            return expect(c.dag()*c, (coherent(cdim,al)+coherent(cdim,-al)).unit())
        
        elif state == 'cat-odd-1':  
            return expect(c.dag()*c, (coherent(cdim,al)-coherent(cdim,-al)).unit())
        
        elif state == 'cat-nop-1':  
            return expect(c.dag()*c, (coherent(cdim,al)+1j*coherent(cdim,-al)).unit())
        
        elif state == 'cat-nmp-1':  
            return expect(c.dag()*c, (coherent(cdim,al)-1j*coherent(cdim,-al)).unit())

        else:
            print("State ", state, "invalid!")
            pass

def sort_state_list_by_photon_number(alphabetical_state_list):
    """Takes in an alphabetically sorted state_list"""
    state_avg_photon_number = [
        find_avg_photon_number(x) for x in alphabetical_state_list
    ]
    # for i, state in enumerate(state_list):
    state_avg_photon_number = np.asarray(state_avg_photon_number)

    ind = np.lexsort((alphabetical_state_list, state_avg_photon_number))

    sorted = np.asarray(
        [(alphabetical_state_list[i], state_avg_photon_number[i]) for i in ind]
    )
    return ind, sorted[:, 0]


# Sort the state list in increasing photon number
alphabetical_state_list = np.sort(np.asarray(state_list))
ind, state_list_sorted = sort_state_list_by_photon_number(alphabetical_state_list)

# Initial State
initial = tensor(thermal_dm(qdim, nbar_qb), thermal_dm(cdim, nbar_cav))

# Collapse Operators
c_ops = [
    np.sqrt((1 + nbar_qb) / T1) * Q,  # Qubit Relaxation
    np.sqrt(nbar_qb / T1) * Qd,  # Qubit Thermal Excitations
    np.sqrt(2 / Tphi) * Qd * Q,  # Qubit Dephasing
    np.sqrt((1 + nbar_cav) / cavT1) * C,  # Cavity Relaxation
    np.sqrt(nbar_cav / cavT1) * Cd,  # Cavity Thermal Excitations
]

# Drift Hamiltonian
H0 = (
    - 2 * np.pi * chi * Cd * C * Qd * Q
    - 2 * np.pi * chi_prime / 2 * Qd * Q * Cd * Cd * C * C
    - 2 * np.pi * Kerr / 2 * Cd * Cd * C * C
    - 2 * np.pi * alpha / 2 * Qd * Qd * Q * Q
)

# fock target states
def target_state(name):
    if len(name) == 5:  # fock0, fock1, fock2...
        target = fock(cdim, int(name[-1]))
        return target

    elif len(name) == 6:  # fock01, fock02, fock34...
        target = (fock(cdim, int(name[-2])) + fock(cdim, int(name[-1]))).unit()
        return target

    elif len(name) == 7:  # fock0i1, fock0i2, fock3i4...
        target = (fock(cdim, int(name[-3])) + 1j * fock(cdim, int(name[-1]))).unit()
        return target
    
    elif len(name) == 8: # fock0-i3, fock1-i3, ...
        target = (fock(cdim, int(name[-4])) - 1j * fock(cdim, int(name[-1]))).unit()
        return target

    elif name == 'cat-eve-1':  
        target = (coherent(cdim,al)+coherent(cdim,-al)).unit()
        return target
        
    elif name == 'cat-odd-1':  
        target = (coherent(cdim,al)-coherent(cdim,-al)).unit()
        return target
    
    elif name == 'cat-nop-1':  
        target = (coherent(cdim,al)+1j*coherent(cdim,-al)).unit()
        return target
    
    elif name == 'cat-nmp-1':  
        target = (coherent(cdim,al)-1j*coherent(cdim,-al)).unit()
        return target

    else:
        print("State ", name, "invalid!")
        pass

def drive_amp(t, dt, drive):
    """Returns the drive amplitude for a given time"""
    drive_index = int(t // dt)

    if drive_index == len(cavQ):
        drive_index -= 1

    return drive[drive_index]

# Data to be collected
F = np.zeros(len(state_list_sorted))
Pe = np.zeros(len(state_list_sorted))
overdrive = np.zeros(len(state_list_sorted))

maxQubitI = np.zeros(len(state_list_sorted))
maxQubitQ = np.zeros(len(state_list_sorted))
maxCavI = np.zeros(len(state_list_sorted))
maxCavQ = np.zeros(len(state_list_sorted))
# For each state, load the grape pulse, check overdrive, simulte the state creation

for i, name in enumerate(state_list_sorted):
    print(f"State: {name}, {i} out of {len(state_list_sorted)}")
    target = target_state(name)

    data = np.load(grape_directory + "/" + name + ".npz", "r")

    dt = data["dt"]
    qubitI = data["QubitI"]
    qubitQ = data["QubitQ"]
    cavI = data["CavityI"]
    cavQ = data["CavityQ"]

    maxQubitI[i] = np.max(qubitI)
    maxQubitQ[i] = np.max(qubitQ)
    maxCavI[i] = np.max(cavI)
    maxCavQ[i] = np.max(cavQ)

    if (
        maxQubitI[i] > qubit_max_amp
        or maxQubitQ[i] > qubit_max_amp
        or maxCavI[i] > cav_max_amp
        or maxCavQ[i] > cav_max_amp
    ):
        overdrive[i] = True
        print("OVERDRIVE PULSE!!")
        print(f"max cav amp = {cav_max_amp}")
        print(f"maxCavI = {maxCavI}")
        print(f"maxCavQ = {maxCavQ}")
        print()
        print(f"max qubit amp = {qubit_max_amp}")
        print(f"maxQubitI = {maxQubitI}")
        print(f"maxQubitQ = {maxQubitQ}")
        print()
    else:
        overdrive[i] = False

    if simulate:
        tlist = [dt * i for i in range(len(cavQ))]

        H_drive = [
            [2 * np.pi * (Q + Qd), lambda t, *args: drive_amp(t, dt, qubitI)],
            [2j * np.pi * (Q - Qd), lambda t, *args: drive_amp(t, dt, qubitQ)],
            [2 * np.pi * (C + Cd), lambda t, *args: drive_amp(t, dt, cavI)],
            [2j * np.pi * (C - Cd), lambda t, *args: drive_amp(t, dt, cavQ)],
        ]

        H = [H0, *H_drive]

        # Dynamics
        options = Options(max_step=2, nsteps=1e6)
        results = mesolve(
            H,
            initial,
            tlist,
            c_ops=c_ops,
            options=options,
        )  # progress_bar= True)

        if save_simulated_state:
            np.savez(states_directory + "/" + name + ".npz", rho=results.states[-1])

        cav_state = results.states[-1].ptrace(1)
        qubit_state = results.states[-1].ptrace(0)

        # plot_wigner(cav_state)

        F[i] = np.round(fidelity(cav_state, target)**2, 3)
        Pe[i] = expect(fock_dm(qdim, 1), qubit_state)

        print("Fidelity: ", F[i])
        print("Pe after grape: ", Pe[i])
        print()

fock0_rho = tensor(thermal_dm(qdim, nbar_qb), thermal_dm(cdim, nbar_cav))

np.savez(states_directory + "/fock0.npz", rho=fock0_rho)

print(f"The overdrive pulses are {state_list_sorted[overdrive>0]}")
print(f"The low fidelity states are {state_list_sorted[F<0.95]}")
print(f"The high Pe > 5% after grape states are {state_list_sorted[Pe > 0.05]}")

plt.figure()
plt.xticks(rotation=90)
plt.plot(state_list_sorted, maxQubitI, "-", label="qubit I")

plt.plot(state_list_sorted, maxQubitQ, "-", label="qubit I")
plt.plot(state_list_sorted, maxCavI, "-", label="cav I")
plt.plot(state_list_sorted, maxCavQ, "-", label="cav Q")
plt.axhline(y=qubit_max_amp, color="purple", label="max qubit amp", linestyle="--")
plt.axhline(y=cav_max_amp, color="pink", label="max cav amp", linestyle="--")

plt.legend()

plt.figure()
plt.xticks(rotation=90)
plt.plot(state_list_sorted, Pe)
plt.title("Pe after grape")
plt.show()
