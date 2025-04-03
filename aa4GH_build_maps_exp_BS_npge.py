#%%
import os
os.chdir('/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2')
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from qutip import *
from TK_basics import *
start_time = time.time()  # checking how long the code takes

# Dimension number and number of observables (number of displacements)
D = 6
Ntr = D**2
RM = D**2 - 1
nD = D**2 - 1

qdim = 3  # GRAPE qubit dim
cdim = 30  # GRAPE cavity dim

# Directory for experimental data
exp_data = f'/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/Data/D{D}'
target_states = '/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/GRAPE_states'

two_pi_half = False

pge_avg = 0#0.032

# 0 - no selection, 1 - selection qubit in g before grape
postSel = 1

# all the possible fock states
master_state_list = np.asarray(
    [
        "fock0",
        "fock01",
        "fock02",
        "fock03",
        "fock04",
        "fock05",
        "fock0i1",
        "fock0i2",
        "fock0i3",
        "fock0i4",
        "fock0i5",
        "fock1",
        "fock12",
        "fock13",
        "fock14",
        "fock15",
        "fock1i2",
        "fock1i3",
        "fock1i4",
        "fock1i5",
        "fock2",
        "fock23",
        "fock24",
        "fock25",
        "fock2i3",
        "fock2i4",
        "fock2i5",
        "fock3",
        "fock34",
        "fock35",
        "fock3i4",
        "fock3i5",
        "fock4",
        "fock45",
        "fock4i5",
        "fock5",
        "fock0-i3",
        "fock1-i3",
        "fock1-i5",
        "fock2-i3",
        "fock2-i4",
        "fock2-i5",
        "fock3-i4",
    ],
    dtype="str",
)

state_list = []
point_list = []

for file in Path(exp_data).glob("*"):
    file_name = file.stem[::]
    state_name = "".join(file_name.split(f"D={D}_grape_")[1].split("_point")[0])
    if state_name not in state_list:
        state_list.append(state_name)
    point = file_name.split("_point")[1]
    if point not in point_list:
        point_list.append(point)

# Sorting helper methods
def find_avg_photon_number(state):
    if len(state) == 5:  # fock0, fock1, fock2...
        return int(state[-1])

    elif len(state) == 6:  # fock01, fock02, fock34...
        return np.mean([int(state[-2]), int(state[-1])])

    elif len(state) == 7:  # fock0i1, fock0i2, fock3i4...
        return np.mean([int(state[-3]), int(state[-1])])
    
    elif len(state) == 8: # fock0-i1, fock0-i2
        return np.mean([int(state[-4]), int(state[-1])])
    
    elif len(state) == 9: # cat-eve-1, cat-odd-1, cat-nop-1
        return 10

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

missing_states_exp = master_state_list[
    np.in1d(master_state_list, state_list_sorted) == False
]

print(f"Missing states from experiment are {missing_states_exp}")

print("Using", len(state_list_sorted), "states, with", len(point_list), "points each.")

print("States:", state_list_sorted)

def select2_flat(filepath, postSel):
    file = h5py.File(filepath, "r")
    data = file["data"]
    # threshold = file["operations/readout_pulse"].attrs["threshold"]
    threshold = 2.9011118282404986e-05
    I = data["I"][::]

    # sweep_points = int(np.shape(x)[0])  # 5
    flat_data = np.array(I.flatten())
    flat_data[flat_data > threshold] = 1
    flat_data[flat_data != 1] = 0

    I_first = flat_data[0::4]
    I_second = flat_data[1::4]

    I_first_m = flat_data[2::4]
    I_second_m = flat_data[3::4]

    if postSel == 1:
        select_mask = np.where(I_first == 0)[0]  # first_selection
        select_mask_m = np.where(I_first_m == 0)[0]
    else:
        select_mask = np.arange(len(I_first))  # no selection
        select_mask_m = np.arange(len(I_first_m))

    # print(len(select_mask))
    thrownrate = 100 * (len(I_first) - len(select_mask)) / len(I_first)
    thrownrate_m = 100 * (len(I_first_m) - len(select_mask_m)) / len(I_first_m)
    # print('{} % data are thrown'.format(thrownrate))

    selected_data = I_second[select_mask]
    selected_data_m = I_second_m[select_mask_m]
    if two_pi_half:
        selected_data = selected_data[0:500]
        selected_data_m = selected_data_m[0:500]
    # print(f'len pi/2 is {len(selected_data)}')
    # print(f'len -pi/2 is {len(selected_data_m)}')

    return selected_data, selected_data_m


# Target state function simulating GRAPE pulses and ideal displacements
def Y_target(state_name, states_directory):
    # Target states
    # if state_name == "vacuum":
    #     rho_tar = fock_dm(cdim, 0)  # Assume cavity in perfect vacuum
    # else:
    file = np.load(states_directory + "/" + state_name + ".npz", "r")
    # 6 is to remove "pulse_"
    rho_tar = Qobj(file["rho"], dims=[[qdim, cdim], [qdim, cdim]]).ptrace(1)

    rho_tar_D = Qobj(rho_tar[0:D, 0:D])  # no need 30 DIMS, CUT AT 6
    rho_tar_D = rho_tar_D / rho_tar_D.tr()  # normalise it, .unit()
    # Target observables
    Y_tar = np.zeros(nD)
    Y_tar[: D - 1] = np.diagonal(rho_tar_D).real[:-1]  # Diagonal of rho
    off_diag = rho_tar_D[np.triu_indices(D, 1)]  # Upper triangle of rho
    Y_tar[D - 1 :: 2] = np.real(off_diag)
    Y_tar[D::2] = np.imag(off_diag)

    return rho_tar_D, Y_tar

all_files = np.array(os.listdir(exp_data))

# this part if for obtaining the map
X_r = np.zeros([1 + RM, Ntr])  # store readouts
X_r[0, :] = np.ones([1, Ntr])  # setting the ones
Y_r = np.zeros([nD, Ntr])  # store the targets

for nB in range(20):#bootstrap

    for j, state_name in enumerate(state_list_sorted):  # State List
        data = np.zeros([len(point_list)])  # Initialize data vector

        # ideal = target_state(state_name)

        for point in range(1, len(point_list) + 1):  # Point1 to Point35
            ending = (
                "D="
                + str(D)
                + "_grape_"
                + str(state_name)
                + "_point"
                + str(point)
                + ".h5"
            )
            matching = [
                filename for filename in all_files if filename.endswith(str(ending))
            ]
            filepath = exp_data + "/" + matching[0]
            # file = h5py.File(filepath, "r")

            selected_data, selected_data_m = select2_flat(filepath, postSel)
            
            #Bootstrap here
            selected_data = np.random.choice(selected_data, size=1000, replace=True)

            signal = np.average(selected_data)
            signal_m = np.average(selected_data_m)

            pge = pge_avg

            # print(pge)
            aux = signal
            aux_m = signal_m

            if two_pi_half:
                data[point - 1] = (aux - aux_m)/(1-2*pge)
            else:
                data[point - 1] = (2 * aux - 1)/(1-2*pge)
            # theory = expect(
            #     fock_dm(cdim, D - 1), displace(cdim, disp_points[point - 1]) * ideal
            # )

            # c = destroy(cdim)
            # print("")

        # Experimental observables
        X_exp = np.zeros([1 + len(data)])
        X_exp[0] = 1
        X_exp[1:] = data
        X_r[:,j] = X_exp

        rho_tar_D, Y_tar = Y_target(state_name, target_states)
        Y_r[:,j] = Y_tar
        
    # ridge regression
    lamb = [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]

    Error = np.zeros(len(lamb), dtype = np.float_)
    for kk in range(len(lamb)):
        # training, now to obtain the map
        X_R = np.zeros([1 + nD, Ntr])  # will contain the parameters
        X_R[0, :] = np.ones([1, Ntr])  # setting the ones
        Y_R = np.zeros([RM, Ntr])  # will contain the obs

        # re-defining variables
        X_R[1 : nD + 1, :] = Y_r
        Y_R[:, :] = X_r[1 : RM + 1, :]

        Error[kk], beta = QN_regression(X_R, Y_R, lamb[kk])

    ind = np.argsort(Error)

    lambmin = lamb[ind[0]]
    Error_f, beta = QN_regression(X_R, Y_R, lambmin)
    M = beta[:, 1 : nD + 1]  # the map
    W = np.matmul(np.linalg.inv(np.matmul(np.transpose(M), M)), np.transpose(M))
    CN = np.linalg.norm(M, 2) * np.linalg.norm(W, 2)
    print(f'condition number is {CN}')
    print(f'error is {Error_f}')

    np.savez(f"/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/Exp_maps_npge/D{D}11/map_exp_D{D}_{nB}.npz", 
            M = M, W = W, beta = beta, CN = CN)

print("")
print("--- %s seconds ---" % (time.time() - start_time))
