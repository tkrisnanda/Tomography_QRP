#%%
''''
This code applies the map (idealised and learnt) to the experimental data (observables) to obtain 
the estimated density matrices via Bayesian inference, and calculate the Fidelity
Takes about 30 mins with bootstrap (20 times)
'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from qutip import *
import os
#current dir for saving the map
os.chdir('/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2')
from TK_basics import *
import time
start_time = time.time()  # checking how long the code takes

NB = 20#number of bootstraps
applyMap = True
# compare_with_sim_fidelity = False
# compare_with_Pe_after_grape = False
two_pi_half = False
# variable_pe_after_grape = False
# master_states = False

pge_avg = 0#0.032

# 0 - no selection, 1 - selection qubit in g before grape
postSel = 1

# Dimension number and number of observables (number of displacements)
D = 6
Nte = 4
nD = D**2 - 1

maplist = ['ideal', 'exp_npge']

Fmean = np.zeros([Nte, len(maplist)], dtype=float)
Fstd = np.zeros([Nte, len(maplist)], dtype=float)
# errY = np.zeros([Nte, len(maplist)], dtype=float)
for kk in range(len(maplist)):
    # Load inverse map variables for converting pe to rho
    data = np.load(f"/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/map_"+maplist[kk]+f"_D{D}.npz")

    W = data["W"]
    beta = data["beta"]

    exp_data = '/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/Data/Test_cat'
    target_states = '/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/GRAPE_states_GH'

    qdim = 3  # GRAPE qubit dim
    cdim = 30  # GRAPE cavity dim

    # Gather all the states and points from the data that was collected

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
    # state_list = ['fock0']

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
        if state_name == "vacuum":
            rho_tar = fock_dm(cdim, 0)  # Assume cavity in perfect vacuum
        else:
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


    # Builds a density matrix from the vector Y
    def rho_from_Y(Y_est):
        rho_est = np.zeros([D, D], dtype=np.complex_)
        diagonal = np.append(Y_est[: D - 1], 1 - sum(Y_est[: D - 1]))
        np.fill_diagonal(rho_est, diagonal)  # Populate diagonal of rho

        index_i_list = np.triu_indices(D, 1)[0]
        index_j_list = np.triu_indices(D, 1)[1]
        for k in range(len(index_i_list)):  # Populate off-diagonals of rho
            index_i = index_i_list[k]
            index_j = index_j_list[k]
            rho_est[index_i, index_j] = Y_est[D - 1 + 2 * k] + 1j * Y_est[D + 2 * k]
            rho_est[index_j, index_i] = Y_est[D - 1 + 2 * k] - 1j * Y_est[D + 2 * k]

        return Qobj(rho_est)


    Obs_exp = np.zeros([len(point_list), len(state_list_sorted)])
    all_files = np.array(os.listdir(exp_data))

    # Mapping
    C = np.matmul(-W, beta[:, 0])
    BETA = np.zeros([nD, nD + 1])
    BETA[:, 0] = C
    BETA[:, 1 : nD + 1] = W

    for j, state_name in enumerate(state_list_sorted):  # State List
        data = np.zeros([len(point_list)])  # Initialize data vector

        f1 = np.zeros(NB, dtype = float)
        for nB in range(NB):
        
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
                file = h5py.File(filepath, "r")

                selected_data, selected_data_m = select2_flat(filepath, postSel)
                
                #bootstrap here
                selected_data = np.random.choice(selected_data, size=1000, replace=True)

                signal = np.average(selected_data)
                signal_m = np.average(selected_data_m)
                
                pge = pge_avg

                # print(pge)
                aux = signal#(signal - pge) / (1 - 2 * pge)
                aux_m = signal_m#(signal_m - pge) / (1 - 2 * pge)

                if two_pi_half:
                    data[point - 1] = (aux - aux_m)/(1 - 2 * pge)
                else:
                    data[point - 1] = (2 * aux - 1)/(1 - 2 * pge)

            # Experimental observables
            X_exp = np.zeros([1 + len(data)])
            X_exp[0] = 1
            X_exp[1:] = data

            rho_tar_D, Y_tar = Y_target(state_name, target_states)

            # Estimate the state by applying the inverse map to the experimental data
            Y_est = np.zeros(nD)
            Y_est = np.matmul(BETA, X_exp)
            rho_est = rho_from_Y(Y_est)  # just a reshaping
            _, _, qRho_est = Bysn_rho_v2(2**10, 1000*(nD), rho_tar_D.full(), rho_est.full())
            # qRho_est = PSD_rho(rho_est)

            f1[nB] = fidelity(rho_tar_D, Qobj(qRho_est)) ** 2

            # diff = Y_tar - Y_est
            # errY[j, kk] = np.linalg.norm(diff)/np.linalg.norm(Y_tar)
        Fmean[j, kk] = np.mean(f1)
        Fstd[j, kk] = np.std(f1)

print("")
print("--- %s seconds ---" % (time.time() - start_time))

#%%plot figure
def create_figure_cm(width_cm, height_cm):
    # Convert cm to inches
    width_inch = (width_cm) / 2.54
    height_inch = (height_cm) / 2.54
    
    # Create the figure with specified size in inches
    plt.figure(figsize=(width_inch, height_inch))
    
font_size = 8
create_figure_cm(5,4)
# plt.xticks(rotation=90)
plt.ylim((0.4, 1.0))
plt.yticks(np.linspace(0.4,1,7))
plt.ylabel('Fidelities', fontsize=font_size)
plt.errorbar(state_list_sorted, Fmean[:,0], yerr=Fstd[:,0], fmt='^r', markersize=3, capsize=3, elinewidth=1, label="Ideal map")
plt.errorbar(state_list_sorted, Fmean[:,1],  yerr=Fstd[:,1], fmt='ok', markersize=3, capsize=3, elinewidth=1, label="Learned map")

plt.legend(fontsize = font_size)
plt.tick_params(labelsize = font_size)
# plt.savefig('fig4_fids.pdf')
plt.show()

