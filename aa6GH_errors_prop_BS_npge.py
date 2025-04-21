#%%This code reproduces Fig4
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
from qutip import *
import os
os.chdir('/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2')
from TK_basics import *
import time
start_time = time.time()  # checking how long the code takes


two_pi_half = False
NB = 20

pge_avg = 0#0.032

# 0 - no selection, 1 - selection qubit in g before grape
postSel = 1

# Dimension number and number of observables (number of displacements)
D = 6
Nte = 4
nD = D**2 - 1

maplist = ['ideal', 'exp_npge']

errmean = np.zeros([Nte, len(maplist)], dtype=float)
errstd = np.zeros([Nte, len(maplist)], dtype=float)
for kk in range(len(maplist)):
    data = np.load(f"/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/map_"+maplist[kk]+f"_D{D}.npz")

    M = data["M"]
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
        Y_tar[: D - 1] = np.diagonal(rho_tar_D.full()).real[:-1]  # Diagonal of rho
        off_diag = rho_tar_D[np.triu_indices(D, 1)]  # Upper triangle of rho
        Y_tar[D - 1 :: 2] = np.real(off_diag)
        Y_tar[D::2] = np.imag(off_diag)

        return rho_tar_D, Y_tar


    # Builds a density matrix from the vector Y
    def rho_from_Y(Y_est):
        rho_est = np.zeros([D, D], dtype=np.complex128)
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

    all_files = np.array(os.listdir(exp_data))

    for j, state_name in enumerate(state_list_sorted):  # State List
        e1 = np.zeros(NB, dtype = float)
        for nB in range(NB):
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

            # Experimental observables (what it is)
            X_exp = np.zeros([len(data)])
            X_exp = data
            Xp_exp = X_exp#-V_exp

            #what it should be assuming the map and given Y_tar
            rho_tar_D, Y_tar = Y_target(state_name, target_states)
            XXp = M@Y_tar + beta[:,0]

            dX = XXp-Xp_exp
            # err[j, kk] = np.sum(np.abs(dX))/len(dX)#np.linalg.norm(dX)#
            # e1[nB] = np.linalg.norm(dX)
            e1[nB] = np.sum(np.abs(dX)**2)/len(dX)
        errmean[j, kk] = np.mean(e1)
        errstd[j, kk] = np.std(e1)

def create_figure_cm(width_cm, height_cm):
    # Convert cm to inches
    width_inch = (width_cm) / 2.54
    height_inch = (height_cm) / 2.54
    
    # Create the figure with specified size in inches
    plt.figure(figsize=(width_inch, height_inch))

print("")
print("--- %s seconds ---" % (time.time() - start_time))

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example data
categories = ['A', 'B', 'C', 'D']
values_type1 = errmean[:,0].tolist()#[5, 20, 15, 25]
values_type2 = errmean[:,1].tolist()#[40, 22, 18, 28]
errors_type1 = errstd[:,0].tolist()#[10, 2, 1.2, 2.1]
errors_type2 = errstd[:,1].tolist()#[20, 1.8, 1.7, 1.9]

# Create a DataFrame
data = {
    'Category': categories * 2,
    'Type': ['Analytical map'] * len(categories) + ['Learned map'] * len(categories),
    'Value': values_type1 + values_type2,
    'Error': errors_type1 + errors_type2
}
df = pd.DataFrame(data)

# Plotting
# plt.figure(figsize=(10, 6))
create_figure_cm(9,6.5)
barplot = sns.barplot(x='Category', y='Value', hue='Type', data=df, capsize=0.1, legend=False)

# Adding error bars manually
for i in range(len(categories)):
    # Type 1
    barplot.errorbar(i - 0.2, values_type1[i], yerr=errors_type1[i], fmt='none', c='black', capsize=5)
    # Type 2
    barplot.errorbar(i + 0.2, values_type2[i], yerr=errors_type2[i], fmt='none', c='black', capsize=5)

# plt.title('Barplot with Specified Error Bars for Type 1 and Type 2')
plt.xlabel('Category')
plt.ylabel('Value')
# plt.ylim(0.035, 0.112)
plt.ylim(0.0015, 0.02)
# plt.legend(title='Type')
plt.savefig('fig3_D6.pdf')
plt.show()




