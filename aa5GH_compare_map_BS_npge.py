#%%This code reproduces Fig3
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2')

def create_figure_cm(width_cm, height_cm):
    # Convert cm to inches
    width_inch = (width_cm) / 2.54
    height_inch = (height_cm) / 2.54
    
    # Create the figure with specified size in inches
    plt.figure(figsize=(width_inch, height_inch))

DD = np.arange(2, 7)

title_font_size = 8
axis_label_font_size = 8

for j, d in enumerate(DD):
    D = d
    data_ideal = np.load("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/map_ideal_D"+str(D)+".npz", 'r')
    data_exp = np.load("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/map_exp_npge_D"+str(D)+".npz", 'r')
    
    beta_ideal = data_ideal["beta"]
    beta_exp = data_exp["beta"]

    vbeta_ideal = beta_ideal.flatten()
    vbeta_exp = beta_exp.flatten()
    x = np.linspace(-2,2,5)

    create_figure_cm(4.5,3.5)
    plt.plot(vbeta_exp, vbeta_ideal, 'ok', markersize=2)
    plt.plot(x,x, 'b-')
    plt.xlabel(rf'$\beta_L$', fontsize = axis_label_font_size)
    plt.ylabel(rf'$\beta_I$', fontsize = axis_label_font_size)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.tick_params(labelsize=title_font_size)
    plt.title(rf'Dimension $D={D}$', fontsize = title_font_size)

    # plt.legend(fontsize=title_font_size)
    # plt.savefig(rf'fig2_Map_D{D}.pdf')
    plt.show()

#%%with bootstrap
d1mean = np.zeros(len(DD), dtype=np.float_)
d1std = np.zeros(len(DD), dtype=np.float_)
# d2 = np.zeros(len(DD), dtype=np.float_)
NB = 20#number of bootstraps
for j, d in enumerate(DD):
    d1nB = np.zeros(NB, dtype = np.float_)
    for nB in range(NB):
        D = d
        data_ideal = np.load("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/map_ideal_D"+str(D)+".npz", 'r')
        data_exp = np.load("/Users/tanjungkrisnanda/Library/CloudStorage/Dropbox/NTU Grad/Research/Python codes/nus_tomo_23/20240606_exp_learning_map_2/Exp_maps_npge/D"+str(D)+"/map_exp_D"+str(D)+"_"+str(nB)+".npz", 'r')

        beta_ideal = data_ideal["beta"]
        beta_exp = data_exp["beta"]
        # beta_sim = data_sim["beta"]

        vbeta_ideal = beta_ideal.flatten()
        vbeta_exp = beta_exp.flatten()
        # vbeta_sim = beta_sim.flatten()

        No_el = len(vbeta_ideal)
        d1nB[nB] = np.sum(np.abs(vbeta_ideal - vbeta_exp))/No_el
    d1mean[j] = np.mean(d1nB)
    d1std[j] = np.std(d1nB)

# create_figure_cm(5,4)
create_figure_cm(8,5.5)
plt.errorbar(DD, d1mean, yerr = d1std, fmt='ok', markersize=3, capsize=3, elinewidth=1)#, label='Ideal-sim')
# plt.plot(DD, d_exp_sim, 'ok', label='Learned-sim')
# plt.legend(fontsize=title_font_size)
plt.xlabel(rf'Dimension $D$', fontsize=title_font_size)
plt.xticks(np.arange(2,7))
# plt.ylabel(rf'Normalised $||\beta_I-\beta_L||$', fontsize=title_font_size)
plt.ylabel(rf'Average difference', fontsize=title_font_size)
plt.tick_params(labelsize=title_font_size)
# plt.savefig(rf'fig2_Map_error.pdf')
plt.show()
