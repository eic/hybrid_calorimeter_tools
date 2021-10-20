'''
    efficiency_fit.py

    Functions for hybrid calorimeter analysis on the EIC.

    Authors:    Nathan Branson, Dmitry Romanov
    Updated:    10/20/2021
'''

# TODO Need to change when it plots.  Currently plotting in build_eff_plots.
# TODO open_files should be done by the user because their files will be different obviously. I put opening the files in main for that reason.
# TODO "... = check_calibration(input_data??)" and "axes = plot_fit_results([(ind_fit_result, ind_plot_axes)], ax=ax)" need to be done 
#       if that is still in the plan.


#%matplotlib inline
import uproot4
#from ROOT import TH1F
from hist import Hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib import cm
from matplotlib.colors import LogNorm, Normalize
from uncertainties import unumpy as unp
import numpy as np
import awkward1 as ak
from scipy.optimize import curve_fit
from scipy.stats import crystalball
from scipy.special import erf
from zfit.models.physics import crystalball_func
import math


mpl.rcParams['text.usetex'] = True
plt.rc('text', usetex=False)


######################################################################################################################################
# -------------- Functions
######################################################################################################################################
''' --- CRYSTAL BALL FUNCTION --- '''
### --- https://en.wikipedia.org/wiki/Crystal_Ball_function
def crystalball(x, alpha, n, mean, sigma, N):#N=amp
    func = np.array([])
    for x in x:
        A = np.float_power(n/abs(alpha),n)*np.exp(-1*(alpha**2)/2)
        B = (n/abs(alpha))-abs(alpha)
        if((x-mean)/sigma > -1*alpha):
            func = np.append(func, [N*np.exp(-1*((x-mean)**2)/(2*(sigma**2)))])
        elif((x-mean)/sigma <= -1*alpha):
            func = np.append(func, [N*A*((np.float_power((B-(x-mean)/sigma),(-1*n))))])
    return func

### --- Y = c2/sqrt(E) + c1/E + c0
def efficiency_fit(E, c0, c1, c2):
    return (c2/np.sqrt(E))+(c1/E)+c0


### --- https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array,value):
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx
######################################################################################################################################
######################################################################################################################################


######################################################################################################################################
# -------------- Fits the energy plots to crystal ball function for energy reconstruction
# returns:  [x_interval_for_fit, reco_fit]  - fitted line from crystal ball function
#           ax                              - axes 
#           mean                            - mean result (reconstructed energy) from fitted line
#           std                             - standard dev of fitted line
######################################################################################################################################
def fit_crystal_ball(true_energy, histogram):
    ### reduce x range to where energy is
    bin_centers = histogram[1][:-1] + np.diff(histogram[1]) / 2
    energy = np.argmax(histogram[0])
    e_reco = histogram[1][energy]
    e_min = e_reco-3
    e_max = e_reco+1
    arg_min = find_nearest(histogram[1], e_min)#; arg_min = arg_min[0][0]
    arg_max = find_nearest(histogram[1], e_max)#; arg_max = arg_max[0][0]

    fig, ax = plt.subplots()
    ### plot histogram
    width = .85*(histogram[1][1] - histogram[1][0])
    plt.bar(histogram[1][0:500], histogram[0], align='edge', width=width)

    ### plot fit
    beta, m, scale = true_energy, 3, .4
    x_interval_for_fit = np.linspace(histogram[1][arg_min], histogram[1][arg_max], 10000)
    popt, _ = curve_fit(crystalball, bin_centers, histogram[0], p0=[1, 1, histogram[1][energy], .05, max(histogram[0])])
    reco_fit = crystalball(x_interval_for_fit, *popt) # *popt = alpha, n, mean, sigma, amp
    ax.plot(x_interval_for_fit, reco_fit, label='fit', color="orange")
    mean = popt[2]
    std = popt[3]
    # place a text box in upper left in axes coords
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ### --- print mean and std on graph
    textstr = '\n'.join((r'$\mu=%.4f$' % (round(mean,4), ), r'$\sigma=%.4f$' % (round(std, 4), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.legend()
    ax.set_xlabel("Energy")
    ax.set_ylabel("Events")

    return [x_interval_for_fit, reco_fit], ax, mean, std
    ### [x axis, crystal ball fit], mean, standard deviation 

######################################################################################################################################
# -------------- builds the energy plots
# returns:  eff_fit         - fit of the efficiency plot std/energy vs energy
#           eff_axes        - axes of the efficiency plot
#           ind_fit_result  - [ind fit, axes] individual reconstructed fits and axes from the crystal ball fits
######################################################################################################################################
def build_eff_plot(input_data, reco_fitter=fit_crystal_ball, eff_fitter=efficiency_fit):
    mean_list = []
    std_list = []
    true_energy_list = []
    ax_list = []
    fit_result_list = []
    for i in range(len(input_data)):
        fit_result, ax, mean, std = reco_fitter(input_data[i][0],input_data[i][1])
        fit_result_list.append(fit_result)
        mean_list.append(mean)
        std_list.append(std)
        ax_list.append(ax)
        true_energy_list.append(input_data[i][0])
    
    ### mean/energy ratio
    mean_e_r = []
    ### std/energy ratio
    std_e_r = []

    for i in range(len(mean_list)):
        mean_e_r.append(mean_list[i]/true_energy_list[i]*100)
        std_e_r.append((std_list[i]/true_energy_list[i])*100)

    x_interval_for_fit = np.linspace(np.min(true_energy_list)-.75, np.max(true_energy_list)+2, 10000)
    popt, _ = curve_fit(eff_fitter, true_energy_list, std_e_r, p0=[1, 1, 0])

    eff_fit = eff_fitter(x_interval_for_fit, *popt)

    eff_axes = graph_energy_plots(true_energy_list, mean_list, std_list, x_interval_for_fit, eff_fit)

    ind_fit_result = []
    for i in range(len(fit_result_list)):
        ind_fit_result.append([fit_result_list[i], ax_list[i]])

    return eff_fit, eff_axes, ind_fit_result
    
######################################################################################################################################
# -------------- Graphs the energy plots
# returns:  ax          - axes of plots
######################################################################################################################################
def graph_energy_plots(energies, mean_list, std_list, x_interval_for_fit, fit):
    # energies  =   true energy
    # mean_list =   reconstructed energies from fits
    # std_list  =   standard deviations from fits

    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    
    plt.suptitle(r"Efficiency Plots", fontsize=45)

    ### --- Reconstructed Energy vs True Energy --- ###
    ax[0][0].set_title("Reconstructed Energy vs True Energy")#, fontsize=45)
    ax[0][0].scatter(mean_list, energies)#, s=100)
    ax[0][0].set_ylabel("Reconstructed Energy (GeV)")#, fontsize=40)
    ax[0][0].set_xlabel("True Energy (GeV)")#, fontsize=40)
    ax[0][0].set_yticks(range(math.floor(min(mean_list)), math.ceil(max(mean_list))+1))#, fontsize=35)
    ax[0][0].set_xticks(range(math.floor(min(energies)), math.ceil(max(energies))+1))#, fontsize=35)
    #plt.savefig(f"Fits_2/y40cm_recovsreal",facecolor='w')
    #plt.clf()

    ### --- Reconstructed Energy/True Energy x 100% vs Energy --- ###
    ratio = []
    for i in range(len(mean_list)):
        ratio.append(mean_list[i]/energies[i]*100)
    ax[0][1].set_title(r'Reconstructed Energy/True Energy $\times$ 100% vs True Energy')#, fontsize=45)
    ax[0][1].scatter(energies, ratio)#, s=100)
    ax[0][1].set_xlabel("Energy (GeV)")#, fontsize=40)
    ax[0][1].set_ylabel(r'Reconstructed Energy/True Energy $\times$ 100%')#, fontsize=36)
    ax[0][1].set_xticks(range(math.floor(min(energies)), math.ceil(max(energies))+1))#, fontsize=35)
    #ax[0][1].set_yticks()#fontsize=35)
    #plt.savefig(f"Fits_2/y40cm_recorealEvsE",facecolor='w')
    #plt.clf()

    ### --- Sigma Energy vs Energy --- ###
    ax[1][0].set_title(r'$\sigma$ Energy vs True Energy')#, fontsize=45)
    ax[1][0].scatter(energies, std_list)#, s=100)
    ax[1][0].set_xlabel("Energy (GeV)")#, fontsize=40)
    ax[1][0].set_ylabel(r'$\sigma$ Energy (GeV)')#, fontsize=40)
    ax[1][0].set_xticks(range(math.floor(min(energies)), math.ceil(max(energies))+1))#, fontsize=35)
    #ax[1][0].set_yticks()#fontsize=35)
    #plt.savefig(f"Fits_2/y40cm_sigvsE",facecolor='w')
    #plt.clf()

    ### --- (Sigma Energy/Energy)*100% vs Energy --- ###
    ratio = []
    for i in range(len(std_list)):
        ratio.append((std_list[i]/energies[i])*100)
    ax[1][1].set_title(r'$\sigma$ Energy/True Energy $\times$ 100% vs True Energy')#, fontsize=45)
    ax[1][1].scatter(energies, ratio)#, s=100)
    ax[1][1].set_xlabel("True Energy (GeV)")#, fontsize=40)
    ax[1][1].set_ylabel(r'$\sigma$ Energy/True Energy $\times$ 100%')#, fontsize=40)
    ax[1][1].set_xticks(range(math.floor(min(energies)), math.ceil(max(energies))+1))#, fontsize=35)
    #ax[1][1].set_yticks()#fontsize=35)
    ax[1][1].plot(x_interval_for_fit, fit, label='fit', color="orange")
    plt.savefig(f"Fits_2/full",facecolor='w')
    #plt.clf()

    return ax

    
######################################################################################################################################
# -------------- Main
######################################################################################################################################
def main():
    true_energies = [3,5,7,9]
    locations = [40,40,40,40]
    
    input_data = [] # [ (true_energy0, histogram_data0),
                    #   (true_energy1, histogram_data1),
                    #   (true_energy2, histogram_data2), ... ]
                    ### Add hit location?

    for i in range(len(true_energies)):
        file = uproot4.open(f"../work/calib_gam_y{locations[i]}cm_{true_energies[i]}GeV_10000evt.ana.root")
        input_data.append([true_energies[i], file['ce_emcal_calib;1/reco_energy;1'].to_numpy()])
        
    eff_fit, eff_axes, ind_fit_result = build_eff_plot(input_data)

if __name__ == "__main__":
    main()
