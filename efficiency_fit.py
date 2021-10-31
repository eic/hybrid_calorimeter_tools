'''
    efficiency_fit.py

    Functions for hybrid calorimeter analysis on the EIC.

    CURRENTLY REQUIRES NUMPY ARRAYS

    Authors:    Nathan Branson, Dmitry Romanov
    Updated:    10/29/2021
'''

# TODO "... = check_calibration(input_data??)" and "axes = plot_fit_results([(ind_fit_result, ind_plot_axes)], ax=ax)" need to be done 
# TODO                if that is still in plan.


#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import uproot4
from hist import Hist
import hist


from scipy.stats import crystalball ### UNUSED
from scipy.special import erf ### UNUSED

import awkward1 as ak ### UNUSED
from zfit.models.physics import crystalball_func ### UNUSED
from uncertainties import unumpy as unp ### UNUSED
import math ### UNUSED


# Formatting plots
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
# fit_crystal_ball
# Fits the energy plots to crystal ball function for energy reconstruction
# returns:  [x_interval_for_fit, reco_fit]  - fitted line from crystal ball function
#           ax                              - axes of individual reconstruction fit plots 
#           mean                            - mean result (reconstructed energy) from fitted line
#           std                             - standard dev of fitted line
######################################################################################################################################
def fit_crystal_ball(true_energy, histogram):
    '''array = []
    print(len(histogram))
    for i in range(len(histogram[1])-1):
        diff = histogram[1][i+1]-histogram[1][i]
        array.append(diff)'''
    ### get bin centers
    bin_centers = histogram[1][:-1] + np.diff(histogram[1]) / 2
    #bin_centers = histogram[1][:-1] + array
    for i in bin_centers:
        i = i / 2

    energy = np.argmax(histogram[0])
    e_reco = histogram[1][energy]
    e_min = e_reco-3
    e_max = e_reco+1


    arg_min = find_nearest(histogram[1], e_min)#; arg_min = arg_min[0][0]
    arg_max = find_nearest(histogram[1], e_max)#; arg_max = arg_max[0][0]

    fig, ax = plt.subplots()
    ### plot histogram
    width = .85*(histogram[1][1] - histogram[1][0])
    plt.bar(histogram[1][0:len(histogram[0]-1)], histogram[0], align='edge', width=width)

    ### plot fit
    beta, m, scale = true_energy, 3, .4
    x_interval_for_fit = np.linspace(histogram[1][0], histogram[1][len(histogram[0]-1)], 10000)
    popt, _ = curve_fit(crystalball, bin_centers, histogram[0], p0=[1, 1, histogram[1][energy], .05, max(histogram[0])])
    reco_fit = crystalball(x_interval_for_fit, *popt) # *popt = alpha, n, mean, sigma, amp
    ax.plot(x_interval_for_fit, reco_fit, label='fit', color="orange")
    mean = popt[2]
    std = popt[3]
    
    ### --- print mean and std on graph
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = '\n'.join((r'$\mu=%.4f$' % (round(mean,4), ), r'$\sigma=%.4f$' % (round(std, 4), )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    ax.legend()
    ax.set_xlabel("Energy")
    ax.set_ylabel("Events")

    return [x_interval_for_fit, reco_fit], ax, mean, std
    ### [x axis, crystal ball fit], mean, standard deviation 


######################################################################################################################################
# build_eff_plot
# builds the efficiency plots
# REQUIRES NUMPY ARRAY
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

    ### Gets percentages of energy ratios
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
    ax[0][0].set_yticks(range(np.floor(min(mean_list)), np.ceil(max(mean_list))+1))#, fontsize=35)
    ax[0][0].set_xticks(range(np.floor(min(energies)), np.ceil(max(energies))+1))#, fontsize=35)
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
    #plt.clf()

    #plt.savefig(f"Fits/full",facecolor='w')

    return ax
