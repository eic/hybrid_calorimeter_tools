'''
    examples.py

    Data examples for 

    Authors:    Nathan Branson, Dmitry Romanov
    Updated:    10/29/2021
'''


import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import uproot4
from hist import Hist
import hist
import sys
sys.path.append("../")
from efficiency_fit import *

######################################################################################################################################
# -------------- Main
######################################################################################################################################
def main():

    ######################
    ### G4E File Types ###
    ######################
    '''
    true_energies = [3,5,7,9]
    locations = [40,40,40,40]
    
    input_data = [] # [ (true_energy0, histogram_data0),
                    #   (true_energy1, histogram_data1),
                    #   (true_energy2, histogram_data2), ... ]
                    ### Add hit location?

    for i in range(len(true_energies)):
        file = uproot4.open(f"g4e_example_data/calib_gam_y{locations[i]}cm_{true_energies[i]}GeV_10000evt.ana.root")
        input_data.append([true_energies[i], file['ce_emcal_calib;1/reco_energy;1'].to_numpy()])
    '''


    #########################
    ### DD4HEP File Types ###
    #########################
    files = [
        [1.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_1.0GeV.root"],
        [1.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_1.5GeV.root"],
        [2.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_2.0GeV.root"],
        [2.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_2.5GeV.root"],
        [3.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_3.0GeV.root"],
        [3.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_3.5GeV.root"],
        [4.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_4.0GeV.root"],
        [4.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_4.5GeV.root"]
    ]

    input_data = []
    true_energies = []
    sigmas = []
    sigma_div_e_vals = []
    for true_energy, file_name in files:
        events_tree = uproot4.open(file_name)["events/EcalEndcapNHits/"]    
        edeps = events_tree['EcalEndcapNHits.energyDeposit'].array()
        energy_per_event = ak.sum(edeps, axis=-1)
        sigma = np.std(energy_per_event)
        mean = np.mean(energy_per_event)
        rms = np.sqrt(np.mean(energy_per_event**2))
        
        #print(f"Energy = {true_energy}, mean = {mean}, RMS = {rms}, Sigma = {sigma} sigma/Energy = {sigma/rms}")
        
        sigma_div_e_vals.append(sigma/true_energy)
        sigmas.append(sigma)
        true_energies.append(true_energy)
        # noinspection PyTypeChecker
        h1_energies = Hist(hist.axis.Regular(100, true_energy - true_energy/5, true_energy + true_energy/5, name="E GeV"))
        h1_energies.fill(energy_per_event)
        #h1_energies.plot()
        #plt.show()

        input_data.append([true_energy, h1_energies.to_numpy()])




    #######################################
    ### Can be used for both file types ###
    #######################################
    eff_fit, eff_axes, ind_fit_result = build_eff_plot(input_data)
    fig = plt.plot(figsize=(20, 15))
    plt.add_axes(eff_axes)
    plt.savefig(f"Fits/full",facecolor='w')

if __name__ == "__main__":
    main()