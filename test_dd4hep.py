# %%
import uproot4
from hist import Hist
import hist
import awkward1 as ak
import numpy as np
import matplotlib.pyplot as plt
import sys
from efficiency_fit import build_eff_plot

# Pretty printing arrays
from pprint import pprint


# %%
input_data = [
(1.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_1.0GeV.root"),
(1.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_1.5GeV.root"),
(2.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_2.0GeV.root"),
(2.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_2.5GeV.root"),
(3.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_3.0GeV.root"),
(3.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_3.5GeV.root"),
(4.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_4.0GeV.root"),
(4.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_4.5GeV.root"),
(5.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_5.0GeV.root"),
(5.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_5.5GeV.root"),
(6.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_6.0GeV.root"),
(6.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_6.5GeV.root"),
(7.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_7.0GeV.root"),
(7.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_7.5GeV.root"),
(8.0, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_8.0GeV.root"),
(8.5, "test/dd4pod_example_data/test_1000evt_gamma_1x_31y_0z_8.5GeV.root"),
]


# %%
fit_input_data = []
true_energies = []
sigmas = []
sigma_div_e_vals = []
for true_energy, file_name in input_data:
    events_tree = uproot4.open(file_name)["events/EcalEndcapNHits/"]    
    edeps = events_tree['EcalEndcapNHits.energyDeposit'].array()
    energy_per_event = ak.sum(edeps, axis=-1)
    sigma = np.std(energy_per_event)
    mean = np.mean(energy_per_event)
    rms = np.sqrt(np.mean(energy_per_event**2))
    
    print(f"Energy = {true_energy}, mean = {mean}, RMS = {rms}, Sigma = {sigma} sigma/Energy = {sigma/rms}")
    sigma_div_e_vals.append(sigma/true_energy)
    sigmas.append(sigma)
    true_energies.append(true_energy)
    # noinspection PyTypeChecker
    h1_energies = Hist(hist.axis.Regular(100, true_energy - true_energy/5, true_energy + true_energy/5, name="E GeV"))
    h1_energies.fill(energy_per_event)
    #h1_energies.plot()
    #plt.show()

    fit_input_data.append([true_energy, h1_energies.to_numpy()])


# %%
plt.plot(true_energies, sigma_div_e_vals, "o--")
plt.show()


# %%
eff_fit, eff_axes, ind_fit_result = build_eff_plot(fit_input_data)


# %%



