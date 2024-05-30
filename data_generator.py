# Important, remember to install savar with `python setup.py isntall`
from savar.savar import SAVAR
import matplotlib.pyplot as plt
from savar.functions import create_random_mode, check_stability
from savar.model_generator import SavarGenerator
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.independence_tests as itests
print(dir(itests))


import numpy as np

# Some definitions

nx = 30  
ny = 90 # Each component is 30x30
T = 500 # Time 

# Setup spatial weights of underlying processes

N = 4 # Three components

noise_weights = np.zeros((N, nx, ny))
modes_weights = np.zeros((N, nx, ny))

# This is an important parameter allows us to identify the modes because the covariance noise at grid level.
spatial_covariance = 10 

# There is a function to create random  modes 
_ = create_random_mode((30, 30), plot=True, random = True)

# If no random X is independent of y
_ = create_random_mode((30, 30), plot=True, random = False)

# We can create the modes with it.
noise_weights = np.zeros((N, nx, ny))
noise_weights[0, :, :30] = create_random_mode((30, 30), random = False)  # Random = False make modes round.
noise_weights[1, :, 30:60] = create_random_mode((30, 30), random = False)
noise_weights[2, :, 60:90] = create_random_mode((30, 30), random = False)
noise_weights[3, :, 50:80] = create_random_mode((30, 30), random = False)

# How the modes look like
plt.imshow(noise_weights.sum(axis=0))
plt.colorbar()

# We can use the same
modes_weights = noise_weights

# And the causal model
links_coeffs = {
    0: [((0, -1), 0.5), ((2, -2), -0.2)],
    1: [((1, -1), 0.5), ((0, -1), 0.2)],
    2: [((2, -1), 0.5), ((1, -1), 0.2)],
    3: [((-1, -1), 0.5), ((2, 0), 0.2)]
}

# One good thing of SAVAR is that if the underlying process is stable and stationary, then SAVAR is also both. 
# Independently of W. This is, we only need to check for stationarity of \PHI and not of W^+\PHI W
check_stability(links_coeffs)

f_1, f_2, f_time_1, f_time_2 = 1, 2, 100, 150
time_len = 500
w_f = modes_weights
# A very simple method for adding a focring term (bias on the mean of the noise term)
forcing_dict = {
    "w_f": w_f,  # Shape of the mode of the forcing
    "f_1": f_1,  # Value of the forcing at period_1
    "f_2": f_2,  # Value of the forcing at period_2
    "f_time_1": f_time_1,  # The period one goes from t=0  to t=f_time_1
    "f_time_2": f_time_2,  # The period two goes from t= f_time_2 to the end. Between the two periods, the forcing is risen linearly
    "time_len": time_len,
}
# We could introduce seasonality if we would wish
season_dict = {"amplitude": 0.08,
               "period": 12}
# Add the parameters
savar_model = SAVAR(links_coeffs=links_coeffs,
                    time_length=time_len,
                    mode_weights=modes_weights,
                    season_dict=season_dict,
                    forcing_dict=forcing_dict
                    )
savar_model.generate_data()  # Remember to generate data, otherwise the data field will be empty

savar_model.data_field.shape  # Here is stored the data field
data = savar_model.data_field

# Apply PCMCI
# Convert data to Tigramite's DataFrame
dataframe = DataFrame(data)

# Set up the PCMCI object
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())

# Apply PCMCI to discover causal relationships
results = pcmci.run_pcmci(tau_max=3, pc_alpha=0.05)

# Extract the causal graph
causal_graph = results['graph']

# Step 3: Evaluate Performance
def evaluate_performance(true_graph, predicted_graph):
    true_flat = true_graph.flatten()
    predicted_flat = predicted_graph.flatten()
    accuracy = accuracy_score(true_flat, predicted_flat)
    precision = precision_score(true_flat, predicted_flat, average='binary')
    recall = recall_score(true_flat, predicted_flat, average='binary')
    return accuracy, precision, recall

# Convert 'links_coeffs' to a matrix form for comparison
true_causal_graph = np.zeros((len(links_coeffs), len(links_coeffs)))
for var, links in links_coeffs.items():
    for (linked_var, lag), coeff in links:
        if lag == -1:
            true_causal_graph[linked_var, var] = 1

# Evaluate the performance of PCMCI
accuracy, precision, recall = evaluate_performance(true_causal_graph, causal_graph)

print(f"Causal Graph Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Visualization
for i in range(data.shape[0]):
    plt.plot(data[i, :], label=f'Variable {i}')
plt.legend()
plt.show()

# Plot the learned causal graph
plt.imshow(causal_graph, cmap='hot', interpolation='nearest')
plt.title('Learned Causal Graph')
plt.show()

# You can use the varimax functions that come with SAVAR
# Or use the package varimax^+ [install it `pip install git+https://github.com/xtibau/varimax_plus.git#egg=varimax_plus`]
from copy import deepcopy
from savar.dim_methods import get_varimax_loadings_standard as varimax
modes = varimax(deepcopy(savar_model.data_field.transpose()))  # Use variamx to try to recover the weights
for i in range(8):
    plt.imshow(modes['weights'][:, i].reshape(30, 90))
    plt.colorbar()
    plt.show()

    # We can print the modes directly
singal = savar_model.data_field.transpose() @ modes['weights']
for i in range(8):
    plt.plot(singal[:, i])
    plt.show()

"""savar_generator = SavarGenerator(n_variables=10,
                      n_cross_links=5,
                      time_length=800)
# You need to generate the model
savar_model = savar_generator.generate_savar()
# You need to generate the data
savar_model.generate_data()"""