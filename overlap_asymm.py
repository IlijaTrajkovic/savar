# Important, remember to install savar with `python setup.py isntall`
from savar.savar import SAVAR
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from savar.functions import create_random_mode, check_stability
from savar.model_generator import SavarGenerator
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import tigramite.independence_tests as itests
from mpl_toolkits.axes_grid1 import make_axes_locatable



import numpy as np

# Some definitions

nx = 90  
ny = 90 # Each component is 30x30
T = 10000 # Time 
comp_size = 20

# Setup spatial weights of underlying processes

N = 4 # Number of components

noise_weights = np.zeros((N, nx, ny))
modes_weights = np.zeros((N, nx, ny))

# This is an important parameter allows us to identify the modes because the covariance noise at grid level.
spatial_covariance = 10 

# There is a function to create random  modes 
#_ = create_random_mode((60, 60), plot=True, random = True)

# If no random X is independent of y
#_ = create_random_mode((60, 60), plot=True, random = False)

# Function to create a circular mode
def create_circular_mode(shape, radius=10):
    mode = np.zeros(shape)
    center = (shape[0] // 2, shape[1] // 2)
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    mask = dist_from_center <= radius
    mode[mask] = np.random.randn(np.sum(mask))
    return mode

# Create noise weights
#noise_weights = np.zeros((N, nx, ny))
modes_weights[0, :comp_size, :comp_size] = create_random_mode((comp_size, comp_size), random=False)
modes_weights[1, :comp_size, 70:] = create_random_mode((comp_size, comp_size), random=False)
modes_weights[2, 70:, :comp_size] = create_random_mode((comp_size, comp_size), random=False)
modes_weights[3, 70:, 70:] = create_random_mode((comp_size, comp_size), random=False)


# noise_weights[0, :comp_size, :comp_size] = create_random_mode((comp_size, comp_size), random=True)
# noise_weights[1, :comp_size, 70:] = create_random_mode((comp_size, comp_size), random=True)
# noise_weights[2, 70:, :comp_size] = create_random_mode((comp_size, comp_size), random=True)
# noise_weights[3, 70:, 70:] = create_random_mode((comp_size, comp_size), random=True)
# We can use the same
#modes_weights = noise_weights


# Plot each mode separately to ensure they are generated correctly
for i in range(N):
    plt.imshow(modes_weights[i, :, :])
    plt.colorbar()
    plt.title(f'Mode {i+1}')
    plt.show()

for i in range(N):
    plt.imshow(noise_weights[i, :, :])
    plt.colorbar()
    plt.title(f'Noise {i+1}')
    plt.show()

# And the causal model
# links_coeffs = {
#     0: [((0, -1), 0.5), ((2, -2), -0.2)],
#     1: [((1, -1), 0.5), ((0, -1), 0.2)],
#     2: [((2, -1), 0.5), ((1, -1), 0.2)]
# }

links_coeffs = {
    0: [((0, -1), 0.2)],
    1: [((1, -4), 0.26), ((2, -2), 0.23)],
    2: [((2, -3), 0.35), ((3, -1), 0.12)],
    3: [((3, -2), 0.31), ((0, -3), 0.52)]
}

comment = 'smalldistancerandomnoisenoseasonalitysimpleconnections'

# One good thing of SAVAR is that if the underlying process is stable and stationary, then SAVAR is also both. 
# Independently of W. This is, we only need to check for stationarity of \PHI and not of W^+\PHI W
check_stability(links_coeffs)

f_1, f_2, f_time_1, f_time_2 = 1, 2, 4000, 8000
time_len = 10000
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
# We could introduce seasonality if 20we would wish
season_dict = {"amplitude": 0.08,
               "period": 12}

name = f"m_{N}_x_{nx}_y_{ny}_cps_{comp_size}_{comment}_tl_{time_len}"
# Specify the path where you want to save the data
save_path = '/home/uzwnx/savar/input/' + name + '.npy'  # Update this path as needed

# Plot the sum of mode weights
sum_modes = modes_weights.sum(axis=0)
fig, ax = plt.subplots()
im = ax.imshow(sum_modes)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_title('Sum of Circular Modes')
fig_path = '/home/uzwnx/savar/input/' + name + '_modes'
plt.savefig(fig_path + '.png')
np.save(fig_path + '.npy', sum_modes)
plt.close()

# Plot the sum of noise weights
sum_noise = noise_weights.sum(axis=0)
fig, ax = plt.subplots()
im = ax.imshow(sum_noise)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
ax.set_title('Sum of Circular Noise')
fig_path = '/home/uzwnx/savar/input/' + name + '_noise'
plt.savefig(fig_path + '.png')
np.save(fig_path + '.npy', sum_noise)
plt.close()

# Creating a dictionary of parameters
parameters = {
    "name": name,
    "nx": nx,
    "ny": ny,
    "T": T,
    "N": N,
    "links_coeffs": links_coeffs,
    "f_1": f_1,
    "f_2": f_2,
    "f_time_1": f_time_1,
    "f_time_2": f_time_2,
    "time_len": time_len,
    "season_dict": season_dict,
    "seasonality" : False
}

# Specify the path to save the parameters
params_path = '/home/uzwnx/savar/input/' + name + '_parameters.npy'
# Save the dictionary of parameters to a .npy file
np.save(params_path, parameters)


# Add the parameters
savar_model = SAVAR(links_coeffs=links_coeffs,
                    time_length=time_len,
                    mode_weights=modes_weights,
                    #season_dict=season_dict,
                    forcing_dict=forcing_dict
                    )
savar_model.generate_data()  # Remember to generate data, otherwise the data field will be empty

# Specify the path where you want to save the data

np.save(save_path, savar_model.data_field)

print("Done!")


#savar_model.data_field.shape  # Here is stored the data field


# You can use the varimax functions that come with SAVAR
# Or use the package varimax^+ [install it `pip install git+https://github.com/xtibau/varimax_plus.git#egg=varimax_plus`]
# from copy import deepcopy
# from savar.dim_methods import get_varimax_loadings_standard as varimax
# modes = varimax(deepcopy(savar_model.data_field.transpose()))  # Use variamx to try to recover the weights
# for i in range(3):
#     plt.imshow(modes['weights'][:, i].reshape(90, 90))
#     plt.colorbar()
#     plt.show()

#     # We can print the modes directly
# singal = savar_model.data_field.transpose() @ modes['weights']
# for i in range(3):
#     plt.plot(singal[:, i])
#     plt.show()


