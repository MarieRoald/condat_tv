
# Example of denoising each row in a matrix with total variation minimization

import condat_tv 
import numpy as np
import matplotlib.pyplot as plt


### Generate syntetic data

np.random.seed(0)

nrow = 10
ncol = 100

# Generate a sparse "derivative" vectors for the rows
signal_derivative = np.random.standard_normal((nrow, ncol))*4
for k in range(nrow):
    signal_derivative[k] *= (np.random.uniform(0, 1, size=(ncol)) > 0.95)
    
# Integrate the sparse derivative vectors to obtain piecewise constant vectors
signal = np.cumsum(signal_derivative, axis=1)

# Add noise
noisy_signal = signal + np.random.standard_normal(signal.shape)


### Denoise each row with total variation minimization

denoised_signal = condat_tv.tv_denoise_matrix(noisy_signal, regularisation_strength=2)

### Visualize denoising result

fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True, sharey=True)

ax[0].plot(signal.T)
ax[0].set_xlabel("Noise free signal")
ax[0].set_xlim(0, ncol)

ax[1].plot(noisy_signal.T)
ax[1].set_xlabel("Corrupted signal")

ax[2].plot(denoised_signal.T)
ax[2].set_xlabel("Denoised signal")

fig.suptitle("Visualisation of signal matrix (one line per row)")
plt.show()