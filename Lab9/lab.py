import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

# Define the parameter space
Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

# Set up the figure for plotting
fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(14, 10))
fig.suptitle("Posterior Distribution of $n$ for Different $Y$ and $\\theta$ Combinations", fontsize=16)

# Iterate over each combination of Y and theta
for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        # Bayesian Model
        with pm.Model() as model:
            # Prior for n (Poisson)
            n = pm.Poisson('n', mu=10)
            # Likelihood (Binomial)
            Y_obs = pm.Binomial('Y_obs', n=n, p=theta, observed=Y)

            # Use Metropolis for discrete sampling
            step = pm.Metropolis()
            # Inference with reduced sampling using Metropolis
            trace = pm.sample(500, step=step, return_inferencedata=True, cores=1, tune=250, chains=2)

            # Plot the posterior for n
            az.plot_posterior(trace, var_names=['n'], ax=axes[i, j])
            axes[i, j].set_title(f'Y={Y}, θ={theta}')

# Adjust layout for clarity
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
