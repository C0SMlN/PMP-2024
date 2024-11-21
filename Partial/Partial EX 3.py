import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymc as pm

n = 10 # numarul de aruncari ale monedei
observations = ['s','s','b','b','s','s','b','s','b','s']

# distributia a priori
distr_beta = pm.Beta('distr_beta', alpha=1, beta=1)

# verosimilitatea
likelihood = pm.Beta('observations', alpha=1, beta=1, observed=observations)

# distr a posteriori nenormalizata
unnorm_posteriori =likelihood * distr_beta
posterior_pdf = unnorm_posteriori / np.trapz(unnorm_posteriori, observations)

# media a posteriori
mean_posteriori = np.trapz(observations * posterior_pdf, observations)


plt.figure(figsize=(10, 6))
plt.plot(observations, posterior_pdf, label='Distribuția a posteriori')
plt.axvline(mean_posteriori, color='r', linestyle='--', label=f'Media = {mean_posteriori:.2f}')
plt.title('Distribuția a posteriori a cu prior normal trunchiat')
plt.xlabel('numarul de aruncari')
plt.ylabel('aruncarile')
plt.legend()
plt.grid(True)
plt.show()


# ======== 2 ===========
observations = ['b','b','s','b','s','s','b','s','s','b']

# distributia a priori
distr_beta = pm.Beta('distr_beta', alpha=1, beta=1)

# verosimilitatea
likelihood = pm.Beta('observations', alpha=1, beta=1, observed=observations)

# distr a posteriori nenormalizata
unnorm_posteriori =likelihood * distr_beta
posterior_pdf = unnorm_posteriori / np.trapz(unnorm_posteriori, observations)

# media a posteriori
mean_posteriori = np.trapz(observations * posterior_pdf, observations)

