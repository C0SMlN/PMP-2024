import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('date_alegeri_turul2.csv')

# a. Model pentru a prezice varianta dependenta

with pm.Model() as model:
  beta_0 = pm.Normal("beta_0", mu = 0, sigma = 1) #intercept
  beta_1 = pm.Normal("beta_1", mu = 0, sigma = 1) # varsta
  beta_2 = pm.Normal("beta_2", mu = 0, sigma = 1) # sex
  beta_3 = pm.Normal("beta_3", mu = 0, sigma = 1) # educatie
  beta_4 = pm.Normal("beta_4", mu = 0, sigma = 1) # venit
  sigma = pm.HalfNormal("sigma", sigma=1) # deviatia standard

  mu = beta_0 + beta_1 * data['Varsta'] + beta_2 * data['Sex'] + beta_3 * data['Educatie'] + beta_4*data['Venit']

  p = pm.Deterministic("p", pm.math.sigmoid(mu))

  y_obs = pm.Bernoulli("y_obs", p=p, observed=data['Vot'])

  trace = pm.sample(2000)


idata = az.convert_to_inference_data(trace)
az.plot_posterior(idata, var_names=["beta_1", "beta_2", "beta_3", "beta_4"])
plt.show()


print("Coeficienți estimați (medii):")
print("(beta_1):", trace.posterior['beta_1'].mean())
print("(beta_2):", trace.posterior['beta_2'].mean())
print("(beta_3):", trace.posterior['beta_3'].mean())
print("(beta_4):", trace.posterior['beta_4'].mean())

# cele doua variabile care vor influenta cel mai mult vor fi
# cele cu cel mai mare coeficient beta, deci, din graficele obtinute ar fi educatia si varsta

with pm.Model() as model2:
  beta_0x = pm.Normal("beta_0x", mu = 0, sigma = 1) #intercept
  beta_1x = pm.Normal("beta_1x", mu = 0, sigma = 1) # varsta
  beta_3x = pm.Normal("beta_3x", mu = 0, sigma = 1) # educatie
  sigmaz = pm.HalfNormal("sigma", sigma=1) # deviatia standard

  mu_ = beta_0x + beta_1x * data['Varsta']+ beta_3x * data['Educatie']

  p2 = pm.Deterministic("p2", pm.math.sigmoid(mu_))

  y_obs2 = pm.Bernoulli("y_obs2", p=p, observed=data['Vot'])

  trace2 = pm.sample(2000)


decision_boundary = -(trace2.posterior['beta_1x'].mean() / trace2.posterior['beta_3x'].mean())
print(f"Granița de decizie (medie): {decision_boundary:.2f}")




