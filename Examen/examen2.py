import pymc as pm
import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('iris.csv')


# A) model mixt in pymc
with pm.Model() as model:
    weights = pm.Dirichlet('weights', a=np.ones(3)) # prior pt weight urile varibailelor
    means = pm.Normal('means', mu=0, sigma=10, shape=(3, 4)) # prior pt mediile componentelor
    stds = pm.HalfNormal('stds', sigma=10, shape=(3, 4)) # dev standard
    cluster = pm.Categorical('cluster', p=weights, shape=len(data)) # apartenenta la cluster
    y = pm.NormalMixture('y', w=weights, mu=means, sigma=stds, observed=data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values)
    idata_mg = pm.sample(random_seed=123, return_inferencedata=True)

az.plot_trace(idata_mg, var_names=['weights', 'means', 'stds'])
plt.show()

# B) ipmartim in clustere, apoi gasim caracteristica cu cea mai mare valoare de separare
separations = {}
for feature in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
    separation = data.groupby('cluster')[feature].mean().std()
    separations[feature] = separation

best_feature = max(separations, key=separations.get)
print(f"The best feature to separate the clusters is: {best_feature}")





