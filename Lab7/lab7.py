import pymc as pm
import numpy as np

# observatiile
data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])
x = np.mean(data)

# modelul bayesian
with pm.Model() as noise_model:
    # Priori pentru media nivelului de zgomot => mu ~ N(x(=58), 10)
    mu = pm.Normal("mu", mu=x, sigma=10)

    # Priori pt deviatia standard, sigma ~ HalfNormal(10)
    sigma = pm.HalfNormal("sigma", sigma=10)

    # Likelihood ul datelor
    noise = pm.Normal("noise", mu=mu, sigma=sigma, observed=data)

    with noise_model:
        # sampling ptr a obtine distributiile posterioare pentru mu și sigma
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

        # obtine intervalele HDI de 95% pentru miu și sigma
        mu_hdi = pm.stats.hdi(trace, hdi_prob=0.95)["mu"]
        sigma_hdi = pm.stats.hdi(trace, hdi_prob=0.95)["sigma"]

    print("Intervalul HDI 95% pentru mu:", mu_hdi)
    print("Intervalul HDI 95% pentru sigma:", sigma_hdi)

