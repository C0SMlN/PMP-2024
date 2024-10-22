from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import numpy as np

# Sample plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
plt.title("Sine Wave Example")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

plt.show()
model = BayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')])

cpd_O = TabularCPD(variable='O', variable_card=2,
                          values=[[0.1, 0.3], #S=0
                                  [0.9, 0.7]] #S=1
                   , evidence=['S'], evidence_card=[2])

cpd_L = TabularCPD(variable='L', variable_card=2,
                          values=[[0.3, 0.2], #S=0
                                  [0.7, 0.8]] #S=1
                   , evidence=['S'], evidence_card=[2])

cpd_M = TabularCPD(variable='M', variable_card=2,
                          values=[
                              [0.8, 0.4, 0.5, 0.1],  # P(D = 0 | B, T)
                              [0.2, 0.6, 0.5, 0.9]  # P(D = 1 | B, T)
                          ], evidence=['S', 'L'], evidence_card=[2, 2])

cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.6], [0.4]])



model.add_cpds(cpd_S, cpd_M, cpd_L, cpd_O)

inference = VariableElimination(model)

probabilitate = inference.query(variables=['S'], evidence = {'O': 1, 'L': 1, 'M': 1})
print(probabilitate)

prob_spam = probabilitate[1]
