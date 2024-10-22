from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# boala (B) -> tuse (T)
#           -> dificultate respiratie (D)
#           -> radiografie anormala (X)
# daca are B => ^tuse si ^radiografie
# D dependenta de prezenta bolii si de tuse
model = BayesianNetwork([('B', 'T'), ('B', 'X'), ('B', 'D'), ('T', 'D')])


cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.9], [0.1]])
cpd_T = TabularCPD(variable='T', variable_card=2,
                          values=[[0.2, 0.7],
                                  [0.8, 0.3]], evidence=['B'], evidence_card=[2])
cpd_X = TabularCPD(variable='X', variable_card=2,
                          values=[[0.9, 0.1], [0.1, 0.9]], evidence=['B'], evidence_card=[2])
cpd_D = TabularCPD(variable='D', variable_card=2,
                          values=[
                              [0.1, 0.5, 0.6, 0.9],  # P(D = 0 | B, T)
                              [0.9, 0.5, 0.4, 0.1]  # P(D = 1 | B, T)
                          ], evidence=['B', 'T'], evidence_card=[2, 2])
model.add_cpds(cpd_B, cpd_T, cpd_X, cpd_D)

inference = VariableElimination(model)
prob_B= inference.query(variables=['B'], evidence={'T': 1, 'D':1})
print(prob_B)

prob_X = inference.query(variables=['X'], evidence={'B' : 0})
print(prob_X)
