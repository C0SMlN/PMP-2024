from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random



def coin_toss():
    return random.choice([0,1])

def dice_roll():
    return random.randint(1,6)

def biased_coin_toss():
    return random.choices([0, 1], weights=[4, 3], k=1)[0]

def simulare():
    j0_wins = 0
    j1_wins = 0
    for _ in range(10000):
        # se decide cine incepe
        first_player = coin_toss()  # 0 = J0, 1 = J1

        if first_player == 0:
            # j0 începe
            n = dice_roll()
            m = sum(biased_coin_toss() for _ in range(2 * n))  # j1 arunca moneda masluita de 2n ori
        else:
            # j1 începe
            n = dice_roll()
            m = sum(coin_toss() for _ in range(2 * n))  # j0 arunca zarul



        if n >= m and first_player==0:
            j0_wins+=1
        elif n<m and first_player==0:
            j1_wins+=1
        elif n>=m and first_player==1:
            j1_wins+=1
        elif n<m and first_player==1:
            j0_wins+=1

    return j0_wins, j1_wins

j0_wins, j1_wins = simulare()

print(f"j0 a castigat de {j0_wins} de ori.")
print(f"j1 a castigat de {j1_wins} de ori.")


# subpunctul 2
# variabile: S (start), T (coin toss-ul initial), W (castigatorul),
# n (rezultatul zarului), m (2 * n)
model = BayesianNetwork([('S', 'n'),
                         ('n', 'm'),
                         ('n', 'W'),
                         ('m', 'W')])
cpd_S = TabularCPD(variable='S', variable_card=2, values=[[0.5], [0.5]]) #aruncarea monedei
cpd_n = TabularCPD(variable='n', variable_card=6, values=[[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]]) # dat cu zarul
cpd_M = TabularCPD(variable='m', variable_card=13,
                   values=[[1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2],  # moneda normala
                           [4/7,4/7,4/7,4/7,4/7,4/7,4/7,4/7,4/7,4/7,4/7,4/7,4/7]], # moneda masliuta
                   evidence=['S'], evidence_card=[2])
#cpd_C = TabularCPD(variable='C', variable_card=2,
#                  values=[[coloanele mari sunt n, coloanele mari sunt m si randurile
#                           specifica cine castiga]]

model.add_cpds(cpd_S, cpd_M)
inference = VariableElimination(model)

prob_J0 = inference.query(variables=['S'], evidence={'M': 1}).values[0]
prob_J1 = inference.query(variables=['S'], evidence={'M': 1}).values[1]

print(f"prob ca j0 sa inceapa jocul: {prob_J0}")
print(f"prob ca j0 sa inceapa jocul: {prob_J1}")
