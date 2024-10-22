import pgmpy
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation


mrf = MarkovNetwork()

noduri = ['A1', 'A2', 'A3', 'A4', 'A5']
mrf.add_nodes_from(noduri)

edges = [('A1', 'A2'), ('A1', 'A3'),
         ('A2', 'A4'), ('A2', 'A5'),
         ('A3', 'A4'), ('A4', 'A5')]
mrf.add_edges_from(edges)

pos = nx.circular_layout(mrf)

nx.draw(mrf, with_labels=True, pos=pos, alpha=0.5, node_size=2000)
plt.title('MRF')
plt.show()

cliques = list(nx.find_cliques(mrf))

for clique in cliques:
    print("Clique:", clique)


factor_a1_a2 = DiscreteFactor(variables=['A1', 'A2'], cardinality=[2, 2], values=[10, 8.3, 3.7, 2])
factor_a1_a3 = DiscreteFactor(variables=['A1', 'A3'], cardinality=[2, 2], values=[23.7, 21, 3.7, 2])
factor_a3_a4 = DiscreteFactor(variables=['A3', 'A4'], cardinality=[2, 2], values=[10, 1, 100, 90])
factor_a4_a5 = DiscreteFactor(variables=['A4', 'A5'], cardinality=[2, 2], values=[50, 30, 20, 10])

#factor_a5_a4_a2 = DiscreteFactor(variables=['A5', 'A4', 'A2'], cardinality=[2, 2, 2], values=[80, 60, 20, 10])

mrf.add_factors(factor_a1_a2, factor_a1_a3,factor_a3_a4, factor_a4_a5)
mrf.get_factors()
mrf.get_local_independencies()
bp_infer = BeliefPropagation(mrf)
marginals = bp_infer.map_query(variables=['A1','A2','A3','A4','A5'])
print(marginals)