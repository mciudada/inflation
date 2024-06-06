import numpy as np
from inflation import InflationProblem, InflationLP
from distro_repository import prob_postquantum_with_Eve


triangle_Eve = InflationProblem(dag={"rho_AB": ["A", "B", "E"],
                                    "rho_BC": ["B", "C", "E"], 
                                    "rho_AC": ["A", "C", "E"]},
                                    classical_sources=None,
                                    outcomes_per_party=(2, 2, 2, 2),
                                    settings_per_party=(1, 1, 1, 1),
                                    inflation_level_per_source=(2, 2, 3), 
                                    order=["A", "B", "C", "E"])
InfLP = InflationLP(triangle_Eve, include_all_outcomes=False, verbose=2)

values = {m.name: m.compute_marginal(prob_postquantum_with_Eve(0, np.sqrt(2)-1, 0))
          for m in InfLP.knowable_atoms if 'E' not in m.name}

InfLP.set_values(values, use_lpi_constraints=True)
for k, v in InfLP.known_moments.items():
    if k.is_atomic:
        print(f"{k}={v}")
InfLP.set_objective({'1': 1, 'pAE(00|00)': 2, 'pE(0|0)': -1, 'pA(0|0)': -1}, direction='max')

InfLP.solve(solve_dual=False)
print(InfLP.objective_value)
for m in InfLP.knowable_atoms: 
    print(f"{m.name}->{InfLP.solution_object['x'][m.name]}")

