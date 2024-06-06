import numpy as np
from inflation import InflationProblem, InflationLP
from distro_repository import prob_postquantum

PR_box_in_triangle = prob_postquantum(0, np.sqrt(2)-1, 0)

triangle_classical = InflationProblem(dag={"rho_AB": ["A", "B"],
                                    "rho_BC": ["B", "C"],
                                    "rho_AC": ["A", "C"]},
                                    classical_sources="all",
                                    outcomes_per_party=(2, 2, 2),
                                    settings_per_party=(1, 1, 1),
                                    inflation_level_per_source=(2, 2, 3),
                                    order=["A", "B", "C"])
InfLP = InflationLP(triangle_classical, include_all_outcomes=False, verbose=2)
for m in InfLP.knowable_atoms:
    if m.n_operators <= 2:
        print("Known: ", m)
values = {m.name: m.compute_marginal(PR_box_in_triangle)
          for m in InfLP.knowable_atoms if (m.n_operators <= 2)}
InfLP.update_values(values, use_lpi_constraints=True)
for k, v in InfLP.known_moments.items():
    print(f"{k}={v}")
InfLP.solve(solve_dual=False)
print(InfLP.objective_value)
