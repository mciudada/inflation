import mosek
from inflation import InflationProblem, InflationLP
from inflation.utils import clean_coefficients
import numpy as np

p_Q = np.zeros((2, 2, 2, 1, 1, 1), dtype=float)
q = [1 / 2, 1 / 2]
p_Q_do = np.zeros((2, 2, 2), dtype=float)
# v = 1 / np.sqrt(2)
# v = 0.72
v = 1

for a in range(2):
    for b in range(2):
        for c in range(2):
            p_Q_do[a, b, c] = q[c] * (1 / 2)
            p_Q[a, b, c] = (q[c]) * (1 / 4) * (
                (1 + (v) * ((-1) ** (a + b + b * c))))

Evans = InflationProblem({"U_AB": ["A", "B"],
                          "U_BC": ["B", "C"],
                          "B": ["A", "C"]},
                         outcomes_per_party=(2, 2, 2),
                         settings_per_party=(1, 1, 1),
                         inflation_level_per_source=(2, 2),  # TO BE MODIFIED
                         order=("A", "B", "C"))

Evans_Unpacked = InflationLP(Evans,
                             nonfanout=False,
                             use_only_equalities=False,
                             verbose=2)
semiknown_usage = False
Evans_Unpacked.set_distribution(p_Q, use_lpi_constraints=semiknown_usage)
Evans_Unpacked.update_values({
    '<A_1_0_0_0>': 1 / 2,
    '<A_1_0_1_0>': 1 / 2},
    use_lpi_constraints=semiknown_usage)

# print(f"Known Values: {Evans_Unpacked.known_moments}")
# params = {
#     mosek.iparam.presolve_use: mosek.presolvemode.off,  # REVERT!!
#     mosek.iparam.optimizer: mosek.optimizertype.dual_simplex}  # For precise inequalities
Evans_Unpacked.solve(dualise=False, verbose=2, relax_known_vars=True)
print(Evans_Unpacked.certificate_as_probs())
Evans_Unpacked.solve(dualise=True, verbose=2, relax_known_vars=True)
print(Evans_Unpacked.certificate_as_probs())
