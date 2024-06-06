import numpy as np
from inflation import InflationProblem, InflationSDP

def prob_postquantum(E1, E2, E3):
    p = np.zeros((2, 2, 2, 2, 1, 1, 1, 1))
    for a, b, c in np.ndindex(2, 2, 2):
        ap = 2*a-1
        bp = 2*b-1
        cp = 2*c-1
        p[a, b, c, 0] = (1/8) * (1 
                                 + (ap + bp + cp)*E1 
                                 + (ap*bp + bp*cp + cp*ap)*E2 
                                 + ap*bp*cp*E3)
    return p

# print('Normalization check:', prob_postquantum(0, np.sqrt(2), 0).sum())

def prob_RenouBeigi_coarsegrainned(lambda_0): #notice that lambda_0 must be different from 1/sqrt(2)
    p_R_coarse = np.zeros((3, 3, 3, 3, 1, 1, 1, 1))
    lambda_1 = np.sqrt(1-lambda_0**2)
    u = 0.95 #notice that 2/3<u**2<1
    v = np.sqrt(1-u**2)
    p_R_coarse[0,0,1,0] = p_R_coarse[0,1,0,0] = p_R_coarse[1,0,0,0] = (lambda_1**4) * (lambda_0**2) * (u**2) + (lambda_0**4) * (lambda_1**2) * (v**2)
    p_R_coarse[0,0,2,0] = p_R_coarse[0,2,0,0] = p_R_coarse[2,0,0,0] = (lambda_1**4) * (lambda_0**2) * (v**2) + (lambda_0**4) * (lambda_1**2) * (u**2)

    p_R_coarse[1,1,2,0] = p_R_coarse[1,2,1,0] = p_R_coarse[2,1,1,0] = (lambda_1**3 * u**2 * v - lambda_0**3 * v**2 * u)**2
    p_R_coarse[1,2,2,0] = p_R_coarse[2,2,1,0] = p_R_coarse[2,1,2,0] = (lambda_1**3 * u * v**2 + lambda_0**3 * v * u**2)**2
    p_R_coarse[1,1,1,0] = (lambda_1**3 * u**3 +lambda_0**3 * v**3)**2
    p_R_coarse[2,2,2,0] = (lambda_1**3 * v**3 - lambda_0**3 * u**3)**2

    return p_R_coarse
# print("normalization check:", prob_RenouBeigi_coarsegrainned().sum())

def prob_RenouBeigi(u):
    p_R = np.zeros((4, 4, 4, 4, 1, 1, 1, 1))  # initialization
    v = np.sqrt(1 - u * u)
    u_0 = u
    v_1 = -u
    v_0 = v
    u_1 = v

    p_R[2, 0, 1, 0] = (1 / 8) * u_0 ** 2
    p_R[3, 0, 1, 0] = (1 / 8) * u_1 ** 2
    p_R[2, 1, 0, 0] = (1 / 8) * v_0 ** 2
    p_R[3, 1, 0, 0] = (1 / 8) * v_1 ** 2
    p_R[0, 2, 1, 0] = (1 / 8) * v_0 ** 2
    p_R[0, 3, 1, 0] = (1 / 8) * v_1 ** 2
    p_R[1, 0, 2, 0] = (1 / 8) * v_0 ** 2
    p_R[1, 0, 3, 0] = (1 / 8) * v_1 ** 2
    p_R[0, 1, 2, 0] = (1 / 8) * u_0 ** 2
    p_R[0, 1, 3, 0] = (1 / 8) * u_1 ** 2
    p_R[1, 2, 0, 0] = (1 / 8) * u_0 ** 2
    p_R[1, 3, 0, 0] = (1 / 8) * u_1 ** 2

    vars = [2, 3]
    vars_u = [u_0, u_1]
    vars_v = [v_0, v_1]
    for i, j, k in np.ndindex(2, 2, 2):
        p_R[vars[i], vars[j], vars[k], 0] = (1 / 8) * (
                    vars_u[i] * vars_u[j] * vars_u[k] + vars_v[i] * vars_v[j] *
                    vars_v[k]) ** 2
    return p_R

print("normalization check:", prob_RenouBeigi(0.8).sum())

triangle_Eve = InflationProblem(dag={"rho_AB": ["A", "B", "E"],
                                    "rho_BC": ["B", "C", "E"], 
                                    "rho_AC": ["A", "C", "E"]},
                                    classical_sources=None,
                                    outcomes_per_party=(2, 2, 2, 2),
                                    settings_per_party=(1, 1, 1, 1),
                                    inflation_level_per_source=(2, 2, 3), 
                                    order=["A", "B", "C", "E"])
InfSDP = InflationSDP(triangle_Eve, verbose=2)
InfSDP.generate_relaxation("npa2")
for m in InfSDP.knowable_atoms:
    print("kn", m)

values = {m.name: m.compute_marginal(prob_postquantum(0, np.sqrt(2)-1, 0)) 
          for m in InfSDP.knowable_atoms if 'E' not in m.name}
print(values)

# values = {m.name: m.compute_marginal(prob_RenouBeigi_coarsegrainned(lambda_0=1/np.sqrt(3))) 
#           for m in InfSDP.knowable_atoms if 'E' not in m.name}
# print(values)

# values = {m.name: m.compute_marginal(prob_RenouBeigi(0.8)) 
#           for m in InfSDP.knowable_atoms if 'E' not in m.name}
# print(values)

InfSDP.set_objective({'1': 1, 'pAE(00|00)': 2, 'pE(0|0)': -1, 'pA(0|0)': -1}, direction='max')
InfSDP.set_values(values, use_lpi_constraints=True)
InfSDP.solve(solve_dual=False)
print("objective:", InfSDP.objective_value)
for m in InfSDP.knowable_atoms: 
    print(f"{m.name}->{InfSDP.solution_object['x'][m.name]}")

