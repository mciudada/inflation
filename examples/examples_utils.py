from copy import deepcopy
import itertools
import numpy as np
from qutip import tensor, basis, expect, qeye, ket2dm

def pfunc2array(func, outcomes=[2, 2, 2], settings=[1, 1, 1]):
    p = np.zeros([*outcomes, *settings])
    for a,b,c,x,y,z in itertools.product(*[range(i) for i in [*outcomes, *settings]]):
        p[a,b,c,x,y,z] = func(a,b,c,x,y,z)
    return p

def P_2PR(a, b, c, x, y, z):
    return ( 1 + (-1) ** (a + b + c + x*y + y*z) ) / 8

P_2PR_array = pfunc2array(P_2PR, [2, 2, 2], [2, 2, 2])

def P_W(a, b, c, x, y, z):
    if a + b + c == 1:
        return 1 / 3
    else:
        return 0

P_W_array = pfunc2array(P_W, [2, 2, 2], [1, 1, 1])


def P_PRbox():
    P_PRbox_array = np.zeros((2,2,2,2))
    for x, y, a, b in itertools.product(range(2), repeat=4):
        if (x, y) == (1, 1):
            if a != b:
                P_PRbox_array[a, b, x, y] = 1/2
        else:
            if a == b:
                P_PRbox_array[a, b, x, y] = 1/2
    return P_PRbox_array

P_PRbox_array = P_PRbox()


def EveGuessingProbability(noise=1):
    prob = np.zeros((2, 4, 2, 2, 2))
    for a, b0, b1, c, x, z in itertools.product(range(2), range(2), range(2),
                                                range(2), range(2), range(2)):
        prob[a,2*b1+b0,c,x,z] = (1 + noise**2*(-1)**(a + c)*(((-1)**b0 + (-1)**(b1 + x + z))/2))/2**4
    return prob

def bisection(InfSDP, probarray, tol_vis=1e-4, verbose=0, max_iter=20, use_lpi_constraints=False):
    v0, v1 = 0, 1
    vm = (v0 + v1)/2
    
    InfSDP.verbose = 0
    iteration = 0
    last_good = deepcopy(InfSDP)

    outputdims = probarray.shape[:int(len(probarray.shape)/2)]
    nroutputs = np.prod(outputdims)

    while abs(v1 - v0) >= tol_vis and iteration < max_iter:
        pnoisy = vm * probarray + (1-vm) * np.ones(outputdims) / nroutputs#distribution(visibility=vm)
        InfSDP.set_distribution(pnoisy, use_lpi_constraints=use_lpi_constraints)
        InfSDP.solve(feas_as_optim=True)
        if verbose:
            print(iteration, "Maximum smallest eigenvalue:", "{:10.4g}".format(
                InfSDP.primal_objective), "\tvisibility =", "{:.4g}".format(vm))
        iteration += 1
        if InfSDP.objective_value >= 0:
            v0 = vm
            vm = (v0 + v1)/2
        elif InfSDP.objective_value < 0:
            v1 = vm
            vm = (v0 + v1)/2
            last_good = deepcopy(InfSDP)
        # if abs(InfSDP.objective_value) <= 1e-7:
        #     break
    
    return last_good, vm


def get_W_state(N):
    # Taken from https://github.com/FlavioBaccari/Hierarchy-for-nonlocality-detection
    """Generates the density matrix for the N-partite W state.

    :param N: number of parties.
    :type N: int

    :returns: the W density matrix as a qutip.qobj.Qobj
    """
    state = tensor([basis(2, 1)] + [basis(2, 0) for _ in range(N - 1)])
    for i in range(1, N):
        components = [basis(2, 0) for _ in range(N)]
        components[i] = basis(2, 1)
        state += tensor(components)
    return 1. / N**0.5 * state


def get_W_reduced(N):
    # Taken from https://github.com/FlavioBaccari/Hierarchy-for-nonlocality-detection
    """Generates the reduced four-body state for the N-partite W state. Since
    the W state is symmetric, it is independent of the choice of the four
    parties that one considers.

    :param N: number of parties for the global state.
    :type N: int

    :returns: the reduced state as a qutip.qobj.Qobj
    """
    w = ket2dm(get_W_state(4))
    rest = ket2dm(tensor([basis(2, 0) for _ in range(4)]))

    return 4. / N * w + (N - 4.) / N * rest

def get_GHZ_reduced(N):
    # Taken from https://github.com/FlavioBaccari/Hierarchy-for-nonlocality-detection
    """Generates the reduced four-body state for the N-partite GHZ state. Since
    the GHZ state is symmetric, it is independent of the choice of the four
    parties that one considers.

    :param N: number of parties for the global state,
    :type N: int

    :returns: the reduced state as a qutip.qobj.Qobj
    """
    zero = tensor([basis(2, 0) for _ in range(N)])
    one = tensor([basis(2, 1) for _ in range(N)])
    return 1 / 2 * (ket2dm(zero) + ket2dm(one))

def noisy_known_values_W_state(vis, W_state, W_operators, meas, outcomes_per_party, settings_per_party):
    from qutip import expect, tensor, qeye
    import itertools
    import numpy as np

    N = 7
    noise = tensor([qeye(2) for _ in range(4)]) / 16

    known_values = {}
    for if_party_involved in itertools.product(*[range(2) for _ in range(N)]):
        nr_parties_involved = sum(if_party_involved)
        if nr_parties_involved == 0:
            known_values[1] = 1
        elif nr_parties_involved <= 4:
            parties_involved = [p for p, b in enumerate(if_party_involved) if b == 1]
            settings_of_parties_involved = [settings_per_party[p] for p in parties_involved]
            outcomes_of_parties_involved = [outcomes_per_party[p]-1 for p in parties_involved]  # -1 because of CG notation
            for settings in itertools.product(*[range(x) for x in settings_of_parties_involved]):
                for outcomes in itertools.product(*[range(a) for a in outcomes_of_parties_involved]):
                    projectors = []
                    sdpvar = 1
                    for i in range(nr_parties_involved):
                        p, x, a = parties_involved[i], settings[i], outcomes[i]
                        projectors.append(W_operators[p][x][a])
                        sdpvar *= meas[p][0][x][a]
                    # The W state is independent of the choice of the four parties that one
                    # considers. We use this to simplify the calculation of the reduced moments.
                    for i in range(4-nr_parties_involved):
                        projectors.append(qeye(2))
                    known_values[sdpvar] = expect(tensor(projectors),
                                                  vis * W_state + (1-vis) * noise)
    return known_values

def bisect_W(InfSDP, state, state_measurements, tol_vis=1e-4, verbose=0, max_iter=20, use_lpi_constraints=False):
    v0, v1 = 0, 1
    vm = (v0 + v1)/2

    settings_per_party = InfSDP.setting_cardinalities
    outcomes_per_party = InfSDP.outcome_cardinalities
    SDPmeas = InfSDP.measurements
    
    InfSDP.verbose = 0
    iteration = 0
    last_good = deepcopy(InfSDP)

    while abs(v1 - v0) >= tol_vis and iteration < max_iter:
        InfSDP.set_values(noisy_known_values_W_state(vm, state, state_measurements,
                                                     SDPmeas, settings_per_party,
                                                     outcomes_per_party))
        InfSDP.solve(feas_as_optim=True)
        if verbose:
            print(iteration, "Maximum smallest eigenvalue:", "{:10.4g}".format(
                InfSDP.primal_objective), "\tvisibility =", "{:.4g}".format(vm))
        iteration += 1
        if InfSDP.objective_value >= 0:
            v0 = vm
            vm = (v0 + v1)/2
        elif InfSDP.objective_value < 0:
            v1 = vm
            vm = (v0 + v1)/2
            last_good = deepcopy(InfSDP)
        # if abs(InfSDP.objective_value) <= 1e-7:
        #     break


