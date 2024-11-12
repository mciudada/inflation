# import numpy as np
from inflation import InflationProblem, InflationLP

Classical3322 = InflationProblem(dag={"rho_ABC": ["A", "B", "C"]},
                                    classical_sources="all",
                                    outcomes_per_party=(2, 2, 2),
                                    settings_per_party=(3, 3, 3),
                                    inflation_level_per_source=(1,),
                                    order=["A", "B", "C"])

NoSig3322 = InflationProblem(dag={"rho_ABC": ["A", "B", "C"]},
                                    classical_sources=None,
                                    outcomes_per_party=(2, 2, 2),
                                    settings_per_party=(3, 3, 3),
                                    inflation_level_per_source=(1,),
                                    order=["A", "B", "C"])

NoSig3322LP = InflationLP(NoSig3322, include_all_outcomes=False, verbose=0)

I3322_AB = {"P[A_0=1]": -1,
            "P[A_1=1]": -1,
            "P[B_0=1]": -1,
            "P[B_1=1]": -1,

            "P[A_1=1 B_0=1]": 1,
            "P[A_2=1 B_0=1]": 1,

            "P[A_0=1 B_1=1]": 1,
            "P[A_1=1 B_1=1]": -1,
            "P[A_2=1 B_1=1]": 1,

            "P[A_0=1 B_2=1]": 1,
            "P[A_1=1 B_2=1]": 1,
            "P[A_2=1 B_2=1]": -1}

I3322_BC = {k.replace("B", "C").replace("A","B"): v for k, v in I3322_AB.items()}
from collections import defaultdict
I3322_simult = defaultdict(int)
for k,v in I3322_AB.items():
    I3322_simult[k] += v
for k,v in I3322_BC.items():
    I3322_simult[k] += v
NoSig3322LP.set_objective(I3322_AB, direction="max")
NoSig3322LP.solve()
print("One copy NS maximized:", NoSig3322LP.primal_objective)
NoSig3322LP.set_objective(I3322_simult, direction="max")
NoSig3322LP.solve()
print("P3 version NS maximized:", NoSig3322LP.primal_objective)

Classical3322LP = InflationLP(Classical3322, include_all_outcomes=False, verbose=0)

Classical3322LP.set_objective(I3322_AB, direction="max")
Classical3322LP.solve()
print("One copy classical maximized:", Classical3322LP.primal_objective)
Classical3322LP.set_objective(I3322_simult, direction="max")
Classical3322LP.solve()
print("P3 version classical maximized:", Classical3322LP.primal_objective)