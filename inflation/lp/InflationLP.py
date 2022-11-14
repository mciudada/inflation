"""
The module generates the semidefinite program associated to a quantum inflation
instance (see arXiv:1909.10519).

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np
import sympy as sp

from collections import Counter, deque
from functools import reduce
from gc import collect
from itertools import chain, count, product, permutations, repeat, groupby
from numbers import Real
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Any
from warnings import warn

from inflation import InflationProblem
from ..sdp.fast_npa import (nb_all_commuting_q,
                            apply_source_perm,
                            commutation_matrix,
                            to_canonical)
from ..sdp.fast_npa import nb_is_knowable as is_knowable
from .monomial_classes import InternalAtomicMonomial, CompoundMonomial
from ..sdp.quantum_tools import (clean_coefficients,
                                 expand_moment_normalisation,
                                 flatten_symbolic_powers,
                                 format_permutations,
                                 generate_operators,
                                 party_physical_monomials,
                                 reduce_inflation_indices,
                                 to_symbol)
from ..sdp.sdp_utils import solveSDP_MosekFUSION
from ..utils import flatten


class InflationLP(object):
    """
    Class for generating and solving an LP relaxation for fanout or nonfanout
     inflation.

    Parameters
    ----------
    inflationproblem : InflationProblem
        Details of the scenario.
    nonfanout : bool, optional
        Whether to consider only nonfanout inflation (GPT problem) or
        fanout inflation (classical problem). By default ``False``.
    supports_problem : bool, optional
        Whether to consider feasibility problems with distributions, or just
        with the distribution's support. By default ``False``.
    verbose : int, optional
        Optional parameter for level of verbose:

            * 0: quiet (default),
            * 1: monitor level: track program process and show warnings,
            * 2: debug level: show properties of objects created.
    """
    constant_term_name = "constant_term"

    def __init__(self,
                 inflationproblem: InflationProblem,
                 nonfanout: bool = False,
                 supports_problem: bool = False,
                 verbose=None) -> None:
        """Constructor for the InflationSDP class.
        """
        self.supports_problem = supports_problem
        if verbose is not None:
            if inflationproblem.verbose > verbose:
                warn("Overriding the verbosity from InflationProblem")
            self.verbose = verbose
        else:
            self.verbose = inflationproblem.verbose
        self.nonfanout = nonfanout
        self.commuting = self.nonfanout # Legacy terminology.
        self.InflationProblem = inflationproblem
        self.names = self.InflationProblem.names
        self.names_to_ints = {name: i + 1 for i, name in enumerate(self.names)}
        if self.verbose > 1:
            print(self.InflationProblem)

        self.nr_parties = len(self.names)
        self.nr_sources = self.InflationProblem.nr_sources
        self.hypergraph = self.InflationProblem.hypergraph
        self.inflation_levels = \
            self.InflationProblem.inflation_level_per_source
        self.outcome_cardinalities = \
            np.array(self.InflationProblem.outcomes_per_party,
                     copy=True) + 1
        # We consider all outcomes of all parties in LP inflation.
        self.has_children = np.ones(self.nr_parties, dtype=int)
        self.setting_cardinalities = self.InflationProblem.settings_per_party

        self.measurements = self._generate_parties()
        if self.verbose > 1:
            print("Number of single operator measurements per party:", end="")
            prefix = " "
            for i, measures in enumerate(self.measurements):
                counter = count()
                deque(zip(chain.from_iterable(
                    chain.from_iterable(measures)),
                          counter),
                      maxlen=0)
                print(prefix + f"{self.names[i]}={next(counter)}", end="")
                prefix = ", "
            print()
        self.use_lpi_constraints = False
        self.network_scenario    = self.InflationProblem.is_network
        self._is_knowable_q_non_networks = \
            self.InflationProblem._is_knowable_q_non_networks
        self.rectify_fake_setting = self.InflationProblem.rectify_fake_setting
        self.factorize_monomial = self.InflationProblem.factorize_monomial

        self._nr_operators = len(flatten(self.measurements))
        self._nr_properties = 1 + self.nr_sources + 2
        self.np_dtype = np.find_common_type([
            np.min_scalar_type(np.max(self.setting_cardinalities)),
            np.min_scalar_type(np.max(self.outcome_cardinalities)),
            np.min_scalar_type(self.nr_parties + 1),
            np.min_scalar_type(np.max(self.inflation_levels) + 1)], [])
        self.identity_operator = np.empty((0, self._nr_properties),
                                          dtype=self.np_dtype)
        self.zero_operator = np.zeros((1, self._nr_properties),
                                      dtype=self.np_dtype)

        # Define default lexicographic order through np.lexsort
        lexorder = self._interpret_name(flatten(self.measurements))
        # lexorder = np.concatenate((self.zero_operator, lexorder))
        self._default_lexorder = lexorder[np.lexsort(np.rot90(lexorder))]
        self._lexorder = self._default_lexorder.copy()

        self._default_notcomm = commutation_matrix(self._lexorder,
                                                   self.commuting)
        self._notcomm = self._default_notcomm.copy()
        self.all_commuting_q = lambda mon: nb_all_commuting_q(mon,
                                                              self._lexorder,
                                                              self._notcomm)

        self.canon_ndarray_from_hash    = dict()
        self.canonsym_ndarray_from_hash = dict()
        # These next properties are reset during generate_lp, but are needed in
        # init so as to be able to test the Monomial constructor function
        # without generate_lp.
        self.atomic_monomial_from_hash  = dict()
        self.monomial_from_atoms        = dict()
        self.monomial_from_name         = dict()
        self.Zero = self.Monomial(self.zero_operator, idx=0)
        self.One  = self.Monomial(self.identity_operator, idx=1)
        self.generate_lp()

    ###########################################################################
    # MAIN ROUTINES EXPOSED TO THE USER                                       #
    ###########################################################################
    def generate_lp(self) -> None:
        """Creates the LP associated with the inflation problem.

        In the inflated graph there are many symmetries coming from invariance
        under swaps of the copied sources, which are used to remove variables
        from the LP.
        """
        self.atomic_monomial_from_hash  = dict()
        self.monomial_from_atoms        = dict()
        self.monomial_from_name         = dict()
        self.Zero = self.Monomial(self.zero_operator, idx=0)
        self.One  = self.Monomial(self.identity_operator, idx=1)

        # TODO: Overhual build_columns
        generating_monomials = self.build_columns()
        # Generate dictionary to indices (used in dealing with symmetries and
        # column-level equalities)
        genmon_hash_to_index = {self._from_2dndarray(op): i
                                for i, op in enumerate(generating_monomials)}
        # Check for duplicates
        if len(genmon_hash_to_index) < len(generating_monomials):
            generating_monomials = [generating_monomials[i]
                                    for i in genmon_hash_to_index.values()]
            genmon_hash_to_index = {hash: i for i, hash
                                    in enumerate(genmon_hash_to_index.keys())}
            if self.verbose > 0:
                warn("Duplicates were detected in the list of generating " +
                     "monomials and automatically removed.")
        self.genmon_hash_to_index = genmon_hash_to_index
        self.n_columns            = len(generating_monomials)
        self.generating_monomials = generating_monomials
        del generating_monomials, genmon_hash_to_index
        collect()
        if self.verbose > 0:
            print("Number of nonnegativity constraints in the LP:",
                  self.n_columns)
        symmetrization_required = np.any(self.inflation_levels - 1)

        # Calculate the inflation symmetries
        self.inflation_symmetries = self._discover_inflation_symmetries()
        self.old_reps, orbits = np.unique(
            np.amin(np.vstack((np.arange(self.n_columns),
                               self.inflation_symmetries)),
                    axis=0),
            return_inverse=True)
        old_min = min(self.old_reps)
        self.orbits = orbits + min(self.old_reps)
        self.after_sym_n_columns = len(self.old_reps)
        self.new_reps = np.arange(self.after_sym_n_columns) + old_min
        # Calculate normalization equalities
        self.column_level_equalities = self._discover_normalization_eqns()
        # TODO: merge the above into once consistent concept...

        # Associate Monomials to the remaining entries.
        self.compmonomial_from_idx = dict()
        for idx, old_idx in tqdm(zip(self.new_reps.flat, self.old_reps.flat),
                               disable=not self.verbose,
                               desc="Initializing monomials   ",
                               total=len(self.generating_monomials)):
            mon = self.generating_monomials[old_idx]
            self.compmonomial_from_idx[idx] = self.Monomial(mon, idx)
        self.first_free_idx = max(self.compmonomial_from_idx.keys()) + 1

        self.monomials = list(self.compmonomial_from_idx.values())
        assert all(v == 1 for v in Counter(self.monomials).values()), \
            "Multiple indices are being associated to the same monomial"
        knowable_atoms = set()
        for mon in self.monomials:
            knowable_atoms.update(mon.knowable_factors)
        self.knowable_atoms = [self._monomial_from_atoms([atom])
                               for atom in knowable_atoms]
        del knowable_atoms

        _counter = Counter([mon.knowability_status for mon in self.monomials])
        self.n_knowable           = _counter["Knowable"]
        self.n_something_knowable = _counter["Semi"]
        self.n_unknowable         = _counter["Unknowable"]
        if self.verbose > 1:
            print(f"The problem has {self.n_knowable} knowable monomials, " +
                  f"{self.n_something_knowable} semi-knowable monomials, " +
                  f"and {self.n_unknowable} unknowable monomials.")

        # This dictionary useful for certificates_as_probs
        self.names_to_symbols = {mon.name: mon.symbol
                                 for mon in self.monomials}
        self.names_to_symbols[self.constant_term_name] = sp.S.One

        # In non-network scenarios we do not use Collins-Gisin notation for
        # some variables, so there exist normalization constraints between them
        self.moment_equalities = []
        for (norm_idx, summation_idxs) in self.column_level_equalities:
            new_norm_idx = self.orbits[norm_idx]
            new_summation_idx = self.orbits[list(summation_idxs)]
            eq_dict = {self.compmonomial_from_idx[new_norm_idx]: 1}
            summation_mons = [self.compmonomial_from_idx[idx] for idx
                              in new_summation_idx.flat]
            extra_dict = dict(zip(summation_mons, repeat(-1)))
            eq_dict.update(extra_dict)
            self.moment_equalities.append(eq_dict)

        self.moment_inequalities = []
        self.moment_upperbounds  = dict()
        self.moment_lowerbounds  = {m: 0. for m in self.monomials}

        self._set_lowerbounds(None)
        self._set_upperbounds(None)
        self.set_objective(None)
        self.set_values(None)

        self._relaxation_has_been_generated = True

    def set_bounds(self,
                   bounds: Union[dict, None],
                   bound_type: str = "up") -> None:
        r"""Set numerical lower or upper bounds on the moments generated in the
        SDP relaxation. The bounds are at the level of the SDP variables,
        and do not take into consideration non-convex constraints. E.g., two
        individual lower bounds, :math:`p_A(0|0) \geq 0.1` and
        :math:`p_B(0|0) \geq 0.1` do not directly impose the constraint
        :math:`p_A(0|0)*p_B(0|0) \geq 0.01`, which should be set manually if
        needed.

        Parameters
        ----------
        bounds : Union[dict, None]
            A dictionary with keys as monomials and values being the bounds.
            The keys can be either CompoundMonomial objects, or names (`str`)
            of Monomial objects.
        bound_type : str, optional
            Specifies whether we are setting upper (``"up"``) or lower
            (``"lo"``) bounds, by default "up".

        Examples
        --------
        >>> self.set_bounds({"pAB(00|00)": 0.2}, "lo")
        """
        assert bound_type in ["up", "lo"], \
            "The 'bound_type' argument should be either 'up' or 'lo'"
        if bound_type == "up":
            self._set_upperbounds(bounds)
        else:
            self._set_lowerbounds(bounds)

    def set_distribution(self,
                         prob_array: Union[np.ndarray, None],
                         use_lpi_constraints=False,
                         shared_randomness=False) -> None:
        r"""Set numerically all the knowable (and optionally semiknowable)
        moments according to the probability distribution specified.

        Parameters
        ----------
            prob_array : numpy.ndarray
                Multidimensional array encoding the distribution, which is
                called as ``prob_array[a,b,c,...,x,y,z,...]`` where
                :math:`a,b,c,\dots` are outputs and :math:`x,y,z,\dots` are
                inputs. Note: even if the inputs have cardinality 1 they must
                be specified, and the corresponding axis dimensions are 1.
                The parties' outcomes and measurements must be appear in the
                same order as specified by the ``order`` parameter in the
                ``InflationProblem`` used to instantiate ``InflationSDP``.

            use_lpi_constraints : bool, optional
                Specification whether linearized polynomial constraints (see,
                e.g., Eq. (D6) in `arXiv:2203.16543
                <https://www.arxiv.org/abs/2203.16543/>`_) will be imposed or
                not. By default ``False``.
            shared_randomness : bool, optional
                Specification whether higher order monomials may be calculated.
                If universal shared randomness is present (i.e., the flag is
                ``True``), only atomic monomials are assigned numerical values.
        """
        if prob_array is not None:
            knowable_values = {atom: atom.compute_marginal(prob_array)
                               for atom in self.knowable_atoms}
        else:
            knowable_values = dict()

        self.set_values(knowable_values,
                        use_lpi_constraints=use_lpi_constraints,
                        only_specified_values=shared_randomness)

    def set_objective(self,
                      objective: Union[sp.core.expr.Expr,
                                       dict,
                                       None],
                      direction: str = "max") -> None:
        """Set or change the objective function of the polynomial optimization
        problem.

        Parameters
        ----------
        objective : Union[sp.core.expr.Expr, dict, None]
            The objective function, either as a combination of sympy symbols,
            as a dictionary with keys the monomials or their names, and as
            values the corresponding coefficients, or ``None`` for clearing
            a previous objective.
        direction : str, optional
            Direction of the optimization (``"max"``/``"min"``). By default
            ``"max"``.
        """
        assert (not self.supports_problem) or (objective is None), \
            "Supports problems do not support specifying objective functions."
        assert direction in ["max", "min"], ("The 'direction' argument should "
                                             + " be either 'max' or 'min'")

        self._reset_objective()
        if direction == "max":
            self.maximize = True
        else:
            self.maximize = False
        if objective is None:
            return
        elif isinstance(objective, sp.core.expr.Expr):
            if objective.free_symbols:
                objective_raw = sp.expand(objective).as_coefficients_dict()
                objective_raw = {k: float(v)
                                 for k, v in objective_raw.items()}
            else:
                objective_raw = {self.One: float(objective)}
            return self.set_objective(objective_raw, direction)
        else:
            if self.use_lpi_constraints and self.verbose > 0:
                warn("You have the flag `use_lpi_constraints` set to True. Be "
                     + "aware that imposing linearized polynomial constraints "
                     + "will constrain the optimization to distributions with "
                     + "fixed marginals.")
            sign = (1 if self.maximize else -1)
            objective_dict = {self.One: 0}
            for mon, coeff in objective.items():
                if not np.isclose(coeff, 0):
                    mon = self._sanitise_monomial(mon)
                    objective_dict[mon] = \
                        objective_dict.get(mon, 0) + (sign * coeff)
            self.objective = objective_dict
            surprising_objective_terms = {mon for mon in self.objective.keys()
                                          if mon not in self.monomials}
            assert len(surprising_objective_terms) == 0, \
                ("When interpreting the objective we have encountered at " +
                 "least one monomial that does not appear in the original " +
                 f"generating set:\n\t{surprising_objective_terms}")
            self._update_objective()

    def set_values(self,
                   values: Union[Dict[Union[CompoundMonomial,
                                            InternalAtomicMonomial,
                                            sp.core.symbol.Symbol,
                                            str],
                                      Union[float, sp.core.expr.Expr]],
                                 None],
                   use_lpi_constraints: bool = False,
                   only_specified_values: bool = False) -> None:
        """Directly assign numerical values to variables in the generating set.
        This is done via a dictionary where keys are the variables to have
        numerical values assigned (either in their operator form, in string
        form, or directly referring to the variable in the generating set), and
        the values are the corresponding numerical quantities.

        Parameters
        ----------
        values : Union[None, Dict[Union[CompoundMonomial, InternalAtomicMonomial, sympy.core.symbol.Symbol, str], float]]
            The description of the variables to be assigned numerical values
            and the corresponding values. The keys can be either of the
            Monomial class, symbols or strings (which should be the name of
            some Monomial).

        use_lpi_constraints : bool
            Specification whether linearized polynomial constraints (see, e.g.,
            Eq. (D6) in arXiv:2203.16543) will be imposed or not.

        only_specified_values : bool
            Specifies whether one wishes to fix only the variables provided
            (``True``), or also the variables containing products of the
            monomials fixed (``False``). Regardless of this flag, unknowable
            variables can also be fixed.
        """
        self._reset_values()

        if (values is None) or (len(values) == 0):
            self._cleanup_after_set_values()
            return

        self.use_lpi_constraints = use_lpi_constraints

        if (len(self.objective) > 1) and self.use_lpi_constraints:
            warn("You have an objective function set. Be aware that imposing "
                 + "linearized polynomial constraints will constrain the "
                 + "optimization to distributions with fixed marginals.")

        if not only_specified_values:
            atomic_knowns = {mon.knowable_factors[0]: val
                             for mon, val in self.known_moments.items()
                             if len(mon) == 1}
            atomic_knowns.update({atom.dagger: val
                                  for atom, val in atomic_knowns.items()})
            monomials_not_present = set(self.known_moments.keys()
                                        ).difference(self.monomials)
            for mon in monomials_not_present:
                del self.known_moments[mon]

            # Get the remaining monomials that need assignment
            if all(atom.is_knowable for atom in atomic_knowns):
                if not self.use_lpi_constraints:
                    remaining_mons = (mon for mon in self.monomials
                                      if ((not mon.is_atomic)
                                          and mon.is_knowable))
                else:
                    remaining_mons = (mon for mon in self.monomials
                                      if ((not mon.is_atomic)
                                          and mon.knowability_status
                                          in ["Knowable", "Semi"]))
            else:
                remaining_mons = (mon for mon in self.monomials
                                  if not mon.is_atomic)
            surprising_semiknowns = set()
            for mon in remaining_mons:
                if mon not in self.known_moments.keys():
                    value, unknown_factors, known_status = mon.evaluate(
                        atomic_knowns,
                        self.use_lpi_constraints)
                    if known_status == "Known":
                        self.known_moments[mon] = value
                    elif known_status == "Semi":
                        if self.use_lpi_constraints:
                            unknown_mon = \
                                self._monomial_from_atoms(unknown_factors)
                            self.semiknown_moments[mon] = (value, unknown_mon)
                            if self.verbose > 0:
                                if unknown_mon not in self.monomials:
                                    surprising_semiknowns.add(unknown_mon)
                    else:
                        pass
            if (len(surprising_semiknowns) >= 1) and (self.verbose > 0):
                warn("When processing LPI constraints we encountered at " +
                     "least one monomial that does not appear in the " +
                     f"generating set:\n\t{surprising_semiknowns}")
            del atomic_knowns, surprising_semiknowns
        self._cleanup_after_set_values()

    def solve(self,
              interpreter="MOSEKFusion",
              feas_as_optim=False,
              dualise=True,
              solverparameters=None,
              solver_arguments={}) -> None:
        r"""Call a solver on the SDP relaxation. Upon successful solution, it
        returns the primal and dual objective values along with the solution
        matrices.

        Parameters
        ----------
        interpreter : str, optional
            The solver to be called. By default ``"MOSEKFusion"``.
        feas_as_optim : bool, optional
            Instead of solving the feasibility problem

                :math:`(1) \text{ find vars such that } \Gamma \succeq 0`

            setting this label to ``True`` solves instead the problem

                :math:`(2) \text{ max }\lambda\text{ such that }
                \Gamma - \lambda\cdot 1 \succeq 0.`

            The correspondence is that the result of (2) is positive if (1) is
            feasible, and negative otherwise. By default ``False``.
        dualise : bool, optional
            Optimize the dual problem (recommended). By default ``True``.
        solverparameters : dict, optional
            Extra parameters to be sent to the solver. By default ``None``.
        solver_arguments : dict, optional
            By default, solve will use the dictionary of SDP keyword arguments
            given by ``_prepare_solver_arguments()``. However, a user may
            manually override these arguments by passing their own here.
        """
        if not self._relaxation_has_been_generated:
            raise Exception("Relaxation is not generated yet. " +
                            "Call \"InflationSDP.get_relaxation()\" first")
        if feas_as_optim and len(self._processed_objective) > 1:
            warn("You have a non-trivial objective, but set to solve a " +
                 "feasibility problem as optimization. Setting "
                 + "feas_as_optim=False and optimizing the objective...")
            feas_as_optim = False

        args = self._prepare_solver_arguments()
        args.update(solver_arguments)
        args.update({"feas_as_optim": feas_as_optim,
                     "verbose": self.verbose,
                     "solverparameters": solverparameters,
                     "solve_dual": dualise})

        self.solution_object = solveSDP_MosekFUSION(**args)

        self.status = self.solution_object["status"]
        if self.status == "feasible":
            self.primal_objective = self.solution_object["primal_value"]
            self.objective_value  = self.solution_object["primal_value"]
            self.objective_value *= (1 if self.maximize else -1)
        else:
            self.primal_objective = self.status
            self.objective_value  = self.status
        collect()

    ###########################################################################
    # PUBLIC ROUTINES RELATED TO THE PROCESSING OF CERTIFICATES               #
    ###########################################################################
    def certificate_as_probs(self,
                             clean: bool = True,
                             chop_tol: float = 1e-10,
                             round_decimals: int = 3) -> sp.core.add.Add:
        """Give certificate as symbolic sum of probabilities. The certificate
        of incompatibility is ``cert < 0``.

        Parameters
        ----------
        clean : bool, optional
            If ``True``, eliminate all coefficients that are smaller than
            ``chop_tol``, normalise and round to the number of decimals
            specified by ``round_decimals``. By default ``True``.
        chop_tol : float, optional
            Coefficients in the dual certificate smaller in absolute value are
            set to zero. By default ``1e-10``.
        round_decimals : int, optional
            Coefficients that are not set to zero are rounded to the number of
            decimals specified. By default ``3``.

        Returns
        -------
        sympy.core.add.Add
            The expression of the certificate in terms or probabilities and
            marginals. The certificate of incompatibility is ``cert < 0``.
        """
        try:
            dual = self.solution_object["dual_certificate"]
        except AttributeError:
            raise Exception("For extracting a certificate you need to solve " +
                            "a problem. Call \"InflationSDP.solve()\" first.")
        if len(self.semiknown_moments) > 0:
            warn("Beware that, because the problem contains linearized " +
                 "polynomial constraints, the certificate is not guaranteed " +
                 "to apply to other distributions.")
        if clean and not np.allclose(list(dual.values()), 0.):
            dual = clean_coefficients(dual, chop_tol, round_decimals)

        polynomial = sp.S.Zero
        for mon_name, coeff in dual.items():
            if clean and np.isclose(int(coeff), round(coeff, round_decimals)):
                coeff = int(coeff)
            polynomial += coeff * self.names_to_symbols[mon_name]
        return polynomial

    # TODO: Restore
    # def certificate_as_string(self,
    #                           clean: bool = True,
    #                           chop_tol: float = 1e-10,
    #                           round_decimals: int = 3) -> str:

    ###########################################################################
    # OTHER ROUTINES EXPOSED TO THE USER                                      #
    ###########################################################################
    def build_columns(self,
                      symbolic: bool = False) -> List[np.ndarray]:
        r"""Creates the generating set of monomials.

        Parameters
        ----------
        symbolic: bool, optional
            If ``True``, it returns the columns as a list of sympy symbols
            parsable by `InflationLP.generate_lp()`. By default
            ``False``.
        """
        if not self.nonfanout:
            hash_except_outcome = lambda mon: mon[:-1].tobytes()

            non_orthogonal_choices = [
                (self.identity_operator, ) + tuple(ortho_group)
                for _, ortho_group in groupby(self._lexorder,
                                              key=hash_except_outcome)]
            lengths = list(map(len, non_orthogonal_choices))
            # nontriv_cols = map(np.vstack, product(*non_orthogonal_choices))
            nontriv_cols = map(np.vstack,
                               tqdm(product(*non_orthogonal_choices),
                                    disable=not self.verbose,
                                    desc="Building columns         ",
                                    total=np.prod(lengths),
                                    leave=True,
                                    position=0))
            # generating_monomials = [self.identity_operator]
        else:
            commuting_seqs_per_party = []
            for party in range(self.nr_parties):
                relevant_sources = np.flatnonzero(self.hypergraph[:, party])
                relevant_inflevels = self.inflation_levels[relevant_sources]
                max_mon_length = min(relevant_inflevels)
                commuting_seqs_per_party.append([party_physical_monomials(
                    hypergraph=self.hypergraph,
                    inflevels=self.inflation_levels,
                    party=party,
                    max_monomial_length=i,
                    settings_per_party=self.setting_cardinalities,
                    outputs_per_party=self.outcome_cardinalities,
                    lexorder=self._lexorder)
                    for i in range(max_mon_length+1)])
            nontriv_cols = map(np.vstack, product(*commuting_seqs_per_party))
        generating_monomials = list(nontriv_cols)





        # party_freqs = sorted((list(pfreq)
        #                       for pfreq in product(
        #                        *[range(level + 1) for level in lengths]
        #                                            )
        #                       if sum(pfreq) <= max_length),
        #                      key=lambda x: (sum(x), [-p for p in x]))
        #
        # physical_monomials = []
        # for freqs in party_freqs:
        #     if freqs == [0] * self.nr_parties:
        #         physical_monomials.append(self.identity_operator)
        #     else:
        #         physmons_per_party = []
        #         for party, freq in enumerate(freqs):
        #             if freq > 0:
        #                 physmons = party_physical_monomials(
        #                     self.hypergraph,
        #                     self.inflation_levels,
        #                     party, freq,
        #                     self.setting_cardinalities,
        #                     self.outcome_cardinalities,
        #                     self._lexorder)
        #                 physmons_per_party.append(physmons)
        #         for monomial_parts in product(
        #                 *physmons_per_party):
        #             physical_monomials.append(
        #                 self._to_canonical_memoized(
        #                     np.concatenate(monomial_parts)))

        if symbolic:
            generating_monomials = [to_symbol(col, self.names)
                                    for col in generating_monomials]
        return generating_monomials

    def reset(self, which: Union[str, List[str]]) -> None:
        """Reset the various user-specifiable objects in the inflation SDP.

        Parameters
        ----------
        which : Union[str, List[str]]
            The objects to be reset. It can be fed as a single string or a list
            of them. Options include ``"bounds"``, ``"lowerbounds"``,
            ``"upperbounds"``, ``"objective"``, ``"values"``, and ``"all"``.
        """
        if type(which) == str:
            if which == "all":
                self.reset(["bounds", "objective", "values"])
            elif which == "bounds":
                self._reset_bounds()
            elif which == "lowerbounds":
                self._reset_lowerbounds()
            elif which == "upperbounds":
                self._reset_upperbounds()
            elif which == "objective":
                self._reset_objective()
            elif which == "values":
                self._reset_values()
            else:
                raise Exception(f"The attribute {which} is not part of " +
                                "InflationSDP.")
        else:
            for attr in which:
                self.reset(attr)
        collect()

    # TODO: Restore export functionality
    # def write_to_file(self, filename: str) -> None:


    ###########################################################################
    # ROUTINES RELATED TO CONSTRUCTING COMPOUND MONOMIAL INSTANCES            #
    ###########################################################################
    def _AtomicMonomial(self,
                        array2d: np.ndarray) -> InternalAtomicMonomial:
        """Construct an instance of the `InternalAtomicMonomial` class from
        a 2D array description of a monomial.

        See the documentation of the `InternalAtomicMonomial` class for more
        details.

        Parameters
        ----------
        array2d : numpy.ndarray
            Monomial encoded as a 2D array of integers, where each row encodes
            one of the operators appearing in the monomial.

        Returns
        -------
        InternalAtomicMonomial
            An instance of the `InternalAtomicMonomial` class representing the
            input 2D array monomial.
        """
        key = self._from_2dndarray(array2d)
        try:
            return self.atomic_monomial_from_hash[key]
        except KeyError:
            repr_array2d = self._to_inflation_repr(array2d)
            new_key      = self._from_2dndarray(repr_array2d)
            try:
                mon = self.atomic_monomial_from_hash[new_key]
                self.atomic_monomial_from_hash[key] = mon
                return mon
            except KeyError:
                mon = InternalAtomicMonomial(self, repr_array2d)
                self.atomic_monomial_from_hash[key]     = mon
                self.atomic_monomial_from_hash[new_key] = mon
                return mon

    def Monomial(self, array2d: np.ndarray, idx=-1) -> CompoundMonomial:
        r"""Create an instance of the `CompoundMonomial` class from a 2D array.
        An instance of `CompoundMonomial` is a collection of
        `InternalAtomicMonomial`.

        Parameters
        ----------
        array2d : numpy.ndarray
            Moment encoded as a 2D array of integers, where each row encodes
            one of the operators appearing in the moment.
        idx : int, optional
            Assigns an integer index to the resulting monomial, which can be
            used as an id, by default -1.

        Returns
        -------
        CompoundMonomial
            The monomial factorised into AtomicMonomials, all brought to
            representative form under inflation symmetries.

        Examples
        --------

        The moment
        :math:`\langle A^{0,2,1}_{x=2,a=3}C^{2,0,1}_{z=1,c=1}
        C^{1,0,2}_{z=0,c=0}\rangle` corresponds to the following 2D array:

        >>> m = np.array([[1, 0, 2, 1, 2, 3],
                          [3, 2, 0, 1, 1, 1],
                          [3, 1, 0, 2, 0, 0]])

        The resulting monomial, ``InflationSDP.Monomial(m)``, is a collection
        of two ``InternalAtomicMonomial`` s,
        :math:`\langle A^{0,1,1}_{x=2,a=3}C^{1,0,1}_{z=1,c=1}\rangle` and
        :math:`\langle C^{1,0,1}_{z=0,c=0}\rangle`, after factorizing the input
        monomial and reducing the inflation indices of each of the factors.
        """
        _factors = self.factorize_monomial(array2d, canonical_order=False)
        list_of_atoms = [self._AtomicMonomial(factor)
                         for factor in _factors if len(factor)]
        mon = self._monomial_from_atoms(list_of_atoms)
        mon.attach_idx(idx)
        return mon

    def _inflation_orbit_and_rep(self,
                                 monomial: np.ndarray
                                 ) -> Tuple[set, np.ndarray]:
        """Given a monomial as a 2D array, return its representative under
        inflation symmetries and its orbit. Only source swaps up to the maximum
        index of the source that appears in the monomials are considered.

        Parameters
        ----------
        monomial : numpy.ndarray
            Monomial as a 2D array.

        Returns
        -------
        Tuple[set, numpy.ndarray]
            The orbit as a set of all monomials explored, and the
            representative (i.e, the minimum over said set).
        """
        inf_levels = monomial[:, 1:-2].max(axis=0)
        nr_sources = inf_levels.shape[0]
        all_permutations_per_source = [
            format_permutations(list(permutations(range(inflevel))))
            for inflevel in inf_levels.flat]
        seen_hashes = set()
        for permutation in product(*all_permutations_per_source):
            permuted = monomial.copy()
            for source in range(nr_sources):
                permuted = apply_source_perm(permuted,
                                             source,
                                             permutation[source])
            permuted = self._to_canonical_memoized(permuted, True)
            hash     = self._from_2dndarray(permuted)
            seen_hashes.add(hash)
            try:
                representative = self.canonsym_ndarray_from_hash[hash]
                return seen_hashes, representative
            except KeyError:
                pass
        representative = self._to_2dndarray(min(seen_hashes))
        return seen_hashes, representative

    def _monomial_from_atoms(self,
                             atoms: List[InternalAtomicMonomial]
                             ) -> CompoundMonomial:
        """Build an instance of `CompoundMonomial` from a list of instances
        of `InternalAtomicMonomial`.

        Parameters
        ----------
        atoms : List[InternalAtomicMonomial]
            List of instances of `InternalAtomicMonomial`.

        Returns
        -------
        CompoundMonomial
            A `CompoundMonomial` with atomic factors given by `atoms`.
        """
        list_of_atoms = []
        for factor in atoms:
            if factor.is_zero:
                list_of_atoms = [factor]
                break
            elif not factor.is_one:
                list_of_atoms.append(factor)
            else:
                pass
        atoms = tuple(sorted(list_of_atoms))
        try:
            mon = self.monomial_from_atoms[atoms]
            return mon
        except KeyError:
            mon = CompoundMonomial(atoms)
            try:
                mon.idx = self.first_free_idx
                self.first_free_idx += 1
            except AttributeError:
                pass
            self.monomial_from_atoms[atoms]   = mon
            self.monomial_from_name[mon.name] = mon
            return mon

    def _sanitise_monomial(self, mon: Any) -> CompoundMonomial:
        """Return a ``CompoundMonomial`` built from ``mon``, where ``mon`` can
        be either the name of a moment as a string, a SymPy variable, a
        monomial encoded as a 2D array, or an integer in case the moment is the
        unit moment or the zero moment.


        Parameters
        ----------
        mon : Any
            The name of a moment as a string, a SymPy variable with the name of
            a valid moment, a 2D array encoding of a moment or an integer in
            case the moment is the unit moment or the zero moment.

        Returns
        -------
        CompoundMonomial
            Instance of ``CompoundMonomial`` built from ``mon``.

        Raises
        ------
        Exception
            If ``mon`` is the constant monomial, it can only be numbers 0 or 1
        Exception
            If the type of ``mon`` is not supported.
        """
        if isinstance(mon, CompoundMonomial):
            return mon
        elif isinstance(mon, (sp.core.symbol.Symbol,
                              sp.core.power.Pow,
                              sp.core.mul.Mul)):
            symbols = flatten_symbolic_powers(mon)
            if len(symbols) == 1:
                try:
                    return self.monomial_from_name[str(symbols[0])]
                except KeyError:
                    pass
            array = np.concatenate([self._interpret_atomic_string(str(op))
                                    for op in symbols])
            return self._sanitise_monomial(array)
        elif isinstance(mon, (tuple, list, np.ndarray)):
            array = np.asarray(mon, dtype=self.np_dtype)
            assert array.ndim == 2, \
                "The monomial representations must be 2d arrays."
            assert array.shape[-1] == self._nr_properties, \
                "The input does not conform to the operator specification."
            canon = self._to_canonical_memoized(array)
            return self.Monomial(canon)
        elif isinstance(mon, str):
            try:
                return self.monomial_from_name[mon]
            except KeyError:
                return self._sanitise_monomial(self._interpret_name(mon))
        elif isinstance(mon, Real):
            if np.isclose(float(mon), 1):
                return self.One
            elif np.isclose(float(mon), 0):
                return self.Zero
            else:
                raise Exception(f"Constant monomial {mon} can only be 0 or 1.")
        else:
            raise Exception(f"sanitise_monomial: {mon} is of type " +
                            f"{type(mon)} and is not supported.")

    def _to_inflation_repr(self,
                           mon: np.ndarray,
                           apply_only_commutations=False) -> np.ndarray:
        r"""Apply inflation symmetries to a monomial in order to bring it to
        its canonical form.

        Example: Assume the monomial is :math:`\langle D^{350}_{00}D^{450}_{00}
        D^{150}_{00}E^{401}_{00}F^{031}_{00}\rangle`. In array form, the
        information about inflation copies is:

        ::

            [[3 5 0],
             [4 5 0],
             [1 5 0],
             [4 0 1],
             [0 3 1]]

        For each column the function assigns to the first row index 1. Then,
        the next different one will be 2, and so on. Therefore, the
        representative of the monomial above is :math:`\langle D^{110}_{00}
        D^{210}_{00} D^{310}_{00} E^{201}_{00} F^{021}_{00} \rangle`.

        Parameters
        ----------
        mon : numpy.ndarray
            Input monomial that cannot be further factorised.
        apply_only_commutations : bool, optional
            If ``True``, skip checking if monomial is zero and if there are
            multiple same projectors that square to just one of them.

        Returns
        -------
        numpy.ndarray
            The canonical form of the input monomial under relabelling through
            the inflation symmetries.
        """
        key = self._from_2dndarray(mon)
        if len(mon) == 0 or np.array_equiv(mon, 0):
            self.canonsym_ndarray_from_hash[key] = mon
            return mon
        else:
            pass
        try:
            return self.canonsym_ndarray_from_hash[key]
        except KeyError:
            pass
        canonical_mon = self._to_canonical_memoized(mon,
                                                    apply_only_commutations)
        canonical_key = self._from_2dndarray(canonical_mon)
        try:
            repr_mon = self.canonsym_ndarray_from_hash[canonical_key]
            self.canonsym_ndarray_from_hash[key] = repr_mon
            return repr_mon
        except KeyError:
            pass
        repr_mon = reduce_inflation_indices(mon)
        repr_key = self._from_2dndarray(repr_mon)
        try:
            real_repr_mon = self.canonsym_ndarray_from_hash[repr_key]
            self.canonsym_ndarray_from_hash[key]           = real_repr_mon
            self.canonsym_ndarray_from_hash[canonical_key] = real_repr_mon
            return real_repr_mon
        except KeyError:
            pass
        other_keys, real_repr_mon = self._inflation_orbit_and_rep(repr_mon)
        other_keys.update({key, canonical_key, repr_key})
        for key in other_keys:
            self.canonsym_ndarray_from_hash[key] = real_repr_mon
        return real_repr_mon

    ###########################################################################
    # ROUTINES RELATED TO NAME PARSING                                        #
    ###########################################################################
    def _interpret_name(self,
                        monomial: Union[str, sp.core.symbol.Expr, int]
                        ) -> np.ndarray:
        """Build a 2D array encoding of a monomial which can be passed either
        as a string, as a SymPy expression or as an integer.

        Parameters
        ----------
        monomial : Union[str, sympy.core.symbol.Expr, int]
            Input moment.

        Returns
        -------
        numpy.ndarray
            2D array encoding of the input moment.
        """
        if isinstance(monomial, sp.core.symbol.Expr):
            factors = [str(factor)
                       for factor in flatten_symbolic_powers(monomial)]
        elif str(monomial) == '1':
            return self.identity_operator
        elif isinstance(monomial, tuple) or isinstance(monomial, list):
            factors = [str(factor) for factor in monomial]
        else:
            assert "^" not in monomial, "Cannot interpret exponents."
            factors = monomial.split("*")
        return np.vstack(tuple(self._interpret_atomic_string(factor_string)
                               for factor_string in factors))

    def _interpret_atomic_string(self, factor_string: str) -> np.ndarray:
        """Build a 2D array encoding of a moment that cannot be further
        factorised into products of other moments.

        Parameters
        ----------
        factor_string : str
            String representation of a moment in expected value notation, e.g.,
            ``"<A_1_1_1_2*B_2_1_3_4>"``.

        Returns
        -------
        numpy.ndarray
            2D array encoding of the input atomic moment.
        """
        assert ((factor_string[0] == "<" and factor_string[-1] == ">")
                or set(factor_string).isdisjoint(set("| "))), \
            ("Monomial names must be between < > signs, or in conditional " +
             "probability form.")
        if factor_string[0] == "<":
            operators = factor_string[1:-1].split(" ")
            return np.vstack(tuple(self._interpret_operator_string(op_string)
                                   for op_string in operators))
        else:
            return self._interpret_operator_string(factor_string)[np.newaxis]

    def _interpret_operator_string(self, op_string: str) -> np.ndarray:
        """Build a 1D array encoding of an operator passed as a string.

        Parameters
        ----------
        factor_string : str
            String representation of an operator, e.g., ``"B_2_1_3_4"``.

        Returns
        -------
        numpy.ndarray
            2D array encoding of the operator.
        """
        components = op_string.split("_")
        assert len(components) == self._nr_properties, \
            f"There need to be {self._nr_properties} properties to match " + \
            "the scenario."
        components[0] = self.names_to_ints[components[0]]
        return np.array([int(s) for s in components], dtype=self.np_dtype)

    ###########################################################################
    # ROUTINES RELATED TO THE GENERATION OF THE MOMENT MATRIX                 #
    ###########################################################################
    def _discover_normalization_eqns(self) -> List[Tuple[int, List[int]]]:
        """Given the generating monomials, infer implicit normalization
        equalities between them. Each normalization equality is a two element
        tuple; the first element is an integer indicating a particular column,
        the second element is a list of integers indicating other columns such
        that the sum of the latter columns is equal to the former column.

        Returns
        -------
        List[Tuple[int, List[int]]]
            A list of normalization equalities between columns.
        """
        skip_party = [not i for i in self.has_children]
        column_level_equalities = []
        for i, mon in enumerate(self.generating_monomials):
            eqs = expand_moment_normalisation(mon,
                                              self.outcome_cardinalities,
                                              skip_party)
            for eq in eqs:
                try:
                    eq_idxs = [self.genmon_hash_to_index[
                                                self._from_2dndarray(eq[0])]]
                    eq_idxs.append([self.genmon_hash_to_index[
                                    self._from_2dndarray(m)] for m in eq[1]])
                    column_level_equalities += [tuple(eq_idxs)]
                except KeyError:
                    break
        return column_level_equalities

    def _discover_inflation_symmetries(self) -> np.ndarray:
        """Calculates all the symmetries pertaining to the set of generating
        monomials. The new set of operators is a permutation of the old. The
        function outputs a list of all permutations.

        Returns
        -------
        numpy.ndarray[int]
            The list of all permutations of the generating columns implied by
            the inflation symmetries.
        """
        sources_with_copies = [source for source, inf_level
                               in enumerate(self.inflation_levels)
                               if inf_level > 1]
        if len(sources_with_copies):
            inflation_symmetries = []
            identity_perm        = np.arange(self.n_columns, dtype=int)
            for source in tqdm(sources_with_copies,
                               disable=not self.verbose,
                               desc="Calculating symmetries   ",
                               leave=False,
                               position=0):
                one_source_symmetries = [identity_perm]
                inf_level = self.inflation_levels[source]
                perms = format_permutations(list(
                    permutations(range(inf_level)))[1:])
                permutation_failed = False
                for permutation in perms:
                    try:
                        total_perm = np.empty(self.n_columns, dtype=int)
                        for i, mon in enumerate(self.generating_monomials):
                            new_mon = apply_source_perm(mon,
                                                        source,
                                                        permutation)
                            new_mon = self._to_canonical_memoized(new_mon,
                                                                  True)
                            total_perm[i] = self.genmon_hash_to_index[
                                                self._from_2dndarray(new_mon)]
                        one_source_symmetries.append(total_perm)
                    except KeyError:
                        permutation_failed = True
                inflation_symmetries.append(one_source_symmetries)
            if permutation_failed and (self.verbose > 0):
                warn("The generating set is not closed under source swaps."
                     + " Some symmetries will not be implemented.")
            inflation_symmetries = [reduce(np.take, perms) for perms in
                                    product(*inflation_symmetries)]
            return np.unique(inflation_symmetries[1:], axis=0)
        else:
            return np.empty((0, len(self.generating_monomials)), dtype=int)

    def _generate_parties(self) -> List[List[List[List[sp.Symbol]]]]:
        """Generates all the party operators in the quantum inflation.

        Returns
        -------
        List[List[List[List[sympy.Symbol]]]]
            The measurement operators as symbols. The array is indexed as
            measurements[p][c][i][o] for party p, inflation copies c, input i,
            and output o.
        """
        settings = self.setting_cardinalities
        outcomes = self.outcome_cardinalities

        assert len(settings) == len(outcomes), \
            "There\'s a different number of settings and outcomes"
        assert len(settings) == self.hypergraph.shape[1], \
            "The hypergraph does not have as many columns as parties"
        measurements = []
        parties = self.names
        n_states = self.hypergraph.shape[0]
        for pos, [party, ins, outs] in enumerate(zip(parties,
                                                     settings,
                                                     outcomes)):
            party_meas = []
            # Generate all possible copy indices for a party
            all_inflation_indices = product(
                *[list(range(self.inflation_levels[p_idx]))
                  for p_idx in np.flatnonzero(self.hypergraph[:, pos])])
            # Include zeros in the positions of states not feeding the party
            all_indices = []
            for inflation_indices in all_inflation_indices:
                indices = []
                i = 0
                for idx in range(n_states):
                    if self.hypergraph[idx, pos] == 0:
                        indices.append("0")
                    elif self.hypergraph[idx, pos] == 1:
                        # The +1 is just to begin at 1
                        indices.append(str(inflation_indices[i] + 1))
                        i += 1
                    else:
                        raise Exception("You don\'t have a proper hypergraph")
                all_indices.append(indices)
            # Generate measurements for every combination of indices.
            # The -1 in outs - 1 is because the use of Collins-Gisin notation
            # (see [arXiv:quant-ph/0306129]), whereby the last operator is
            # understood to be written as the identity minus the rest.
            for indices in all_indices:
                meas = generate_operators(
                    [outs - 1 for _ in range(ins)],
                    party + "_" + "_".join(indices)
                )
                party_meas.append(meas)
            measurements.append(party_meas)
        return measurements

    ###########################################################################
    # HELPER FUNCTIONS FOR ENSURING CONSISTENCY                               #
    ###########################################################################
    def _cleanup_after_set_values(self) -> None:
        """Helper function to reset or make consistent class attributes after
        setting values."""
        if self.supports_problem:
            # Add lower bounds to monomials inside the support
            nonzero_known_monomials = [mon for
                                       mon, value in self.known_moments.items()
                                       if not np.isclose(value, 0)]
            for mon in nonzero_known_monomials:
                self._processed_moment_lowerbounds[mon] = 1.
                del self.known_moments[mon]
            self.semiknown_moments = dict()

        self._update_lowerbounds()
        self._update_upperbounds()
        self._update_objective()
        num_nontrivial_known = len(self.known_moments)
        if self.verbose > 1 and num_nontrivial_known > 0:
            print("Number of variables with fixed numeric value:",
                  num_nontrivial_known)
        num_semiknown = len(self.semiknown_moments)
        if self.verbose > 1 and num_semiknown > 0:
            print(f"Number of semiknown variables: {num_semiknown}")

    def _reset_bounds(self) -> None:
        """Reset the lists of bounds."""
        self._reset_lowerbounds()
        self._reset_upperbounds()
        collect()

    def _reset_lowerbounds(self) -> None:
        """Reset the list of lower bounds."""
        self._reset_solution()
        self._processed_moment_lowerbounds = dict()

    def _reset_upperbounds(self) -> None:
        """Reset the list of upper bounds."""
        self._reset_solution()
        self._processed_moment_upperbounds = dict()

    def _reset_objective(self) -> None:
        """Reset the objective function."""
        self._reset_solution()
        self.objective = {self.One: 0.}
        self._processed_objective = self.objective
        self.maximize = True  # Direction of the optimization

    def _reset_values(self) -> None:
        """Reset the known values."""
        self._reset_solution()
        self.known_moments     = dict()
        self.semiknown_moments = dict()
        self.known_moments[self.One] = 1.
        collect()

    def _update_objective(self) -> None:
        """Process the objective with the information from known_moments
        and semiknown_moments.
        """
        self._processed_objective = self.objective.copy()
        knowns_to_process = set(self.known_moments.keys()
                                ).intersection(
                                    self._processed_objective.keys())
        knowns_to_process.discard(self.One)
        for m in knowns_to_process:
            value = self.known_moments[m]
            self._processed_objective[self.One] += \
                self._processed_objective[m] * value
            del self._processed_objective[m]
        semiknowns_to_process = set(self.semiknown_moments.keys()
                                    ).intersection(
                                        self._processed_objective.keys())
        for mon in semiknowns_to_process:
            coeff = self._processed_objective[mon]
            for (subs_coeff, subs) in self.semiknown_moments[mon]:
                self._processed_objective[subs] = \
                    self._processed_objective.get(subs, 0) + coeff * subs_coeff
                del self._processed_objective[mon]
        collect()

    def _update_lowerbounds(self) -> None:
        """Helper function to check that lowerbounds are consistent with the
        specified known values, and to keep only the lowest lowerbounds
        in case of redundancy.
        """
        for mon, lb in self.moment_lowerbounds.items():
            self._processed_moment_lowerbounds[mon] = \
                max(self._processed_moment_lowerbounds.get(mon, -np.infty), lb)
        for mon, value in self.known_moments.items():
            if isinstance(value, Real):
                try:
                    lb = self._processed_moment_lowerbounds[mon]
                    assert lb <= value, (f"Value {value} assigned for " +
                                         f"monomial {mon} contradicts the " +
                                         f"assigned lower bound of {lb}.")
                    del self._processed_moment_lowerbounds[mon]
                except KeyError:
                    pass
        self.moment_lowerbounds = self._processed_moment_lowerbounds

    def _update_upperbounds(self) -> None:
        """Helper function to check that upperbounds are consistent with the
        specified known values.
        """
        for mon, value in self.known_moments.items():
            if isinstance(value, Real):
                try:
                    ub = self._processed_moment_upperbounds[mon]
                    assert ub >= value, (f"Value {value} assigned for " +
                                         f"monomial {mon} contradicts the " +
                                         f"assigned upper bound of {ub}.")
                    del self._processed_moment_upperbounds[mon]
                except KeyError:
                    pass
        self.moment_upperbounds = self._processed_moment_upperbounds

    ###########################################################################
    # OTHER ROUTINES                                                          #
    ###########################################################################
    def _atomic_knowable_q(self, atomic_monarray: np.ndarray) -> bool:
        """Return ``True`` if the input monomial, encoded as a 2D array,
        can be associated to a knowable value in the scenario, and ``False``
        otherwise.

        Parameters
        ----------
        atomic_monarray : numpy.ndarray
            Monomial encoded as a 2D array.

        Returns
        -------
        bool
            Whether the monomial could be assigned a numerical value.
        """
        if not is_knowable(atomic_monarray):
            return False
        elif self.network_scenario:
            return True
        else:
            return self._is_knowable_q_non_networks(np.take(atomic_monarray,
                                                            [0, -2, -1],
                                                    axis=1))

    def _from_2dndarray(self, array2d: np.ndarray) -> bytes:
        """Obtains the bytes representation of an array. The library uses this
        representation as hashes for the corresponding monomials.

        Parameters
        ----------
        array2d : numpy.ndarray
            Monomial encoded as a 2D array.
        """
        return np.asarray(array2d, dtype=self.np_dtype).tobytes()

    def _prepare_solver_arguments(self) -> dict:
        """Prepare arguments to pass to the solver.

        The solver takes as input the following arguments, which are all
        dicts with keys as scalar SDP variables:
            * "objective": dict with values the coefficient of the key
            variable in the objective function.
            * "known_vars": scalar variables that are fixed to be constant.
            * "semiknown_vars": if applicable, linear proportionality
            constraints between variables in the SDP.
            * "equalities": list of dicts where each dict gives the
            coefficients of the keys in a linear equality constraint.
            * "inequalities": list of dicts where each dict gives the
            coefficients of the keys in a linear inequality constraint.

        Returns
        -------
        dict
            A tuple with the arguments to be passed to the solver.

        Raises
        ------
        Exception
            If the SDP relaxation has not been calculated yet.
        """
        if not self._relaxation_has_been_generated:
            raise Exception("Relaxation is not generated yet. " +
                            "Call \"InflationSDP.get_relaxation()\" first")

        assert set(self.known_moments.keys()).issubset(self.monomials),\
            ("Error: Tried to assign known values outside of the variables: " +
             str(set(self.known_moments.keys()
                     ).difference(self.monomials)))

        solverargs = {"objective": {mon.name: coeff for mon, coeff
                                    in self._processed_objective.items()},
                      "known_vars": {mon.name: val for mon, val
                                     in self.known_moments.items()},
                      "semiknown_vars": {mon.name: (coeff, subs.name)
                                         for mon, (coeff, subs)
                                         in self.semiknown_moments.items()},
                      "equalities": [{mon.name: coeff
                                          for mon, coeff in eq.items()}
                                         for eq in self.moment_equalities],
                      "inequalities": [{mon.name: coeff
                                        for mon, coeff in ineq.items()}
                                       for ineq in self.moment_inequalities]
                      }
        # Add the constant 1 in case of unnormalized problems removed it
        solverargs["known_vars"][self.constant_term_name] = 1.
        for mon, bnd in self._processed_moment_lowerbounds.items():
            lb = {mon.name: 1}
            if not np.isclose(bnd, 0):
                lb[self.constant_term_name] = -bnd
            solverargs["inequalities"].append(lb)
        for mon, bnd in self._processed_moment_upperbounds.items():
            ub = {mon.name: -1}
            if not np.isclose(bnd, 0):
                ub[self.constant_term_name] = bnd
            solverargs["inequalities"].append(ub)
        return solverargs

    def _reset_solution(self) -> None:
        """Resets class attributes storing the solution to the SDP
        relaxation."""
        for attribute in {"primal_objective",
                          "objective_value",
                          "solution_object"}:
            try:
                delattr(self, attribute)
            except AttributeError:
                pass
        self.status = "Not yet solved"

    def _set_upperbounds(self, upperbounds: Union[dict, None]) -> None:
        """Set upper bounds for variables in the SDP relaxation.

        Parameters
        ----------
        upperbounds : Union[dict, None]
            Dictionary with keys as moments and values as upper bounds. The
            keys can be either strings, instances of `CompoundMonomial` or
            moments encoded as 2D arrays.
        """
        self._reset_upperbounds()
        if upperbounds is None:
            return
        sanitized_upperbounds = dict()
        for mon, upperbound in upperbounds.items():
            mon = self._sanitise_monomial(mon)
            if mon not in sanitized_upperbounds.keys():
                sanitized_upperbounds[mon] = upperbound
            else:
                old_bound = sanitized_upperbounds[mon]
                assert np.isclose(old_bound,
                                  upperbound), \
                    (f"Contradiction: Cannot set the same monomial {mon} to " +
                     "have different upper bounds.")
        self._processed_moment_upperbounds = sanitized_upperbounds
        self._update_upperbounds()

    def _set_lowerbounds(self, lowerbounds: Union[dict, None]) -> None:
        """Set lower bounds for variables in the SDP relaxation.

        Parameters
        ----------
        lowerbounds : Union[dict, None]
            Dictionary with keys as moments and values as upper bounds. The
            keys can be either strings, instances of `CompoundMonomial` or
            moments encoded as 2D arrays.
        """
        self._reset_lowerbounds()
        if lowerbounds is None:
            return
        sanitized_lowerbounds = dict()
        for mon, lowerbound in lowerbounds.items():
            mon = self._sanitise_monomial(mon)
            if mon not in sanitized_lowerbounds.keys():
                sanitized_lowerbounds[mon] = lowerbound
            else:
                old_bound = sanitized_lowerbounds[mon]
                assert np.isclose(old_bound, lowerbound), \
                    (f"Contradiction: Cannot set the same monomial {mon} to " +
                     "have different lower bounds.")
        self._processed_moment_lowerbounds = sanitized_lowerbounds
        self._update_lowerbounds()

    def _to_2dndarray(self, bytestream: bytes) -> np.ndarray:
        """Create a monomial array from its corresponding stream of bytes.

        Parameters
        ----------
        bytestream : bytes
            The stream of bytes encoding the monomial.

        Returns
        -------
        numpy.ndarray
            The corresponding monomial in array form.
        """
        array = np.frombuffer(bytestream, dtype=self.np_dtype)
        return array.reshape((-1, self._nr_properties))

    def _to_canonical_memoized(self,
                               array2d: np.ndarray,
                               apply_only_commutations=False) -> np.ndarray:
        """Cached function to convert a monomial to its canonical form.

        It checks whether the input monomial's canonical form has already been
        calculated and stored in the ``InflationSDP.canon_ndarray_from_hash``.
        If not, it calculates it.

        Parameters
        ----------
        array2d : numpy.ndarray
            Moment encoded as a 2D array.
        apply_only_commutations : bool, optional
            If ``True``, skip the removal of projector squares and the test to
            see if the monomial is equal to zero, by default ``False``.

        Returns
        -------
        numpy.ndarray
            Moment in canonical form.
        """
        key = self._from_2dndarray(array2d)
        try:
            return self.canon_ndarray_from_hash[key]
        except KeyError:
            if len(array2d) == 0 or np.array_equiv(array2d, 0):
                self.canon_ndarray_from_hash[key] = array2d
                return array2d
            else:
                # This is a fancy way of just sorting lexicographically.
                new_array2d = to_canonical(array2d,
                                           self._notcomm,
                                           self._lexorder,
                                           commuting=True,
                                           apply_only_commutations=True)
                new_key = self._from_2dndarray(new_array2d)
                self.canon_ndarray_from_hash[key]     = new_array2d
                self.canon_ndarray_from_hash[new_key] = new_array2d
                return new_array2d