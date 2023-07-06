"""
This file contains classes for defining the monomials inside a moment matrix.

@authors: Emanuel-Cristian Boghiu, Elie Wolfe, Alejandro Pozas-Kerstjens
"""
import numpy as np

from collections import Counter
from functools import total_ordering
from numbers import Real
from typing import Dict, List, Tuple

from ..sdp.monomial_utils import (compute_marginal,
                                  name_from_atom_names,
                                  symbol_from_atom_name,
                                  symbol_prod)
from functools import reduce
# import methodtools


@total_ordering
class InternalAtomicMonomial(object):
    __slots__ = ["as_int_vec",
                 "as_bool_vec",
                 "as_2d_array",
                 "is_one",
                 "is_knowable",
                 "n_operators",
                 "rectified_ndarray",
                 "lp",
                 "name",
                 "symbol",
                 "signature"
                 ]

    def __init__(self, inflation_lp_instance, as_1d_vec: np.ndarray):
        r"""This class models a moment
        :math:`\langle Op_1 Op_2\dots Op_n\rangle` on the inflated problem,
        which cannot be decomposed into products of other moments. It is used
        as a building block for the ``CompoundMonomial`` class. It is
        initialized with a 1D array representing a moment and an instance of
        ``InflationLP``, used for methods that depend on the scenario.


        Parameters
        ----------
        inflation_lp_instance : InflationLP
            An instance of the ``InflationLP`` class. It is used to access
            methods specific to the inflation problem. E.g., when instantiating
            an internal atomic moment, the ``InflationSDP`` instance is used to
            check if it already contains such moment.
        array2d : numpy.ndarray
            A moment :math:`\langle Op_1Op_2\dots Op_n\rangle` encoded as a 2D
            array.
        """
        self.lp = inflation_lp_instance
        if as_1d_vec.dtype == bool:
            self.as_bool_vec = as_1d_vec
            self.as_int_vec = np.flatnonzero(self.as_bool_vec).astype(int)
        else:
            self.as_int_vec = as_1d_vec.astype(int)
            self.as_bool_vec = inflation_lp_instance.blank_bool_vec.copy()
            self.as_bool_vec[self.as_int_vec] = True

        self.as_2d_array = inflation_lp_instance._lexorder[self.as_int_vec]
        self.n_operators = len(self.as_int_vec)
        self.is_one      = (self.n_operators == 0)
        self.is_knowable = (self.is_one or
                            self.lp._atomic_knowable_q(self.as_2d_array))
        # Save also array with the original setting, not just the effective one
        if self.is_knowable:
            self.rectified_ndarray = np.asarray(
                inflation_lp_instance.rectify_fake_setting(np.take(self.as_2d_array,
                                                     [0, -2, -1],
                                                     axis=1)),
                dtype=int)

        self.name = self._name
        self.signature = self._signature
        self.symbol = self._symbol

    def __copy__(self):
        """Make a copy of the Monomial"""
        cls = self.__class__
        result = cls.__new__(cls)
        for attr in self.__slots__:
            try:
                result.__setattr__(attr, self.__getattribute__(attr))
            except AttributeError:
                pass
        return result

    def __eq__(self, other):
        """Whether the Monomial is equal to the ``other`` Monomial."""
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        """Return the hash of the Monomial."""
        return hash(self.signature)

    def __lt__(self, other):
        """Whether the Monomial is lexicographically smaller than the ``other``
        Monomial.
        """
        return self.signature < other.signature

    def __repr__(self):
        """Return the name of the Monomial."""
        return self.__str__()

    def __str__(self):
        """Return the name of the Monomial."""
        return self.name

    @property
    def _name(self):
        """A string representing the monomial. In case of knowable monomials,
        it is of the form ``p(outputs|inputs)``. Otherwise it represents the
        expectation value of the monomial with bracket notation.
        """
        if self.is_one:
            return "1"
        elif self.is_knowable:
            # Use notation p(outputs|settings)
            # Convention in numpy monomial format is first party = 1
            party_indices = self.rectified_ndarray[:, 0] - 1
            parties       = np.take(self.lp.names, party_indices.tolist())
            inputs  = [str(input) for input in self.rectified_ndarray[:, -2]]
            outputs = [str(output) for output in self.rectified_ndarray[:, -1]]
            p_divider = ""
            # We will probably never have more than 1 digit cardinalities,
            # but who knows...
            i_divider = "" if all(len(i) == 1 for i in inputs) else ","
            o_divider = "" if all(len(o) == 1 for o in outputs) else ","
            return ("p" + p_divider.join(parties) +
                    "(" + o_divider.join(outputs) +
                    "|" + i_divider.join(inputs) + ")")
        else:
            # Use expectation value notation
            operators = []
            for op in self.as_2d_array:
                operators.append("_".join([self.lp.names[op[0] - 1]]
                                          + [str(i) for i in op[1:]]))
            return "<" + " ".join(operators) + ">"

    @property
    def _signature(self):
        # return self.as_1d_int_vec.tobytes() #FOR QUANTUM OR NONCOMMUTING CASE!!
        return self.as_bool_vec.tobytes()
        return tuple(sorted())



    @property
    def _symbol(self):
        """Return a sympy Symbol representing the monomial."""
        return symbol_from_atom_name(self.name)

    def compute_marginal(self, prob_array: np.ndarray) -> float:
        """Given a probability distribution, compute the numerical value of
        the Monomial.

        Parameters
        ----------
        prob_array : numpy.ndarray
            The target probability distribution. The dimensions of this array
            are (outcomes_per_party, settings_per_party), where the settings
            are explicitly 1 is there is only one measurement performed.

        Returns
        -------
        float
            The value of the corresponding probability (which can be a marginal
            involving only a few parties)
        """
        return compute_marginal(prob_array, self.rectified_ndarray)


class CompoundMonomial(object):
    __slots__ = ["as_counter",
                 "factors",
                 "idx",
                 "is_atomic",
                 "is_knowable",
                 "is_one",
                 "knowability_status",
                 "knowable_factors",
                 "mask_matrix",
                 "n_factors",
                 "n_knowable_factors",
                 "n_operators",
                 "n_unknowable_factors",
                 "name",
                 "signature",
                 "symbol",
                 "unknowable_factors",
                 "as_bool_vec",
                 "as_int_vec"
                 # "_names_of_factors",
                 # "_symbols_of_factors"
                 ]

    def __init__(self, monomials: Tuple[InternalAtomicMonomial]):
        r"""This class models moments :math:`\langle Op_1 Op_2\dots Op_n\rangle
        =\langle Op_i\dots\rangle\langle Op_{i'}\dots\rangle` on the inflated
        problem that are products of other moments. It is built from a tuple of
        instances of the ``InternalAtomicMonomial`` class.

        At initialisation, a moment is classified into knowable, semi-knowable
        or unknowable based on the knowability of each of the atomic moments
        (which in turn is determined through methods of the
        ``InternalAtomicMonomial`` class). This class also computes names for
        the moment, provides the ability to compare (in)equivalence, and to
        assign numerical values to a moment given a probability distribution.

        Parameters
        ----------
        monomials : tuple of InternalAtomicMonomial
            The atomic moments that make up the compound moment.
        """
        self.factors       = tuple(sorted(monomials))
        self.n_factors     = len(self.factors)
        self.n_operators   = sum(factor.n_operators for factor in self.factors)
        self.is_atomic     = (self.n_factors <= 1)
        self.is_knowable   = all(factor.is_knowable for factor in self.factors)

        knowable_factors   = []
        unknowable_factors = []
        for factor in self.factors:
            if factor.is_knowable:
                knowable_factors.append(factor)
            else:
                unknowable_factors.append(factor)
        self.knowable_factors     = tuple(knowable_factors)
        self.unknowable_factors   = tuple(unknowable_factors)
        self.n_knowable_factors   = len(self.knowable_factors)
        self.n_unknowable_factors = len(self.unknowable_factors)
        if self.n_unknowable_factors == 0:
            self.knowability_status = "Knowable"
        elif self.n_unknowable_factors == self.n_factors:
            self.knowability_status = "Unknowable"
        else:
            self.knowability_status = "Semi"
        self.is_one  = (all(factor.is_one for factor in self.factors)
                        or (self.n_factors == 0))
        self.as_counter  = Counter(self.factors)


        if self.is_one:
            self.as_bool_vec = np.zeros(0, dtype=bool)
            self.as_int_vec = np.zeros(0, dtype=int)
        elif self.is_atomic:
            self.as_bool_vec = self.factors[0].as_bool_vec
            self.as_int_vec = self.factors[0].as_int_vec
        else:
            self.as_bool_vec = reduce(np.bitwise_or,
                                      (factor.as_bool_vec for factor in self.factors))
            self.as_int_vec = np.hstack([factor.as_int_vec for factor in self.factors]).astype(int)

        self.name        = name_from_atom_names(self._names_of_factors)
        self.symbol      = symbol_prod(self._symbols_of_factors)
        self.signature   = self.factors

    def __eq__(self, other):
        """Whether the Monomial is equal to the ``other`` Monomial."""
        if isinstance(other, self.__class__):
            return self.as_counter == other.as_counter
        elif isinstance(other, InternalAtomicMonomial):
            return (self.n_factors == 1) and other.__eq__(self.factors[0])
        else:
            assert isinstance(other, self.__class__), \
                (f"Expected object of class {self.__class__}, received {other}"
                 + f" of class {type(other)}{list(map(type, other))}.")
            return False

    def __hash__(self):
        """Return the hash of the Monomial."""
        return hash(self.signature)

    def __len__(self):
        """Return the number of AtomicMonomials in the CompoundMonomial."""
        return self.n_factors

    def __repr__(self):
        """Return the name of the Monomial."""
        return self.__str__()

    def __str__(self):
        """Return the name of the Monomial."""
        return self.name

    @property
    def _names_of_factors(self):
        """Return the names of each of the factors in the Monomial."""
        return [factor.name for factor in self.factors]

    @property
    def _symbols_of_factors(self):
        """Generate a sympy Symbol per factor in the Monomial."""
        return [factor.symbol for factor in self.factors]

    def attach_idx(self, idx: int):
        """Assign an index to the Monomial. This is used when generating the
        monomials in a scenario, and identifying them with integers."""
        if idx >= 0:
            self.idx = idx

    def compute_marginal(self, prob_array: np.ndarray) -> float:
        """Given a probability array, evaluate all the knowable atomic moment
        factors in the monomial and return the product of the resulting values.

        The whole moment needs to be knowable (``self.is_knowable = True``),
        else this method will raise an error.

        Parameters
        ----------
        prob_array : np.ndarray
            A conditional probability distribution over the non-inflated
            scenario with dimensions ``(da,db,...,dx,dy,...)`` (i.e., first
            outcomes and then settings for the parties).

        See also
        --------
        CompoundMonomial.evaluate : This function does not rely on the
            knowability of the monomial.
        """
        assert self.is_knowable, ("Only marginals of knowable monomials can " +
                                  f"be computed, and {self} is not knowable.")
        value = 1.
        for factor, power in self.as_counter.items():
            value *= (factor.compute_marginal(prob_array) ** power)
        return value

    def evaluate(self,
                 known_monomials: Dict[InternalAtomicMonomial, float],
                 use_lpi_constraints=True) -> Tuple[float, List, str]:
        """Given a dictionary of values for known atomic monomials,
        substitute all factors of the compound moment that are specified in
        the dictionary with their values and return the product of the values
        and a remainder compound moment made of the remaining factors.

        Parameters
        ----------
        known_monomials : Dict[InternalAtomicMonomial, float]
            A dictionary of known atomic monomials and their values.
        use_lpi_constraints : bool, optional
            Whether compound moments whose factors are partially present in
            ``known_monomials`` are labelled as ``"Semi"`` or ``"Unknowable"``.
            If ``True``, they are labelled as ``"Semi"``. By default ``True``.

        Returns
        -------
        Tuple[float, List, str]
            A tuple where the first element is the value of the knowable parts
            of the compound moment, the second element is a compound monomial
            of all the remaining factors not present in ``known_monomials`` and
            the third element is a string describing whether all factors are
            present in ``known_monomials`` (``"Known"``), some are
            (``"Semi"``), or none are (``"Unknown"``).
        """
        known_value     = 1.
        unknown_counter = Counter()
        for factor, power in self.as_counter.items():
            try:
                known_value *= (known_monomials[factor] ** power)
            except KeyError:
                unknown_counter[factor] = power
        # unknown_factors = sorted(unknown_counter.elements(),
        #                          key=lambda factor: factor.as_int_vec.tobytes())
        unknown_factors = list(unknown_counter.elements())
        if (len(unknown_factors) == 0):
            known_status = "Known"
        elif ((len(unknown_factors) == self.n_factors)
              or (not use_lpi_constraints)):
            known_status = "Unknown"
        else:
            known_status = "Semi"
            if use_lpi_constraints and isinstance(known_value, Real):
                if np.isclose(known_value, 0):
                    known_status = "Known"
        return known_value, unknown_factors, known_status
