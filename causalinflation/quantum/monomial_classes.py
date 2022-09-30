import numpy as np

from collections import Counter
from functools import total_ordering
from typing import Tuple, Dict

from .general_tools import is_physical
from .monomial_utils import (compute_marginal,
                             name_from_atomic_names,
                             symbol_from_atomic_name,
                             symbol_prod)
from .fast_npa import mon_is_zero

@total_ordering
class InternalAtomicMonomial(object):
    __slots__ = ['as_ndarray',
                 'do_conditional',
                 'is_one',
                 'is_zero',
                 'knowable_q',
                 'n_ops',
                 'op_length',
                 'rectified_ndarray',
                 'sdp'
                 ]

    def __init__(self, inflation_sdp_instance, array2d: np.ndarray):
        """
        This uses methods from the InflationSDP instance, and so must be constructed with that passed as first argument.
        """
        self.sdp = inflation_sdp_instance
        self.as_ndarray = np.asarray(array2d, dtype=self.sdp.np_dtype)
        self.n_ops, self.op_length = self.as_ndarray.shape
        assert self.op_length == self.sdp._nr_properties, "We insist on well-formed 2d arrays as input to AtomicMonomial."
        self.is_zero = mon_is_zero(self.as_ndarray)
        self.is_one = (self.n_ops == 0)
        self.knowable_q = self.is_zero or self.is_one or self.sdp.atomic_knowable_q(self.as_ndarray)
        self.do_conditional = False
        if self.knowable_q:
            self.rectified_ndarray = np.asarray(self.sdp.rectify_fake_setting(np.take(self.as_ndarray, [0, -2, -1], axis=1)), dtype=int)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        for attr in self.__slots__:
            try:
                result.__setattr__(attr, self.__getattribute__(attr))
            except AttributeError:
                pass
        return result

    @property
    def dagger(self):
        conjugate_ndarray = self.sdp.inflation_aware_to_ndarray_conjugate_representative(self.as_ndarray)
        conjugate_signature = self.sdp.from_2dndarray(conjugate_ndarray)
        if conjugate_signature != self.signature:
            dagger = self.__copy__()
            dagger.as_ndarray = conjugate_ndarray
            return dagger
        else:
            return self

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
        """Return the name of the Monomial"""
        return self.__str__()

    def __str__(self):
        """Return the name of the Monomial"""
        return self.name

    @property
    def is_physical(self):
        return self.knowable_q or is_physical(self.as_ndarray)

    @property
    def signature(self):
        return self.sdp.from_2dndarray(self.as_ndarray)

    @property
    def name(self):
        if self.is_one:
            return '1'
        elif self.is_zero:
            return '0'
        elif self.knowable_q:
            party_indices = self.rectified_ndarray[:, 0] - 1
            parties = np.take(self.sdp.names, party_indices.tolist())  # Convention in numpy monomial format is first party = 1
            inputs = [str(input) for input in self.rectified_ndarray[:, -2].tolist()]
            outputs = [str(output) for output in self.rectified_ndarray[:, -1].tolist()]
            p_divider = '' if all(len(p) == 1 for p in parties) else ','
            # We will probably never have more than 1 digit cardinalities, but who knows...
            i_divider = '' if all(len(i) == 1 for i in inputs) else ','
            o_divider = '' if all(len(o) == 1 for o in outputs) else ','
            if self.do_conditional:
                return ('p' + p_divider.join(parties) +
                        '(' + o_divider.join(outputs) + ' do: ' + i_divider.join(inputs) + ')')
            else:
                return ('p' + p_divider.join(parties) +
                        '(' + o_divider.join(outputs) + '|' + i_divider.join(inputs) + ')')
        else:
            operators_as_strings = []
            for op in self.as_ndarray.tolist():  # this handles the UNKNOWN factors.
                operators_as_strings.append('_'.join([self.sdp.names[op[0] - 1]]  # party idx
                                                     + [str(i) for i in op[1:]]))
            return '<' + ' '.join(operators_as_strings) + '>'

    @property
    def symbol(self):
        return symbol_from_atomic_name(self.name)

    def compute_marginal(self, prob_array):
        if self.is_zero:
            return 0.
        else:
            return compute_marginal(prob_array=prob_array,
                                    atom=self.rectified_ndarray)


class CompoundMonomial(object):
    __slots__ = ['factors_as_atomic_monomials',
                 'is_atomic',
                 'is_zero',
                 'is_one',
                 'nof_factors',
                 'knowable_factors',
                 'unknowable_factors',
                 'nof_knowable_factors',
                 'nof_unknowable_factors',
                 'knowability_status',
                 'knowable_q',
                 'idx',
                 'mask_matrix'
                 ]

    def __init__(self, tuple_of_atomic_monomials: Tuple[InternalAtomicMonomial]):
        """
        This class is designed to categorize monomials into known, semiknown, unknown, etc.
        It also computes names for expectation values, and provides the ability to compare (in)equivalence.
        """
        default_factors = tuple(sorted(tuple_of_atomic_monomials))
        conjugate_factors = tuple(sorted(factor.dagger for factor in tuple_of_atomic_monomials))
        self.factors_as_atomic_monomials = min(default_factors, conjugate_factors)
        self.nof_factors = len(self.factors_as_atomic_monomials)
        self.is_atomic = (self.nof_factors <= 1)
        self.knowable_q = all(factor.knowable_q for factor in self.factors_as_atomic_monomials)
        self.knowable_factors = tuple(factor for factor in self.factors_as_atomic_monomials if factor.knowable_q)
        self.unknowable_factors = tuple(factor for factor in self.factors_as_atomic_monomials if not factor.knowable_q)
        self.nof_knowable_factors = len(self.knowable_factors)
        self.nof_unknowable_factors = len(self.unknowable_factors)
        if self.nof_unknowable_factors == 0:
            self.knowability_status = 'Yes'
        elif self.nof_unknowable_factors == self.nof_factors:
            self.knowability_status = 'No'
        else:
            self.knowability_status = 'Semi'
        self.is_zero = any(factor.is_zero for factor in self.factors_as_atomic_monomials)
        self.is_one = all(factor.is_one for factor in self.factors_as_atomic_monomials) or (self.nof_factors == 0)

    @property
    def n_ops(self):
        return sum(factor.n_ops for factor in self.factors_as_atomic_monomials)

    @property
    def is_physical(self):
        return all(factor.is_physical for factor in self.factors_as_atomic_monomials)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def as_counter(self):
        return Counter(self.factors_as_atomic_monomials)

    def __len__(self):
        return self.nof_factors

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_counter == other.as_counter
        elif isinstance(other, InternalAtomicMonomial):
            return (self.nof_factors == 1) and other.__eq__(self.factors_as_atomic_monomials[0])
        else:
            assert isinstance(other, self.__class__), f"Expected {self.__class__}, recieved {other} of {type(other)}{list(map(type, other))}."
            return False

    @property
    def signature(self):
        return tuple(sorted(self.factors_as_atomic_monomials))

    def __hash__(self):
        return hash(self.signature)

    def evaluate_given_valuation_of_knowable_part(self, valuation_of_knowable_part, use_lpi_constraints=True):
        actually_known_factors = np.logical_not(np.isnan(valuation_of_knowable_part))
        known_value = float(np.prod(np.compress(
            actually_known_factors,
            valuation_of_knowable_part)))
        unknown_factors = [factor for factor, known in
                           zip(self.knowable_factors,
                               actually_known_factors)
                           if not known]
        unknown_factors.extend(self.unknowable_factors)
        unknown_len = len(unknown_factors)
        if unknown_len == 0 or (np.isclose(known_value, 0) and use_lpi_constraints):
            known_status = 'Yes'
        elif unknown_len == self.nof_factors or (not use_lpi_constraints):
            known_status = 'No'
        else:
            known_status = 'Semi'
        return known_value, unknown_factors, known_status


    def evaluate_given_atomic_monomials_dict(self, dict_of_known_atomic_monomials: Dict[InternalAtomicMonomial, float], use_lpi_constraints=True):
        "Yields both a numeric value and a CompoundMonomial corresponding to the unknown part."
        known_value = 1.
        unknown_factors_counter = Counter()
        for factor, power in self.as_counter.items():
            temp_value = dict_of_known_atomic_monomials.get(factor, np.nan)
            if np.isnan(temp_value):
                unknown_factors_counter[factor] = power
            else:
                known_value *= (temp_value ** power)
        unknown_factors = list(unknown_factors_counter.elements())
        unknown_len = len(unknown_factors)
        if unknown_len == 0 or (np.isclose(known_value, 0) and use_lpi_constraints):
            known_status = 'Yes'
        elif unknown_len == self.nof_factors or (not use_lpi_constraints):
            known_status = 'No'
        else:
            known_status = 'Semi'
        return known_value, unknown_factors, known_status

    @property
    def _names_of_factors(self):
        return [factor.name for factor in self.factors_as_atomic_monomials]

    @property
    def _symbols_of_factors(self):
        return [factor.symbol for factor in self.factors_as_atomic_monomials]

    @property
    def name(self):
        return name_from_atomic_names(self._names_of_factors)

    @property
    def symbol(self):
        return symbol_prod(self._symbols_of_factors)

    def compute_marginal(self, prob_array):
        assert self.knowable_q, "Can't compute marginals of unknowable probabilities."
        v = 1.
        for factor, power in self.as_counter.items():
            v *= (factor.compute_marginal(prob_array) ** power)
        return v

    def attach_idx_to_mon(self, idx: int):
        if idx >= 0:
            self.idx = idx
