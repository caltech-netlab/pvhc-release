import sys, pathlib
file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]) + '/src')

"""
Tests for the 3-phase LinDistFlow model (three_phase_lindistflow.py).
"""
import copy
import inspect

import numpy as np

import three_phase_lindistflow as ldf
from test_data import test_circuit_1, test_circuit_3, test_circuit_4, test_circuit_5

np.set_printoptions(precision=2, linewidth=256)


def _to_yy(circuit):
    """Return a copy of `circuit` with all transformer windings forced to YY."""
    circuit = copy.deepcopy(circuit)
    for e in circuit:
        if e['element_type'] not in {'line', 'YY'}:
            e['element_type'] = 'YY'
    return circuit


# Reference (C0, R, X, C_S) Frobenius norms, computed once from a known-good
# run of three_phase_lindistflow_real_wrapper() on the YY-only reduction of
# each fixture circuit. A change to these values signals either an intentional
# model change (update the reference) or a regression in the LinDistFlow math.
_REFERENCE_NORMS = {
    'test_circuit_1': {
        'node_list': [0, 1, 6, 2, 7, 3, 4, 5],
        'edge_list': [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (0, 6), (1, 7)],
        'norms': (30.347982, 712.460876, 540.233276, 7.141428),
    },
    'test_circuit_3': {
        'node_list': [0, 1, 2, 7, 3],
        'edge_list': [(0, 1), (1, 2), (2, 3), (1, 7)],
        'norms': (10.099504, 118.793938, 95.624268, 4.898980),
    },
}


def test_regression_against_reference_norms():
    """
    Regression test: the real-valued LinDistFlow wrapper's output matrices
    (C0, R, X, C_S) must match previously-recorded Frobenius norms (to within
    float32 tolerance) on fixed test circuits, and produce a deterministic
    node/edge ordering. Catches unintended changes to the core physics model.
    """
    for name, ref in _REFERENCE_NORMS.items():
        circuit = _to_yy(globals()[name])
        C0, R, X, C_S, node_list, edge_list = ldf.three_phase_lindistflow_real_wrapper(circuit)

        assert node_list == ref['node_list'], f"{name}: node_list={node_list}"
        assert edge_list == ref['edge_list'], f"{name}: edge_list={edge_list}"

        norms = (np.linalg.norm(C0), np.linalg.norm(R), np.linalg.norm(X), np.linalg.norm(C_S))
        assert np.allclose(norms, ref['norms'], rtol=1e-4), (
            f"{name}: norms {norms} do not match reference {ref['norms']}"
        )
    print("Passed:", inspect.stack()[0][3])


def test_random_graphs():
    """
    Smoke test: the general (complex-valued) LinDistFlow derivation must run
    without error on circuits mixing line segments with YY/DD/DY/YD
    transformers -- a case the retired wye-only implementation could not handle.
    """
    for test_circuit in [test_circuit_4, test_circuit_5]:
        ldf.three_phase_lindistflow(test_circuit, node_list=list(range(len(test_circuit) + 1)))
    print("Passed:", inspect.stack()[0][3])


if __name__ == "__main__":
    test_regression_against_reference_norms()
    test_random_graphs()
