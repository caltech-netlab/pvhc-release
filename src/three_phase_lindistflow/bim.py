"""Bus injection model: nodal admittance matrix and branch power flows."""

import numpy as np

from .constants import GG, GI, GTGT, GTI, IG, IGT
from .linalg import block2matrix, matrix2block
from .topology import check_network_validity
from .branches import branch_data_to_admittance_matrix, floating_neutrals, validate_line_phase_structure

__all__ = [
    "BIM_IV_model",
    "compute_branch_power_flow",
    "four_parameter_branch_model",
]

_TRANSFORMER_TYPES = ('YY', 'DD', 'DY', 'DY_alt', 'YD', 'YD_alt', 'openY_openY')


# BIM helpers below
def _line_Y_matrix(y_series, y_shunt=0):
    return np.block([
        [y_series, - y_series],
        [-y_series, (y_series + y_shunt)]
    ])


def _floating_neutral_admittance(y_series):
    """Series admittance (3x3) of a bank with a floating wye neutral.

    The winding currents of a floating neutral sum to zero, so the neutral is a node to
    eliminate (Kron reduction): ``y - (y 1)(1^T y) / (1^T y 1)``. The result passes no
    zero-sequence current from either side, i.e. it is zero on ``1``, from left and right.
    Floating the primary neutral, the secondary one or both gives the same: the two
    windings of each unit carry the same current, up to the turns ratio.
    """
    ones = np.ones(3)
    total = ones @ y_series @ ones
    if total == 0:
        raise ValueError("a floating neutral needs a winding with a non-zero admittance")
    return y_series - np.outer(y_series @ ones, ones @ y_series) / total


def _transformer_Y_matrix_PSA(y_series, a_turns_ratio, element_type, y_shunt=0, floating_neutral=(False, False)):
    """
    Computes the 6x6 transformer admittance matrix for a 3-phase transformer.
    Implementation based on Steven's textbook eqution 8.24.
    Supported transformer types: DY, YY

    floating_neutral: whether the (primary, secondary) wye winding has a floating neutral
    (see floating_neutrals); otherwise the wye neutrals are grounded.
    """
    if any(floating_neutral):
        if np.any(y_shunt):
            raise NotImplementedError("a floating neutral with a shunt (magnetizing) admittance")
        y_series = _floating_neutral_admittance(y_series)
    Y_YY = np.block([
        [y_series, -a_turns_ratio * y_series],
        [-a_turns_ratio * y_series, a_turns_ratio ** 2 * (y_series + y_shunt)]
    ])
    if element_type in ("YY", "openY_openY"):
        return Y_YY
    elif element_type == "DY":
        return GTI @ Y_YY @ GI
    elif element_type == "DY_alt":
        return GI @ Y_YY @ GTI
    elif element_type == "YD":
        return IGT @ Y_YY @ IG
    elif element_type == "YD_alt":
        return IG @ Y_YY @ IGT
    elif element_type == "DD":
        return GTGT @ Y_YY @ GG
    else:
        raise ValueError(f"Unknown transformer type: {element_type}")


def _branch_Y_matrix(branch_data):
    """Primitive admittance matrix [[Y_jj, Y_jk], [Y_kj, Y_kk]] of a branch (6x6)."""
    floating_neutral = floating_neutrals(branch_data)  # (raises for a winding without a neutral)
    if branch_data['element_type'] == 'line':
        return _line_Y_matrix(branch_data_to_admittance_matrix(branch_data))
    if branch_data['element_type'] in _TRANSFORMER_TYPES:
        return _transformer_Y_matrix_PSA(
            branch_data_to_admittance_matrix(branch_data),
            branch_data['turns_ratio'],
            branch_data['element_type'],
            floating_neutral=floating_neutral,
        )
    raise ValueError(f"Unknown element type: {branch_data['element_type']}")


def BIM_IV_model(circuit_data, node_list=None):
    """
    Given an electrical network graph, return its admittance matrix Y.
    Only 3-phase transformers and lines are implemented.
    return: np.ndarray, float, 3N x 3N matrix, where N is the number of nodes.
    """
    _, node_list_g, _, _ = check_network_validity(circuit_data)
    node_list = node_list or node_list_g
    N = len(node_list)
    Y = np.zeros((N, N, 3, 3), dtype=np.complex128)
    for branch_data in circuit_data:
        validate_line_phase_structure(branch_data)
        j, k = node_list.index(branch_data["source"]), node_list.index(branch_data["target"])
        component_Y = _branch_Y_matrix(branch_data)
        # This assumes zero neutral current & voltage.
        component_Y = matrix2block(component_Y, 4)[:,:,:3,:3] \
            if component_Y.shape == (8, 8) else matrix2block(component_Y, 3)
        Y[j,j] += component_Y[0, 0]
        Y[k,k] += component_Y[1, 1]
        Y[j,k] += component_Y[0, 1]
        Y[k,j] += component_Y[1, 0]
    Y = block2matrix(Y)
    return Y


def compute_branch_power_flow(circuit_data, node_list, V_nodal):
    """Compute source-end branch complex power flow and current for each branch from nodal voltages."""
    node2idx = {node: idx for idx, node in enumerate(node_list)}
    # Return a full 3x3 complex power matrix per branch (V_source_i * conj(I_source_j)).
    branch_power = np.zeros((len(circuit_data), 3, 3), dtype=complex)
    branch_source_current = np.zeros((len(circuit_data), 3), dtype=complex)

    for e, branch_data in enumerate(circuit_data):
        source_idx = node2idx[branch_data["source"]]
        target_idx = node2idx[branch_data["target"]]
        component_Y = _branch_Y_matrix(branch_data)
        component_Y = matrix2block(component_Y, 4)[:,:,:3,:3] if component_Y.shape == (8, 8) else matrix2block(component_Y, 3)
        source_current = component_Y[0, 0] @ V_nodal[source_idx] + component_Y[0, 1] @ V_nodal[target_idx]
        branch_source_current[e, :] = source_current
        # Full complex power matrix S_ij = V_i * conj(I_j) so the diagonal yields
        # per-phase apparent power V_i * conj(I_i).
        branch_power[e, :, :] = (V_nodal[source_idx, :, None] * np.conj(source_current[None, :]))

    return branch_power, branch_source_current


def four_parameter_branch_model(branch_data):
    """
    Returns
    -------
    y_jk_s: np.ndarray, shape=(3, 3), the series admittance matrix in the (j,k) direction.
    y_kj_s: np.ndarray, shape=(3, 3), the series admittance matrix in the (k,j) direction.
    y_jk_m: np.ndarray, shape=(3, 3), the shunt admittance matrix at node j.
    y_kj_m: np.ndarray, shape=(3, 3), the shunt admittance matrix at node k.
    """
    validate_line_phase_structure(branch_data)
    component_Y = _branch_Y_matrix(branch_data)
    component_Y = matrix2block(component_Y, 4)[:,:,:3,:3] \
        if component_Y.shape == (8, 8) else matrix2block(component_Y, 3)
    y_jk_s = - component_Y[0, 1]
    y_kj_s = - component_Y[1, 0]
    y_jk_m = component_Y[0, 0] + component_Y[0, 1]
    y_kj_m = component_Y[1, 1] + component_Y[1, 0]
    return y_jk_s, y_kj_s, y_jk_m, y_kj_m
