"""Per-branch impedance handling and the generalized component model."""

import logging
import numpy as np
import warnings

from .constants import A_j_9x9_all, A_k_9x9_all, F, Gamma, I_23, I_3, I_9, Q_delta, STD_DELTA_ORDER, T_delta, U_delta

logger = logging.getLogger(__name__)

__all__ = [
    "convert_series_impedance_3x3",
    "validate_line_phase_structure",
    "to_single_phase_series_impedance",
    "branch_data_to_admittance_matrix",
    "floating_neutrals",
    "component_model",
    "build_permutation_matrix_open_split",
]

# Winding connections (primary, secondary) of each transformer element type. 'split' is the
# center-tapped (120/240 V) secondary of a split-phase bank, grounded at the tap.
TRANSFORMER_WINDINGS = {
    'YY': ('wye', 'wye'),
    'openY_openY': ('wye', 'wye'),
    'DY': ('delta', 'wye'),
    'DY_alt': ('delta', 'wye'),
    'YD': ('wye', 'delta'),
    'YD_alt': ('wye', 'delta'),
    'DD': ('delta', 'delta'),
    'openY_openD': ('wye', 'delta'),
    'openD_openY': ('delta', 'wye'),
    'openD_openD': ('delta', 'delta'),
    'openY_openD_split': ('wye', 'split'),
    'openD_openD_split': ('delta', 'split'),
}


def _winding_phases(branch_data, side):
    """Phases (1-3) of one winding of a transformer, from its per-winding ``phases``, or None.

    A wye winding has one per phase listed. None when ``phases`` is not given per winding
    (as two sequences), or lists no phase (a bare bus: all three).
    """
    phases = branch_data.get('phases')
    if not phases or len(phases) != 2 or not all(isinstance(p, (list, tuple)) for p in phases):
        return None
    return {p for p in phases[side] if p in (1, 2, 3)} or None


def floating_neutrals(branch_data):
    """Whether the (primary, secondary) winding of a branch is a wye with a floating neutral.

    Read from ``branch_data['floating_neutral']``; without it, every wye neutral is grounded.
    Only a wye winding of a transformer has a neutral that can float, so flagging a delta
    winding or a line raises. So does a floating neutral on fewer than two windings, which
    carry no current, or on a transformer whose series impedance has windings its wye winding
    does not (as the parser gives a single transformer on fewer than three phases): the
    current of each winding returns through the others, so every one of them must exist.
    """
    flags = tuple(bool(flag) for flag in branch_data.get('floating_neutral', (False, False)))
    if len(flags) != 2:
        raise ValueError(f"floating_neutral must have one entry per winding (two), got {flags}")
    if not any(flags):
        return flags
    name = branch_data.get('element_name') or f"{branch_data.get('source')}->{branch_data.get('target')}"
    windings = TRANSFORMER_WINDINGS.get(branch_data['element_type'])
    if windings is None:
        raise ValueError(
            f"Branch {name} ({branch_data['element_type']}) has floating_neutral set, but it is not "
            "a transformer with a wye winding, and so has no neutral")
    z_ij = convert_series_impedance_3x3(branch_data['series_impedance'])
    present = {p + 1 for p in range(3) if z_ij[p, p] != 0}
    for side, (flag, winding) in enumerate(zip(flags, windings)):
        side_name = ('primary', 'secondary')[side]
        if not flag:
            continue
        if winding == 'split':
            raise NotImplementedError(f"Branch {name}: a floating center tap ({side_name} side) is not supported")
        if winding != 'wye':
            raise ValueError(
                f"Branch {name} ({branch_data['element_type']}) has floating_neutral set on its "
                f"{side_name} side, which is not a wye winding and so has no neutral")
        wye_phases = _winding_phases(branch_data, side)
        if wye_phases is not None and present - wye_phases:
            raise ValueError(
                f"Branch {name}: its {side_name} wye winding, on a floating neutral, is on phases "
                f"{sorted(wye_phases)}, but its series impedance has windings on phases {sorted(present)} "
                "as well, which the neutral would return current through (a single transformer on "
                "fewer phases is given fictitious windings on the others)")
        if wye_phases is not None and wye_phases - present:
            raise ValueError(
                f"Branch {name}: its {side_name} wye winding, on a floating neutral, has no series "
                f"impedance on phases {sorted(wye_phases - present)} (a transformer rated 0 kVA is given "
                "none; give it one)")
        if len(present) < 2:
            raise ValueError(f"Branch {name}: a floating neutral on a single winding gives it no return path")
    return flags


def convert_series_impedance_3x3(series_z):
    if type(series_z) in (float, complex):
        return series_z * np.eye(3)
    else:
        z_ij = np.array(series_z)
        if z_ij.shape == (3,):
            return np.diag(z_ij)
        elif z_ij.shape == (3,3):
            return z_ij
        elif z_ij.shape == (9):
            return z_ij.reshape(3,3)
        else:
            raise ValueError(f"Invalid series impedance shape: {z_ij.shape}\n{z_ij}")


def validate_line_phase_structure(branch_data):
    """Validate that a line branch's impedance mask matches its declared phases."""
    if branch_data.get('element_type') != 'line':
        return
    phases = tuple(p for p in branch_data.get('phases', ()) if p in (1, 2, 3))
    if not phases:
        logger.warning("Line %s has no declared phases. Check skipped.", branch_data)
        return 
    z_ij = convert_series_impedance_3x3(branch_data['series_impedance'])
    active_phases = tuple(
        idx + 1
        for idx in range(3)
        if not np.allclose(z_ij[idx, idx], 0)
    )
    # Compare as sets: the declared phases keep the DSS node order (e.g. (3, 2) for
    # "Bus1=632.3.2" in the IEEE 13-node feeder), while z_ij is indexed by phase.
    if active_phases != tuple(sorted(phases)):
        branch_name = branch_data.get('element_name', f"{branch_data.get('source', '?')}->{branch_data.get('target', '?')}")
        raise ValueError(
            f"Line phase mask mismatch for branch {branch_name}: declared phases={phases}, "
            f"active phases={active_phases}, series_impedance={z_ij}"
        )


def to_single_phase_series_impedance(branch_data):
    """Return the equivalent per-phase scalar impedance for a line branch."""
    z_ij = convert_series_impedance_3x3(branch_data['series_impedance'])
    phases = tuple(p for p in branch_data.get('phases', ()) if p in (1, 2, 3))
    if not phases:
        phases = tuple(idx + 1 for idx in range(3) if not np.isclose(z_ij[idx, idx], 0))
    # For a three-phase branch component, take its positive-sequence component
    if len(phases) == 3:
        return complex((F.conj().T @ z_ij @ F)[1, 1])
    # For one- and two-phase components, average over active phases
    active = np.array([z_ij[p - 1, p - 1] for p in phases], dtype=complex)
    if active.size == 0:
        return 0.0 + 0.0j
    if active.size == 1:
        return complex(active[0])
    return complex(np.mean(active))


def branch_data_to_admittance_matrix(branch_data):
    """Return a 3x3 admittance matrix for a line branch, preserving missing phases as zeros."""
    z_ij = convert_series_impedance_3x3(branch_data['series_impedance'])
    if not np.any(z_ij):
        # Inverting a zero (ideal) impedance would leave the branch out, i.e. make it an open
        # circuit, so refuse it rather than disconnect everything below it.
        name = branch_data.get('element_name') or f"{branch_data.get('source')}->{branch_data.get('target')}"
        raise ValueError(
            f"Branch {name} has zero series impedance, which an admittance model cannot represent "
            "(it would be an open circuit); give it a small impedance instead."
        )
    phases = tuple(p for p in branch_data.get('phases', ()) if p in (1, 2, 3))
    if phases:
        active_idx = [p - 1 for p in phases]
    else:
        # Transformers: the windings present are those with an impedance (an open bank has a
        # zero on its missing one). However small, a non-zero one is a winding.
        active_idx = [idx for idx in range(3) if z_ij[idx, idx] != 0]
    y_ij = np.zeros((3, 3), dtype=complex)
    if not active_idx:
        return y_ij
    # It creates a zero 3x3 admittance matrix, then inverts only the active
    # submatrix and writes that inverse back into the active phase block.
    z_active = z_ij[np.ix_(active_idx, active_idx)]
    # Robust inversion: prefer direct inverse, but handle singular/ill-conditioned
    # matrices by regularization or pseudoinverse to avoid enormous admittance
    # entries that destabilize downstream calculations.
    try:
        inv_z = np.linalg.inv(z_active)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"Singular series impedance for active phases {active_idx}; using pseudoinverse",
            RuntimeWarning,
        )
        inv_z = np.linalg.pinv(z_active)
    else:
        # Check condition number and regularize if poorly conditioned
        try:
            cond = np.linalg.cond(z_active)
        except Exception:
            cond = np.inf
        if cond > 1e8:
            # regularization scale: relative to diagonal magnitude, with absolute floor
            diag_max = np.max(np.abs(np.diag(z_active)))
            eps = max(diag_max * 1e-8, 1e-12)
            warnings.warn(
                f"High condition number {cond:.3e} for z_active; regularizing with eps={eps:.3e}",
                RuntimeWarning,
            )
            z_reg = z_active + eps * np.eye(z_active.shape[0], dtype=z_active.dtype)
            inv_z = np.linalg.inv(z_reg)

    y_ij[np.ix_(active_idx, active_idx)] = inv_z
    if np.any(np.abs(y_ij) > 1e6):
        logger.warning("Large admittance entries detected in branch\n%s\n%s", branch_data, np.abs(y_ij))

    return y_ij


def component_model(circuit_data):
    """
    For each branch, returns matrices (A_j, A_k, B, B_H) such that:
        A_j @ vec(v_j) - a_jk^2 * A_k @ vec(v_k) = B @ vec(S_jk) + B_H @ vec(S_jk^H)
    Note that this is slightly different form the paper, where we take the 
        pseudoinverse of A_k and multiply both sides by it.

    :param circuit_data: list of dicts, each dict is branch data, e.g.
        {
            'element_type': 'DY',
            'source': 'bus_1',
            'target': 'bus_2',
            'series_impedance': np.array with shape = (3, 3),
        }
    :param z_ij: np.array, shape=(3, 3), series impedance matrix
    :return: 
        np.ndarray A_j, A_k, B, B_H, shape=(N,9,9)
        Note that the arrays are indexed by circuit_data order.
    """
    N = len(circuit_data)
    A_j_9x9, A_k_9x9, B_9x9, B_H_9x9 = [None]*N, [None]*N, [None]*N, [None]*N
    for e, branch_data in enumerate(circuit_data):
        # Standardize series impedance to 3x3 matrix
        if 'series_impedance' in branch_data:
            z_ij = convert_series_impedance_3x3(branch_data['series_impedance'])
        else:
            z_ij = np.zeros((3, 3), dtype=np.complex128)
            name = branch_data['element_name'] if 'element_name' in branch_data \
                else f"{branch_data['source']} -> {branch_data['target']}"
            logger.warning("Branch %s has no series impedance. Using zero matrix.", name)
        # Generalized branch model
        if branch_data['element_type'] in ['line', 'YY', 'DY', 'YD', 'DD', 'DY_alt', 'openY_openY']:
            A_j_9x9[e] = A_j_9x9_all[branch_data['element_type']]
            A_k_9x9[e] = A_k_9x9_all[branch_data['element_type']]
            B_9x9[e] = np.kron(z_ij.conj(), I_3)
            B_H_9x9[e] = np.kron(I_3, z_ij)
        # TODO: The following two components have not been verified. Write test cases. In particular, winding pairing matching needs to be re-designed.
        elif branch_data['element_type'] == 'openY_openD_split':
            wp = list(branch_data['winding_pairing'])
            assert len(wp) == 2, wp
            # Split phase must be present for the derivation of openY_openD transformer (U_delta matrix)
            assert branch_data['split_phase'] in wp, (branch_data['split_phase'], wp)
            permutation_pri, permutation_sec = build_permutation_matrix_open_split(branch_data)
            p_pri_kron = np.kron(permutation_pri, permutation_pri)
            p_sec_kron = np.kron(permutation_sec, permutation_sec)
            # Instantiate the matrices for canonical pairing
            z_ij = z_ij[:2, :2]
            A_j_9x9[e] = np.kron(U_delta, U_delta) @ np.kron(I_23, I_23)
            A_k_9x9[e] = I_9
            B_9x9[e] = np.kron((U_delta @ z_ij.conj()), U_delta) @ np.kron(I_23, I_23)
            B_H_9x9[e] = np.kron(U_delta, (U_delta @ z_ij)) @ np.kron(I_23, I_23)
            # Apply permutation to get the actual pairing
            A_j_9x9[e] = p_sec_kron @ A_j_9x9[e] @ p_pri_kron
            A_k_9x9[e] = p_sec_kron @ A_k_9x9[e] @ p_pri_kron
            B_9x9[e] = p_sec_kron @ B_9x9[e] @ p_pri_kron
            B_H_9x9[e] = p_sec_kron @ B_H_9x9[e] @ p_pri_kron
        elif branch_data['element_type'] in {'openD_openD', 'openD_openD_split'}:
            wp = list(branch_data['winding_pairing'])
            assert len(wp) == 2, wp
            if ('split_phase' in branch_data) and branch_data['split_phase']:
                assert branch_data['split_phase'] in wp, (branch_data['split_phase'], wp)
            permutation_pri, permutation_sec = build_permutation_matrix_open_split(branch_data)
            p_pri_kron = np.kron(permutation_pri, permutation_pri)
            p_sec_kron = np.kron(permutation_sec, permutation_sec)
            # Instantiate the matrices for canonical pairing
            z_ij = z_ij[:2, :2]
            A_j_9x9[e] = I_9
            A_k_9x9[e] = I_9
            B_9x9[e] = 1/9 * np.kron(Gamma.T @ T_delta, Gamma.T @ T_delta) \
                @ np.kron(z_ij.conj() @ Q_delta, Gamma[:2])
            B_H_9x9[e] = 1/9 * np.kron(Gamma.T @ T_delta, Gamma.T @ T_delta) \
                @ np.kron(Gamma[:2], z_ij @ Q_delta)
            # Apply permutation to get the actual pairing
            A_j_9x9[e] = p_sec_kron @ A_j_9x9[e] @ p_pri_kron
            A_k_9x9[e] = p_sec_kron @ A_k_9x9[e] @ p_pri_kron
            B_9x9[e] = p_sec_kron @ B_9x9[e] @ p_pri_kron
            B_H_9x9[e] = p_sec_kron @ B_H_9x9[e] @ p_pri_kron            
        else:
            raise ValueError(f"Unknown element type: {branch_data}")
    return A_j_9x9, A_k_9x9, B_9x9, B_H_9x9


def build_permutation_matrix_open_split(branch_data):
    """
    The permutation matrix (i, j)-th index means (to, from).
    """
    # Canonical pairing: A - AB (split), B - BC, C - CA (open)
    wp = list(branch_data['winding_pairing'])
    permutation_pri, permutation_sec = np.zeros((3, 3), dtype=np.complex128), np.zeros((3, 3), dtype=np.complex128)
    # Split phase first
    p = 0
    if ('split_phase' in branch_data) and branch_data['split_phase']:
        wp.remove(branch_data['split_phase'])
        pri_idx = branch_data['split_phase'][0][0] - 1 if len(branch_data['split_phase'][0]) == 1 else STD_DELTA_ORDER.index(branch_data['split_phase'][0])
        sec_idx = STD_DELTA_ORDER.index(branch_data['split_phase'][1])
        permutation_pri[0, pri_idx] = 1        # Move C to A
        permutation_sec[0, sec_idx] = 1  # Move split phase to A
        p = 1
    # Regular connection
    for pri, sec in wp:
        pri_idx = pri[0] - 1 if len(pri) == 1 else STD_DELTA_ORDER.index(pri)
        sec_idx = sec[0] - 1 if len(sec) == 1 else STD_DELTA_ORDER.index(sec)
        permutation_pri[p, pri_idx] = 1
        permutation_sec[p, sec_idx] = 1
        p += 1
    # Open connection
    pri, sec = branch_data['open_phase']
    # TODO: The sign matters (+1 vs. -1). Right now we just ignored them.
    pri_idx = pri[0] - 1 if len(pri) == 1 else STD_DELTA_ORDER.index(pri)
    sec_idx = sec[0] - 1 if len(sec) == 1 else STD_DELTA_ORDER.index(sec)
    permutation_pri[2, pri_idx] = 1         # Move primary side open phase to C
    permutation_sec[2, sec_idx] = 1         # Move secondary side open phase to CA
    permutation_sec = permutation_sec.T
    assert (permutation_pri.sum(axis=0) == 1).all() and (permutation_pri.sum(axis=1) == 1).all(), (permutation_pri, branch_data)
    assert (permutation_sec.sum(axis=0) == 1).all() and (permutation_sec.sum(axis=1) == 1).all(), (permutation_sec, branch_data)
    return permutation_pri, permutation_sec
