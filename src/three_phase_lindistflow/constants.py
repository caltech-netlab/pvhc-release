"""Fixed matrices and connection maps used throughout the three-phase model."""

import numpy as np
import scipy as sp

__all__ = [
    "I_3",
    "I_9",
    "I_23",
    "Gamma",
    "alpha",
    "pos_seq",
    "F",
    "gamma",
    "alpha_p",
    "GI",
    "IG",
    "GTI",
    "IGT",
    "GG",
    "GTGT",
    "gamma_hat",
    "gamma_hat_H",
    "M",
    "Q_delta",
    "Q_deltasplit",
    "Q_windingsplit",
    "U_delta",
    "U_delta_3x3",
    "T_delta",
    "beta",
    "beta_inv",
    "BETA",
    "BETA_H",
    "D2Y",
    "openD_split2Y",
    "branch_pwr_relation",
    "branch_internal_pwr_relation",
    "A_j_9x9_all",
    "A_k_9x9_all",
    "STD_DELTA_ORDER",
]

I_3 = np.eye(3, dtype=np.complex128)


I_9 = np.eye(9, dtype=np.complex128)


I_23 = np.eye(2,3, dtype=np.complex128)


Gamma = np.array([
    [1, -1, 0],
    [0, 1, -1],
    [-1, 0, 1]
], dtype=np.complex128)


alpha = np.exp(-1j * 2/3 * np.pi)


pos_seq = np.array([1, alpha, alpha**2], dtype=np.complex128)


F = np.array([
    [1, 1, 1],
    [1, alpha, alpha**2],
    [1, alpha**2, alpha]
], dtype=np.complex128) / np.sqrt(3)


gamma = np.array([
    [1, alpha**2, alpha],
    [alpha, 1, alpha**2],
    [alpha**2, alpha, 1]
], dtype=np.complex128)


alpha_p = gamma[:, 0]


GI = sp.linalg.block_diag(Gamma, I_3)


IG = sp.linalg.block_diag(I_3, Gamma)


GTI = sp.linalg.block_diag(Gamma.T, I_3)


IGT = sp.linalg.block_diag(I_3, Gamma.T)


GG = sp.linalg.block_diag(Gamma, Gamma)


GTGT = sp.linalg.block_diag(Gamma.T, Gamma.T)


gamma_hat = np.array([
    [1,        0,       0],
    [alpha,    0,       0],
    [alpha**2, 0,       0],
    [0,        alpha**2,0],
    [0,        1,       0],
    [0,        alpha,   0],
    [0,        0,       alpha],
    [0,        0,       alpha**2],
    [0,        0,       1]
], dtype=np.complex128)


gamma_hat_H = np.array([
    [1,        0,       0],
    [0,        alpha,   0],
    [0,        0,       alpha**2],
    [alpha**2, 0,       0],
    [0,        1,       0],
    [0,        0,       alpha],
    [alpha,    0,       0],
    [0,        alpha**2,0],
    [0,        0,       1]
], dtype=np.complex128)


M = np.array([
    [        1, 0, 0,      0],
    [        0, 0, 1,      0],
    [alpha**2, 0, 0,      0],
    [        0, 1, 0,      0],
    [        0, 0, 0,      1],
    [        0, 0, 0,  alpha],
    [        0, 0, 0,      0],
    [        0, 0, 0,      0],
    [        0, 0, 0,      0],
], dtype=np.complex128)


Q_delta = np.array([
    [ 1, 0,  0],
    [ 0, 0, -1],
], dtype=np.complex128)


Q_deltasplit = np.array([
    [ 1, 0, 0],
    [ 0, 1, 1],
    [ 0, 0, -1],
], dtype=np.complex128)


Q_windingsplit = np.array([
    [ 1/2, -1/2,  0],
    [ 0, 0, 1],
], dtype=np.complex128)


U_delta = np.array([
    [ 1/2,  0],
    [-1/2,  0],
    [-1/2, -1],
], dtype=np.complex128)


U_delta_3x3 = np.array([
    [ 1/2,  0,    0],
    [-1/2,  0,    0],
    [-1/2, -1,    0],
], dtype=np.complex128)


T_delta = np.array([
    [ 1,  0],
    [ 0,  1],
    [-1, -1],
], dtype=np.complex128)


beta = np.array([
    [1],
    [-1],
    [np.sqrt(3)*1j]
], dtype=np.complex128)


beta_inv = np.array([
    [1],
    [-1],
    [-(1/np.sqrt(3))*1j]
], dtype=np.complex128)


BETA = np.kron(I_3, beta @ beta_inv.T)[:, [0, 4, 8]]   # (9, 3), maps diag(s_k) to vec(s_k)


BETA_H = np.kron((beta @ beta_inv.T).conj(), I_3)[:, [0, 4, 8]]  # (9, 3), maps diag(s_k^H) to vec(s_k^H)


# Maps the power of delta legs AB, BC, CA (columns, as in STD_DELTA_ORDER) to the wye-equivalent
# per-phase powers under BVA. Each column sums to 1, since a leg's power is split over its phases.
# This is equal to $\frac{1}{3} (\Gamma^\top \gamma) \circ \Gamma^\top$
D2Y = 1/(1-alpha)*np.array([[1, 0, -alpha], [-alpha, 1, 0], [0, -alpha, 1]])


openD_split2Y = np.kron(Gamma.T, U_delta_3x3 @ gamma)[[0, 4, 8]][:, [0, 4, 8]]


branch_pwr_relation = {
    'line': I_9,
    'YY': I_9,
    'DY': np.kron(Gamma.T, Gamma.T) / 3,
    'DY_alt': np.kron(Gamma, Gamma) / 3,
    'YD': np.kron(Gamma, Gamma) / 3,
    'DD': I_9,
    'openY_openD_split': M @ np.kron(Q_windingsplit @ Q_deltasplit, Gamma[:2]),
    'openY_openY': I_9,
    'openD_openD': I_9,
    'openD_openD_split': (1/3) * np.kron((Gamma[:2].T @ Q_windingsplit @ Q_deltasplit), (Gamma.T @ T_delta @ Gamma[:2]))
}


branch_internal_pwr_relation = {
    'line': I_9,
    'YY': I_9,
    'DY': I_9,
    'DY_alt': I_9,
    'YD': np.kron(Gamma, Gamma) / 3,
    'DD': np.kron(Gamma, Gamma) / 3,
    # TODO: In this implementation, the open/split-phase transformers still use the external power injection
    'openY_openD_split': M @ np.kron(Q_windingsplit @ Q_deltasplit, Gamma[:2]),
    'openY_openY': I_9,
    'openD_openD': I_9,
    'openD_openD_split': (1/3) * np.kron((Gamma[:2].T @ Q_windingsplit @ Q_deltasplit), (Gamma.T @ T_delta @ Gamma[:2]))
}


A_j_9x9_all = {
    'line': I_9, 
    'YY': I_9,
    'DY': np.kron(Gamma, Gamma),
    'DY_alt': np.kron(Gamma.T, Gamma.T),
    'YD': I_9,
    'DD': np.kron(Gamma, Gamma),
    'openY_openY': I_9,
}


A_k_9x9_all = {
    'line': I_9, 
    'YY': I_9,
    'DY': I_9,
    'DY_alt': I_9,
    'YD': np.kron(Gamma, Gamma),
    'DD': np.kron(Gamma, Gamma),
    'openY_openY': I_9,
}


STD_DELTA_ORDER = [(1,2), (2,3), (1,3)]
