"""Three-phase linearized distribution power flow (LinDistFlow).

Public API is re-exported here, so the usual entry point is::

    import three_phase_lindistflow as ldf

    loader = ldf.DataLoader("data/tiny_dy_dy_line_3phase")
    loader.parse()                                   # compile the OpenDSS model first
    circuit_data = loader.to_circuit_data(feeder_root="bus0")
    C_0, R, X, C_inv, node_list, edge_list = ldf.three_phase_lindistflow_real_wrapper(circuit_data)

``DataLoader.parse()`` populates the parsed OpenDSS tables and must be called before
``to_circuit_data()``. See README.md for an end-to-end example that also applies loads
and solves for voltages.

Submodules are importable directly when you want a narrower namespace, e.g.
``from three_phase_lindistflow.bim import BIM_IV_model``.
"""

import logging

from . import (
    constants,
    linalg,
    topology,
    branches,
    bim,
    voltages,
    baselines,
    model,
    plotting,
    analysis,
    dss_parser,
    power_flow,
)

# Matrix constants and connection maps
from .constants import (
    I_3,
    I_9,
    I_23,
    Gamma,
    alpha,
    pos_seq,
    F,
    gamma,
    alpha_p,
    GI,
    IG,
    GTI,
    IGT,
    GG,
    GTGT,
    gamma_hat,
    gamma_hat_H,
    M,
    Q_delta,
    Q_deltasplit,
    Q_windingsplit,
    U_delta,
    U_delta_3x3,
    T_delta,
    beta,
    beta_inv,
    BETA,
    BETA_H,
    D2Y,
    openD_split2Y,
    branch_pwr_relation,
    branch_internal_pwr_relation,
    A_j_9x9_all,
    A_k_9x9_all,
    STD_DELTA_ORDER,
)
# Block-matrix helpers
from .linalg import (
    matrix2block,
    block2matrix,
    vectorize,
    vec_diag,
    print_block_matrix_nonzero_row,
)
# Graph construction and validation
from .topology import (
    to_edge_list,
    to_edge_dict,
    edges_between_nodes,
    subtree,
    is_downward_oriented,
    check_network_validity,
    build_node_data,
    zero_sequence_groups,
    zero_sequence_islands,
)
# Branch impedance and component models
from .branches import (
    convert_series_impedance_3x3,
    validate_line_phase_structure,
    to_single_phase_series_impedance,
    branch_data_to_admittance_matrix,
    floating_neutrals,
    component_model,
    build_permutation_matrix_open_split,
)
# Bus injection (admittance) model
from .bim import (
    BIM_IV_model,
    compute_branch_power_flow,
    four_parameter_branch_model,
)
# Nominal voltage helpers
from .voltages import (
    compute_nominal_voltages_from_turns_ratio,
    compute_nominal_complex_voltages_from_turns_ratio,
    compute_v0_ref,
)
# Baseline models for comparison
from .baselines import (
    gan_hxu_3pldf,
    single_phase_lindistflow,
)
# Three-phase LinDistFlow
from .model import (
    branch_nodal_power,
    three_phase_lindistflow,
    three_phase_lindistflow_real,
    three_phase_lindistflow_real_wrapper,
    three_phase_lindistflow_linear_constraint,
)
# Plotting
from .plotting import (
    plot_nx_graph,
)

# Load profiles and imbalance metrics
from .analysis import (
    average_imbalance,
    average_zero_sequence_component,
    build_imbalanced_load_profile,
    remove_zero_sequence_load_current,
    load_dss_feeder,
)

# Parser and nonlinear solver
from .dss_parser import DataLoader, determine_valid_phases
from .power_flow import solve_power_flow, generate_dataset, PowerFlowDivergedError

# A library must not configure logging for its host application; a NullHandler keeps
# "no handlers" warnings away while leaving all configuration to the caller.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "constants",
    "linalg",
    "topology",
    "branches",
    "bim",
    "voltages",
    "baselines",
    "model",
    "plotting",
    "analysis",
    "dss_parser",
    "power_flow",
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
    "matrix2block",
    "block2matrix",
    "vectorize",
    "vec_diag",
    "print_block_matrix_nonzero_row",
    "to_edge_list",
    "to_edge_dict",
    "edges_between_nodes",
    "subtree",
    "branch_nodal_power",
    "is_downward_oriented",
    "check_network_validity",
    "build_node_data",
    "zero_sequence_groups",
    "zero_sequence_islands",
    "convert_series_impedance_3x3",
    "validate_line_phase_structure",
    "to_single_phase_series_impedance",
    "branch_data_to_admittance_matrix",
    "floating_neutrals",
    "component_model",
    "build_permutation_matrix_open_split",
    "BIM_IV_model",
    "compute_branch_power_flow",
    "four_parameter_branch_model",
    "compute_nominal_voltages_from_turns_ratio",
    "compute_nominal_complex_voltages_from_turns_ratio",
    "compute_v0_ref",
    "gan_hxu_3pldf",
    "single_phase_lindistflow",
    "three_phase_lindistflow",
    "three_phase_lindistflow_real",
    "three_phase_lindistflow_real_wrapper",
    "three_phase_lindistflow_linear_constraint",
    "plot_nx_graph",
    "DataLoader",
    "determine_valid_phases",
    "solve_power_flow",
    "generate_dataset",
    "PowerFlowDivergedError",
    "average_imbalance",
    "average_zero_sequence_component",
    "build_imbalanced_load_profile",
    "remove_zero_sequence_load_current",
    "load_dss_feeder",
]
