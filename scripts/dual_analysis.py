"""
Dual Variable Sensitivity and Infrastructure Upgrade Study
Paper Section IV.D.1, Appendix B, Fig. 6, Table V (line/transformer upgrade rows)

Uses the dual variables mu from the converged LP solution to rank distribution
network branches by their impact on total PVHC. Appendix B derives this for
the *voltage*-driven mechanism (Danskin's theorem, reducing a branch's series
resistance):

    dL/dxi_i = 2 * sum_k  mu_k^T (C^{-T} kron I_3)(e_i e_i^T kron I_3) D_r^{ref} (C^{-1} kron I_3) p^(k)

where:
  mu_k   = mu_+^(k) - mu_-^(k)  (voltage upper minus lower dual variables)
  xi_i   = resistance scaling factor for branch i
  C, D_r^{ref} come from the wye-wye reference LinDistFlow model; see
  `DataLoader.get_branch_sensitivity_matrices()` for the scope of networks this is
  valid for (scalar per-branch turns ratio -- true for this example feeder).

This mechanism only has something to upgrade when voltage is actually a
binding constraint somewhere in the scenario set. On a *thermal*-bound network
(no scenario's voltage constraint ever binds, e.g. this example feeder/subtree)
dL/dxi_i is identically zero for every branch -- reconductoring for lower
resistance genuinely would not increase PVHC, so ranking upgrades by it is
meaningless there. `compute_thermal_rating_sensitivity` below computes the
analogous quantity for the mechanism that binds instead: a branch's own
thermal rating S_i appears directly as the RHS of its own flow constraints
(types 2-9), so by standard LP sensitivity theory the marginal PVHC gain per
unit increase in S_i is just the sum of the (non-negative) dual variables on
those constraints, weighted by how each constraint's RHS scales with S_i (1x
for the direct active/reactive bounds, types 2-5; sqrt(2)x for the linearized
P+-Q octagon bounds, types 6-9) -- no Danskin's-theorem/impedance-sensitivity
machinery needed, since S_i's effect here is direct rather than mediated
through the network.

Functions:
  compute_series_resistance_sensitivity(feeder_name, opt_subfolder):
    Loads saved dual_var array (shape T x 10 x 3N) from optimization results,
    computes dL/dxi_i for each branch, and produces a GIS map colored by
    sensitivity (Fig. 6). Positive sensitivity -> reducing branch resistance
    would increase PVHC; high-sensitivity branches are upgrade candidates.

  compute_thermal_rating_sensitivity(feeder_name, opt_subfolder):
    Same idea for dPVHC/dS_i (thermal rating). Positive sensitivity ->
    increasing branch i's thermal rating (e.g. reconductoring to a
    higher-ampacity conductor) would increase PVHC.

  solve_optimization_upgraded(feeder_name, dLdxi, edge_list, k, upgrade_amount,
  upgrade_type):
    Re-runs the iterative PVHC optimization (`run_pvhc_estimation
    .run_iterative_pvhc_analysis`) with the top-k branches upgraded --
    resistance divided by `upgrade_amount` (upgrade_type='resistance') or
    thermal rating multiplied by `upgrade_amount` (upgrade_type='rating') --
    reporting the resulting PVHC increase (Table V).

Required inputs:
  - Saved optimization results including dual_var, produced by
    `scripts/run_pvhc_estimation.py`:
    temp/optimization_results/<feeder>/iterative_eps0.01_beta0.05_utility_ratio_0pct/
"""
import numpy as np
import pathlib, sys
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.constants import FEEDERS, DEFAULT_FEEDER_NAME, PLOT_TITLE
from src.utils import load_optimization_results, PVHCTestConfig
from src.data_loader import DataLoader
from src.simulator import Simulator
from src.visualization import gis_visualize, load_node_coordinates, fill_missing_node_coordinates
from scripts.plot_figures import plot_opt_results
from scripts.run_pvhc_estimation import run_iterative_pvhc_analysis


def compute_series_resistance_sensitivity(feeder_name, opt_subfolder):
    """Compute dL/dxi_i for each branch -- implements Appendix B (Danskin's theorem).

    Loads the dual variable array from saved optimization results and computes
    the sensitivity of the Lagrangian to scaling branch i's resistance by xi_i.
    The Appendix B formula (vectorized for efficiency) is:

        dL/dxi_i = 2 * sum_k mu_k^T C^{-T} (e_i e_i^T kron I_3) D_r^{ref} C^{-1} p^(k)

    where mu_k = mu_+^(k) - mu_-^(k) are the voltage-bound dual variables for
    scenario k (constraint types 0 minus 1), and p^(k) is the nodal active
    power injection vector for scenario k.

    A high dL/dxi_i means reducing branch i's resistance (e.g., reconductoring)
    yields the largest marginal PVHC gain -- used to rank upgrades in Table V.
    Positive semi-definiteness of the sensitivity follows from convexity.
    """
    cfg = FEEDERS[feeder_name]
    data_loader = DataLoader(feeder_name)
    simulator = Simulator(data_loader)
    network_data = data_loader.get_circuit_model(feeder_root=cfg["feeder_root"])
    C_inv, C_T_inv, D_r, _, _ = data_loader.get_branch_sensitivity_matrices()

    opt_results = load_optimization_results(feeder_name, sub_folder=opt_subfolder)
    print("Loaded optimization results for feeder:", feeder_name, "T, C, N:", opt_results["dual_var"].shape)
    T, C, N = opt_results["dual_var"].shape
    N = N // 3
    # opt_results["dual_var"].shape = (num_scenarios, num_constraint_types, num_elements)
    # Constraint types (10 total):
    #   0: voltage_upper       - Upper bound on squared voltage
    #   1: voltage_lower       - Lower bound on squared voltage
    #   2: active_flow_upper   - Upper bound on active power flow
    #   3: active_flow_lower   - Lower bound on active power flow
    #   4: reactive_flow_upper - Upper bound on reactive power flow
    #   5: reactive_flow_lower - Lower bound on reactive power flow
    #   6: appr_flow_upper     - Upper bound on P+Q (linear approx of apparent power)
    #   7: appr_flow_lower     - Lower bound on P+Q
    #   8: appr_diff_flow_upper - Upper bound on P-Q
    #   9: appr_diff_flow_lower - Lower bound on P-Q

    # mu = mu_+ - mu_- (net voltage dual), shape (T, 3N) -- Appendix B
    mu_p, mu_m = opt_results["dual_var"][:, 0, :], opt_results["dual_var"][:, 1, :]
    mu = mu_p - mu_m

    # Compute real power injection p^(k) for each scenario k
    print("Computing real power injection...")
    node2idx = {n: i for i, n in enumerate(opt_results['node_list'])}
    injection_map = opt_results["load_info"] + opt_results["utility_pv_info"] + opt_results["customer_pv_info"]
    n_loads = opt_results["load_profiles"][0].shape[1]
    n_pvs = len(opt_results["capacities_isa_utility"]) + len(opt_results["capacities_isa_customer"])
    si = opt_results["scenario_indices"]
    p_inj = np.zeros((T, n_loads + n_pvs), dtype=np.float64, order='C')
    p_inj[:, :n_loads] = opt_results["load_profiles"][0, si]
    pv_capacities = np.concatenate([opt_results["capacities_isa_utility"], opt_results["capacities_isa_customer"]])
    p_inj[:, n_loads:] = opt_results["pv_profiles"][si, None] * pv_capacities[None, :]
    p_inj_sparse, _, _ = simulator._map_injections_to_buses(
        p_inj,
        np.zeros_like(p_inj),
        injection_map,
        node2idx, T, N
    )
    assert network_data['node_list'] == opt_results['node_list']
    assert network_data['edge_list'] == opt_results['edge_list']

    # Vectorized Appendix B formula:
    #   left  = mu . C^{-T}   shape (T, 3N)
    #   right = C^{-1} . p^T  shape (3N, T)
    #   for branch i: dL/dxi_i = 2*Tr(left[:,3i:3i+3] . D_r[3i:3i+3,:] . right)
    # C^{-1} is the wye-wye model's, like D_r (network_data["C_tilde_S"] is the 3-phase
    # model's, which is complex below a delta winding)
    left = mu @ C_T_inv   # (T, 3N)
    right = C_inv @ p_inj_sparse.T  # (3N, T)

    dLdxi = np.zeros(N)
    for i in tqdm(range(N)):
        left_i = left[:, i*3:(i+1)*3]   # (T, 3)
        D_r_i  = D_r[i*3:(i+1)*3, :]     # (3, 3N)
        temp   = D_r_i @ right           # (3, T)
        dLdxi[i] = 2 * np.sum(left_i * temp.T)

    assert (dLdxi > -1e-6).all(), "Sensitivity should be non-negative for all lines"

    """Printing and visualization"""
    k = 10  # Number of top branches to upgrade
    top_k_indices = np.argsort(dLdxi)[::-1][:k]
    print(f"\nTop {k} branch components to upgrade (by sensitivity):")
    for rank, idx in enumerate(top_k_indices, 1):
        edge = opt_results['edge_list'][idx]
        print(f"  {rank}. {data_loader.circuit_dict[edge]['element_name']} {edge} (sensitivity: {dLdxi[idx]:.6e})")

    # GIS plot with edges colored by sensitivity
    node_list = opt_results['node_list']
    node2idx = {node: idx for idx, node in enumerate(node_list)}
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in opt_results['edge_list']]

    node_coord_file = pathlib.Path(cfg["data_path"]) / "opendss" / "Buscoords_ckt5.dss"
    node_coords_array = load_node_coordinates(node_coord_file, node_list=node_list, lowercase=True)
    fill_missing_node_coordinates(node_coords_array, edge_list)

    output_dir = pathlib.Path(__file__).resolve().parents[1] / "temp" / "plots" / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gis_visualize(
        node_coords=node_coords_array,
        edge_list=edge_list,
        edge_values=dLdxi,
        outfile=output_dir / f'map_sensitivity_dLdxi_{opt_subfolder}.png',
        cb_label_edge='Sensitivity dL/dxi',
        cmap_edge=LinearSegmentedColormap.from_list('blue_yellow_red', ['blue', 'yellow', 'red']),
        text=f"{np.sum(dLdxi > 1e-3 * np.max(dLdxi))} elements with positive sensitivity" if PLOT_TITLE else None,
        zero_length_edge_threshold=1e-3,
        title=f'Series Resistance Sensitivity: {feeder_name}' if PLOT_TITLE else None
    )
    print(f"\nGIS plot saved to: {output_dir / f'map_sensitivity_dLdxi_{opt_subfolder}.png'}")
    return dLdxi, opt_results['edge_list']


def compute_thermal_rating_sensitivity(feeder_name, opt_subfolder):
    """Compute dPVHC/dS_i (thermal rating sensitivity) for each branch.

    See module docstring for the derivation -- unlike the voltage/resistance
    case, S_i is the direct RHS of branch i's own flow constraints, so this is
    just a weighted sum of that branch's own (already-saved) dual variables;
    no impedance-sensitivity matrices or re-derivation needed.
    """
    cfg = FEEDERS[feeder_name]
    data_loader = DataLoader(feeder_name)
    data_loader._dssParser()
    data_loader._format_circuit_data(cfg["feeder_root"], plot_circuit=False)

    opt_results = load_optimization_results(feeder_name, sub_folder=opt_subfolder)
    dual_var = opt_results["dual_var"]
    T, C, num_elements = dual_var.shape
    N = num_elements // 3

    # Constraint types 2-9 (see module docstring); RHS = S_i for 2-5 (active,
    # reactive), RHS = sqrt(2)*S_i for 6-9 (linearized P+-Q octagon bounds).
    active_sum = dual_var[:, 2, :] + dual_var[:, 3, :]
    reactive_sum = dual_var[:, 4, :] + dual_var[:, 5, :]
    pq_sum = dual_var[:, 6, :] + dual_var[:, 7, :]
    pq_diff_sum = dual_var[:, 8, :] + dual_var[:, 9, :]
    per_element = (active_sum + reactive_sum + np.sqrt(2) * (pq_sum + pq_diff_sum)).sum(axis=0)
    dSdrating = per_element.reshape(N, 3).sum(axis=1)

    assert (dSdrating > -1e-6).all(), "Sensitivity should be non-negative for all branches"

    k = 10
    top_k_indices = np.argsort(dSdrating)[::-1][:k]
    print(f"\nTop {k} branch components to upgrade (by thermal rating sensitivity):")
    for rank, idx in enumerate(top_k_indices, 1):
        edge = opt_results['edge_list'][idx]
        element_name = data_loader.circuit_dict.get(edge, {}).get('element_name', '?')
        print(f"  {rank}. {element_name} {edge} (sensitivity: {dSdrating[idx]:.6e})")

    node_list = opt_results['node_list']
    node2idx = {node: idx for idx, node in enumerate(node_list)}
    edge_list = [(node2idx[e[0]], node2idx[e[1]]) for e in opt_results['edge_list']]

    node_coord_file = pathlib.Path(cfg["data_path"]) / "opendss" / "Buscoords_ckt5.dss"
    node_coords_array = load_node_coordinates(node_coord_file, node_list=node_list, lowercase=True)
    fill_missing_node_coordinates(node_coords_array, edge_list)

    output_dir = pathlib.Path(__file__).resolve().parents[1] / "temp" / "plots" / feeder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gis_visualize(
        node_coords=node_coords_array,
        edge_list=edge_list,
        edge_values=dSdrating,
        outfile=output_dir / f'map_sensitivity_dSdrating_{opt_subfolder}.png',
        cb_label_edge='Thermal Rating Sensitivity',
        cmap_edge=LinearSegmentedColormap.from_list('blue_yellow_red', ['blue', 'yellow', 'red']),
        text=f"{np.sum(dSdrating > 1e-3 * np.max(dSdrating))} elements with positive sensitivity" if PLOT_TITLE else None,
        zero_length_edge_threshold=1e-3,
        title=f'Thermal Rating Sensitivity: {feeder_name}' if PLOT_TITLE else None
    )
    print(f"\nGIS plot saved to: {output_dir / f'map_sensitivity_dSdrating_{opt_subfolder}.png'}")
    return dSdrating, opt_results['edge_list']


def solve_optimization_upgraded(feeder_name, dLdxi, edge_list, k, upgrade_amount=2., upgrade_type='resistance'):
    """Upgrade the top-k most sensitive branches, then re-run Algorithm 1 on
    the upgraded network (Table V).

    upgrade_type='resistance': halve series resistance (Appendix B's original
        formulation) -- helps when voltage is the binding constraint.
    upgrade_type='rating': multiply the branch's thermal (S) rating by
        upgrade_amount (e.g. reconductoring to a higher-ampacity conductor) --
        helps when the branch's own thermal limit is what's binding, which
        `compute_thermal_rating_sensitivity` ranks for.
    """
    cfg = FEEDERS[feeder_name]

    data_loader = DataLoader(feeder_name)
    data_loader._dssParser()
    data_loader._format_circuit_data(cfg["feeder_root"], plot_circuit=False)

    top_k_indices = np.argsort(dLdxi)[::-1][:k]
    print(f"\nTop {k} branch components to upgrade ({upgrade_type}):")
    for idx in top_k_indices:
        edge_data = data_loader.circuit_dict[edge_list[idx]]
        if upgrade_type == 'resistance':
            r, x = np.real(edge_data['series_impedance']), np.imag(edge_data['series_impedance'])
            edge_data['series_impedance'] = r / upgrade_amount + 1j * x
        elif upgrade_type == 'rating':
            if edge_data.get('S_rating'):
                edge_data['S_rating'] = edge_data['S_rating'] * upgrade_amount
        else:
            raise ValueError(f"Unknown upgrade_type: {upgrade_type!r}")

    # Recompute the LinDistFlow model on the upgraded network (bypass the
    # on-disk cache, which holds the pre-upgrade matrices).
    data_loader.get_circuit_model(feeder_root=cfg["feeder_root"], recompute=True, save=False)

    subfolder = f"iterative_eps0.01_beta0.05_utility_ratio_0pct_upgraded_{k}"
    config = PVHCTestConfig(
        feeder_name=feeder_name,
        feeder_root=cfg["feeder_root"],
        v_root_nom=cfg["v_root_nom"],
        v_root_multiplier=cfg["v_root_multiplier"],
        num_scenarios=50,
        initial_sample_size=10,
        maximum_subproblem_size=10,
        seed=42,
        sub_folder=subfolder,
        with_fairness=False,
        utility_pv_ratio=0.0,
        # Must match the base case's capacity regime (use_capacity_limits=True
        # in run_pvhc_estimation.py) -- otherwise this silently re-solves
        # under the flat 10 kW default instead of the real Eq. 12 x
        # capacity_multiplier limits, making the upgraded-vs-base comparison
        # meaningless (found via: upgraded_k results were capped at exactly
        # 10.0 kW for ~90%+ of customers despite the base case reaching ~60 kW).
        use_capacity_limits=True,
        # NOT load_cached_limits=True here, unlike run_pvhc_estimation.py:
        # the *network* limits cache (temp/cache/limits/<feeder>_limits.pkl,
        # a different cache than the capacity one above) is keyed only by
        # feeder name, with no awareness that this network's branches were
        # just modified (upgrade_type). Worse, get_circuit_model(recompute=
        # True) above re-derives node/edge ordering fresh each call (not
        # guaranteed identical to the on-disk-cached ordering every other
        # script relies on), so reusing that cache here would silently
        # compare this run's flows against a *different, stale, and
        # potentially index-misaligned* set of limits -- found via a base
        # (zero-PV) feasibility check that exploded from ~1 violation to
        # 700,000+ once this was left on.
        load_cached_limits=False,
    )

    return run_iterative_pvhc_analysis(
        config=config,
        epsilon=0.01,
        beta=0.05,
        n_I=1,
        max_outer_iterations=20,
        N_test=5000,
        verbose=True,
        data_loader=data_loader,
    )


if __name__ == "__main__":
    feeder_name = DEFAULT_FEEDER_NAME
    opt_subfolder = "iterative_eps0.01_beta0.05_utility_ratio_0pct"
    dLdxi_voltage, edge_list = compute_series_resistance_sensitivity(feeder_name, opt_subfolder)
    dSdrating, _ = compute_thermal_rating_sensitivity(feeder_name, opt_subfolder)

    # Auto-detect which mechanism actually binds on this feeder: if voltage
    # sensitivity is uniformly ~0 (no scenario's voltage constraint ever
    # binds -- true by default on this example feeder/subtree), reconductoring
    # for lower resistance genuinely wouldn't increase PVHC, so rank/upgrade
    # by thermal rating instead. On a feeder where voltage does bind, this
    # falls back to the original Appendix B resistance-based ranking unchanged.
    if np.max(dLdxi_voltage) < 1e-9:
        print("\n[Info] Voltage sensitivity is uniformly zero on this feeder "
              "(thermal/branch-flow is the binding mechanism) -- using "
              "thermal rating sensitivity to select and apply upgrades.")
        dLdxi, upgrade_type = dSdrating, 'rating'
    else:
        dLdxi, upgrade_type = dLdxi_voltage, 'resistance'

    for k in [1, 5, 10, 15]:
        solve_optimization_upgraded(feeder_name, dLdxi, edge_list, k=k, upgrade_type=upgrade_type)
        plot_opt_results(
            feeder_name,
            opt_subfolders=[f"iterative_eps0.01_beta0.05_utility_ratio_0pct_upgraded_{k}"],
            ext='png'
        )
