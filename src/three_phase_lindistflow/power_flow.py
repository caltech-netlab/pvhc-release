"""Backward-forward sweep (BFS) power flow and synthetic dataset generation."""

import logging

import numpy as np
import networkx as nx

from .constants import Gamma, pos_seq
from .topology import check_network_validity, zero_sequence_groups
from .branches import branch_data_to_admittance_matrix, convert_series_impedance_3x3, floating_neutrals
from .bim import four_parameter_branch_model
from .voltages import compute_nominal_complex_voltages_from_turns_ratio

logger = logging.getLogger(__name__)

__all__ = ["PowerFlowDivergedError", "solve_power_flow", "generate_dataset"]


class PowerFlowDivergedError(ValueError):
    """Raised when the backward-forward sweep blows up instead of converging.

    Subclasses ``ValueError`` so that callers written against the previous
    behaviour (which raised a plain ``ValueError``) keep working.
    """


def _reshape_phase_array(values, n_nodes, name):
    array = np.asarray(values, dtype=complex)
    if array.ndim == 1:
        if array.size != n_nodes * 3:
            raise ValueError(f"{name} must have length {n_nodes * 3}, got {array.size}")
        return array.reshape(n_nodes, 3)
    if array.ndim == 2 and array.shape == (n_nodes, 3):
        return array.astype(complex, copy=False)
    if array.ndim == 2 and array.shape == (3, n_nodes):
        return array.T.astype(complex, copy=False)
    raise ValueError(f"{name} must have shape ({n_nodes}, 3) or length {n_nodes * 3}, got {array.shape}")


def _root_voltage_vector(v0):
    """Normalize ``v0`` to a (3,) complex line-to-neutral phasor at the root bus."""
    array = np.asarray(v0, dtype=complex)
    if array.ndim == 0:
        magnitude = array.item()
        if magnitude == 0:
            raise ValueError("v0 must be non-zero: it sets the voltage scale of the initial guess")
        return magnitude * pos_seq
    array = array.reshape(-1)
    if array.size != 3:
        raise ValueError(
            "v0 must be a scalar line-to-neutral magnitude or a (3,) complex phasor, "
            f"got shape {np.shape(v0)}"
        )
    # Every phase must be non-zero, not just the vector as a whole: the sweep divides by
    # the voltages, so a single zero component (e.g. v0=[7200, 0, 0]) poisons the solution
    # with NaN instead of modelling a missing phase.
    if np.any(array == 0):
        raise ValueError(
            "v0 must be non-zero in every phase: it sets the voltage scale of the initial "
            f"guess and the sweep divides by it, got {np.asarray(v0)}. To model a feeder "
            "with missing phases, keep a non-zero starting voltage on every phase."
        )
    return array


def _nominal_initial_voltages(graph, node_list, node2idx, circuit_dict, root, v0):
    """Propagate the root voltage through the transformer turns ratios for a start point."""
    root_voltage = _root_voltage_vector(v0)
    # compute_nominal_complex_voltages_from_turns_ratio seeds row 0 of its output with the
    # root voltage, so hand it an ordering whose first entry is the root and then scatter
    # the result back into the caller's node ordering.
    order = [root] + [node for node in node_list if node != root]
    order2idx = {node: k for k, node in enumerate(order)}
    nominal = compute_nominal_complex_voltages_from_turns_ratio(
        graph, order, root_voltage, circuit_dict, order2idx
    )
    V = np.empty((len(node_list), 3), dtype=complex)
    for node, k in order2idx.items():
        V[node2idx[node]] = nominal[k]
    return V


def _zero_sum_projection(branch_data):
    """Projection removing the mean over the windings of a bank (3x3), for a floating neutral.

    The currents of the windings on a floating neutral sum to zero, and the voltage of the
    neutral is free. So on the windings present (the phases the bank's admittance has; at
    least two, see floating_neutrals) the sweep drops the mean of the current delivered below
    and of the voltage given to the secondary: the zero-sequence part, for a three-phase bank.
    It is the identity on a missing phase.
    """
    active = np.any(branch_data_to_admittance_matrix(branch_data) != 0, axis=1)
    return np.eye(3) - np.outer(active, active) / active.sum()


def _thevenin_impedances(traversal, parents, circuit_dict, z_hat_s, root):
    """Upstream Thevenin impedance (3x3) at every node of the radial network.

    ``z_th[k]`` maps a current drawn at node k to the voltage drop it causes there, with the
    root an ideal source and every other injection held fixed. It follows the sweep
    equations: a line adds its impedance; a transformer refers the upstream impedance plus
    its own through its winding connection. A delta winding blocks zero sequence, so below
    a DY bank the zero-sequence impedance is that of the bank alone. A floating wye neutral
    blocks it too, and the sweep gives what it leaves below no zero-sequence voltage.
    """
    z_th = {root: np.zeros((3, 3), dtype=complex)}
    for node in traversal:
        if node == root:
            continue
        parent = parents[node]
        branch_data = circuit_dict[(parent, node)]
        element_type = branch_data["element_type"]
        z = z_hat_s[(parent, node)]
        z_up = z_th[parent]
        a2 = branch_data.get("turns_ratio", 1.0) ** 2 if element_type != "line" else 1.0
        # The projection is the identity unless the neutral floats (see _zero_sum_projection).
        P = _zero_sum_projection(branch_data) if any(floating_neutrals(branch_data)) else np.eye(3)
        if element_type == "line":
            z_th[node] = z_up + z
        elif element_type in ("YY", "openY_openY"):
            z_th[node] = P @ (z_up + z) @ P / a2
        elif element_type == "DY":
            z_th[node] = P @ (Gamma @ z_up @ Gamma.T + z) @ P / a2
        elif element_type == "DY_alt":
            z_th[node] = P @ (Gamma.T @ z_up @ Gamma + z) @ P / a2
        elif element_type == "DD":
            z_th[node] = Gamma.T @ (Gamma @ z_up @ Gamma.T + z) @ Gamma / (9 * a2)
        elif element_type == "YD":
            z_th[node] = Gamma.T @ (z_up + z) @ Gamma / (9 * a2)
        elif element_type == "YD_alt":
            z_th[node] = Gamma @ (z_up + z) @ Gamma.T / (9 * a2)
        else:
            raise ValueError(f"Un-implemented branch type {element_type} for branch {(parent, node)}")
    return z_th


def _zero_sequence_impedance_matrix(banks, traversal, parents, circuit_dict, z_hat_s, root):
    """Zero-sequence impedance matrix between the primary buses of wye-delta banks.

    ``z0[i, j]`` is the drop of sum(V) at the primary bus of bank i for a current of 1 A
    drawn on each phase at the primary bus of bank j (the zero-sequence current circulating
    in bank j), with the root an ideal source. It is that of the upstream path they share,
    up to their lowest common ancestor, carried to and from it through wye-wye turns ratios.
    Below a delta winding or a floating wye neutral a zero-sequence current or voltage does
    not reach the other side, so banks separated by one are not coupled.
    """
    z_thevenin = _thevenin_impedances(traversal, parents, circuit_dict, z_hat_s, root)

    def path_from_root(node):
        path = [node]
        while path[-1] != root:
            path.append(parents[path[-1]])
        return path[::-1]

    def referral(path):
        """Factor a zero-sequence current or voltage picks up between path[0] and path[-1]."""
        factor = 1.0
        for edge in zip(path[:-1], path[1:]):
            element_type = circuit_dict[edge]["element_type"]
            if element_type in ("YY", "openY_openY") and not any(floating_neutrals(circuit_dict[edge])):
                factor /= circuit_dict[edge]["turns_ratio"]
            elif element_type != "line":
                return 0.0
        return factor

    paths = [path_from_root(edge[0]) for edge in banks]
    z0 = np.zeros((len(banks), len(banks)), dtype=complex)
    for i, path_i in enumerate(paths):
        for j, path_j in enumerate(paths):
            n_common = 0
            while n_common < min(len(path_i), len(path_j)) and path_i[n_common] == path_j[n_common]:
                n_common += 1
            ancestor = n_common - 1
            z0[i, j] = (np.sum(z_thevenin[path_i[ancestor]])
                        * referral(path_i[ancestor:]) * referral(path_j[ancestor:]))
    return z0


def solve_power_flow(circuit_data, p, q, slack_bus, node_list, V_init=None, rtol=1e-12,
                     max_iter=1e5, v0=None):
    """
    Solve 3-phase AC power flow using Backward-Forward Sweep.

    Parameters
    ----------
    circuit_data : list of branch dicts describing the (radial) circuit
    p, q : real/reactive power injections, shape (n, 3) or flat (3n,) in
        phase-major order (node 0 phases A/B/C, node 1 phases A/B/C, ...).
        The slack-bus row is ignored -- it is zeroed before the sweep (a non-zero
        slack row is dropped with a warning), because the slack current is fixed
        by the network rather than specified: the returned ``I_bus`` at the slack
        bus is the sum of the currents leaving the root, which is what makes
        ``I_bus = Y @ V`` hold at the root as well.
    slack_bus : index into ``node_list`` (or node name) of the slack/reference bus.
        It must be the root of the radial network.
    node_list : ordering of nodes used for every returned array. It must list
        exactly the nodes of the radial network built from ``circuit_data``, once
        each -- extra ("off-tree") names are rejected rather than silently given
        undefined voltages and ignored injections. ``None`` falls back to the BFS
        ordering derived from ``circuit_data``.
    V_init : complex starting voltages, shape (n, 3) or flat (3n,), in the same
        units as the network impedances (volts, unless the data is per-unit).
        Only the non-root rows are an initial guess. The forward sweep never
        writes to the root row, so ``V_init[slack_bus]`` is the slack boundary
        condition: it is the root voltage of the converged solution, and the sole
        lever on the voltage scale when ``V_init`` is used. Every entry must be
        non-zero, since the sweep divides by the voltages.
    rtol : relative tolerance for convergence (voltage and current mismatch, and with
        wye-delta banks the zero-sequence voltage mismatch their circulating currents leave)
    max_iter : maximum number of sweeps
    v0 : root/slack voltage, used *instead of* ``V_init`` to build the whole
        starting point. Either a scalar line-to-neutral magnitude (expanded to a
        balanced positive-sequence phasor) or a (3,) complex phasor, non-zero in
        every phase. It is propagated down the tree through the transformer turns
        ratios, and its root value is the slack boundary condition (see ``V_init``).

    Wye neutrals are grounded, except where a transformer's ``floating_neutral`` says
    otherwise (see ``floating_neutrals``). Below a delta winding or a floating neutral, no
    zero-sequence current can be delivered, so the sweep drops the zero-sequence part of the
    current drawn below (``I_bus = Y @ V`` then fails by it there) and gives the secondary no
    zero-sequence voltage. Grounded wye-delta banks on a part of the network that nothing else
    supplies with zero-sequence current (grounding banks on an ungrounded system, below a delta
    winding or a floating neutral), one or several, are not supported.

    Exactly one voltage scale is required: pass ``V_init`` or ``v0``, never both.
    There is no safe default -- a flat 1.0 V start silently diverges on a feeder
    modelled in volts -- so passing neither raises ``ValueError``; passing both
    also raises ``ValueError`` instead of quietly dropping one of them. To move
    the slack voltage of a warm start, write its root row (``V_init[slack_bus]``)
    rather than adding ``v0``.

    Returns
    -------
    V : (3n,) complex nodal voltage phasors, flattened phase-major
        (node 0 phases A/B/C, node 1 phases A/B/C, ...)
    I_bus : (3n,) complex nodal current injections, same flattening as ``V``
    I_branch : (3m,) complex branch currents at the source (sending) end, where
        m = len(circuit_data), ordered as ``circuit_data`` and flattened per phase
    converged : bool, True if every mismatch fell below ``rtol``

    Raises
    ------
    ValueError
        If neither ``V_init`` nor ``v0`` is given, if both are given, or if the
        inputs are malformed -- including a zero entry in the starting voltages
        and a ``node_list`` that is not exactly the node set of the network.
    PowerFlowDivergedError
        If the voltage magnitudes blow up (a subclass of ``ValueError``).
    NotImplementedError
        For grounded wye-delta banks that are the only zero-sequence sources of their part of
        the network (see above).
    """
    graph, graph_node_list, edge_list, root = check_network_validity(circuit_data)
    node_list = list(node_list) if node_list is not None else graph_node_list
    node2idx = {node: idx for idx, node in enumerate(node_list)}

    # node_list indexes every returned array and every lookup in the sweep, so it has to be
    # exactly the node set of the radial tree. An extra name would keep its initial voltage
    # for ever (zero on the v0 path -> NaN mismatches and converged=False, whatever V_init
    # happens to hold on the V_init path) and its p/q row would be silently ignored.
    graph_nodes = set(graph_node_list)
    unknown = [node for node in node_list if node not in graph_nodes]
    missing = [node for node in graph_node_list if node not in node2idx]
    if unknown or missing or len(node2idx) != len(node_list):
        raise ValueError(
            "node_list must contain exactly the nodes of the radial network built from "
            f"circuit_data ({len(graph_node_list)} nodes), once each; got {len(node_list)} "
            f"entries with unknown (off-tree) nodes {unknown[:10]} and missing network nodes "
            f"{missing[:10]}. Build it with check_network_validity(circuit_data), or pass "
            "node_list=None to use that ordering."
        )

    if isinstance(slack_bus, str):
        slack_idx = node2idx[slack_bus]
    else:
        slack_idx = int(slack_bus)
    slack_node = node_list[slack_idx]
    if slack_node != root:
        raise ValueError(f"slack_bus must point to the root node {root!r}, got {slack_node!r}")

    n_nodes = len(node_list)
    p = _reshape_phase_array(p, n_nodes, "p")
    q = _reshape_phase_array(q, n_nodes, "q")

    # The slack bus has no free injection: its current follows from the network and is
    # written back into I_bus[root] at the end of the sweep. Anything left in the slack row
    # of p/q would be added on top of that and break the I_bus = Y @ V identity, so drop it
    # (and copy first -- _reshape_phase_array may hand back a view of the caller's array).
    if np.any(p[slack_idx] != 0) or np.any(q[slack_idx] != 0):
        logger.warning(
            "Ignoring the non-zero slack-bus injection p[%s]=%s, q[%s]=%s: the slack current "
            "is determined by the network, not by p/q.",
            slack_idx, p[slack_idx], slack_idx, q[slack_idx],
        )
    p = p.copy()
    q = q.copy()
    p[slack_idx] = 0.0
    q[slack_idx] = 0.0

    circuit_dict = {(branch["source"], branch["target"]): branch for branch in circuit_data}
    edge_to_idx = {(branch["source"], branch["target"]): idx for idx, branch in enumerate(circuit_data)}

    if V_init is not None and v0 is not None:
        raise ValueError(
            "pass either V_init or v0, not both: only one of them can set the voltage scale, "
            "and v0 would be silently dropped. V_init[slack_bus] is itself the slack boundary "
            "condition, so to move the slack voltage of a warm start, overwrite that row."
        )
    if V_init is not None:
        V = _reshape_phase_array(V_init, n_nodes, "V_init").copy()
    elif v0 is not None:
        V = _nominal_initial_voltages(graph, node_list, node2idx, circuit_dict, root, v0)
        logger.debug("Initial guess from nominal propagation of v0, |V| range [%.4g, %.4g]",
                     float(np.min(np.abs(V))), float(np.max(np.abs(V))))
    else:
        raise ValueError(
            "solve_power_flow requires a voltage scale: pass V_init (an (n, 3) or (3n,) "
            "complex initial guess) or v0 (the root/slack voltage, either a scalar "
            "line-to-neutral magnitude or a (3,) complex phasor). A flat 1.0 V start is "
            "not a usable default for a network modelled in volts -- it diverges."
        )
    # The sweep divides by V, so a zero (or non-finite) start yields NaN everywhere and can
    # never converge. Checking the assembled guess covers scalar v0, (3,) v0 and V_init alike.
    if not np.all(np.isfinite(V)):
        raise ValueError(
            "the initial voltages contain NaN or Inf; check V_init / v0 "
            f"(first offending node index {int(np.argwhere(~np.isfinite(V))[0, 0])})"
        )
    if np.any(V == 0):
        zero_nodes = sorted({int(idx) for idx in np.argwhere(V == 0)[:, 0]})
        raise ValueError(
            "the initial voltages must be non-zero on every phase of every node, but nodes "
            f"{[node_list[idx] for idx in zero_nodes[:10]]} (indices {zero_nodes[:10]}) start "
            "at zero. The sweep divides by V, so a zero start gives NaN, not a missing phase. "
            "Check V_init / v0."
        )
    V_max = np.max(np.abs(V))

    # Pre-compute branch parameters and their inverses
    branch_params = {}
    z_ij_s = {}
    z_hat_s = {}
    # For a transformer with a floating wye neutral, the projection that drops the mean over
    # its windings (see _zero_sum_projection); None otherwise.
    zero_sum = {}
    for edge in edge_list:
        branch_data = circuit_dict[edge]
        branch_params[edge] = four_parameter_branch_model(branch_data)
        z_ij_s[edge] = np.linalg.pinv(branch_params[edge][0])
        z_hat_s[edge] = convert_series_impedance_3x3(branch_data['series_impedance'])
        floating = floating_neutrals(branch_data)
        zero_sum[edge] = _zero_sum_projection(branch_data) if any(floating) else None
        # The circulating current of a grounded wye-delta bank is the primary zero-sequence
        # voltage over its leakage impedance (see below), so it needs one.
        if branch_data["element_type"] in ("YD", "YD_alt") and not floating[0] and np.sum(z_hat_s[edge]) == 0:
            raise ValueError(
                f"Wye-delta transformer {branch_data.get('element_name', edge)} has zero series "
                "impedance; the power flow models its wye winding as grounded, which needs a "
                "non-zero leakage impedance."
            )
        # The sweep models a wye-delta element as a three-phase bank. A lone single-phase
        # unit has no windings on the other two phases, but would be given currents on them.
        # (No phases, as from a bare bus name, means all three.)
        primary_phases = (branch_data.get("phases") or [()])[0]
        if branch_data["element_type"] in ("YD", "YD_alt") and set(primary_phases) - {0} not in (set(), {1, 2, 3}):
            raise ValueError(
                f"Wye-delta transformer {branch_data.get('element_name', edge)} has primary "
                f"phases {primary_phases}; the power flow only models three-phase wye-delta banks."
            )

    bfs_tree = nx.bfs_tree(graph, root)
    parents = {child: parent for parent, child in bfs_tree.edges}
    traversal = list(bfs_tree.nodes())

    # A grounded wye-delta bank draws a current c circulating in its delta, set by the
    # zero-sequence voltage at its primary bus, i.e. it is a stiff zero-sequence shunt.
    # Updated from the previous sweep's voltage alone, c diverges once the bank's
    # zero-sequence admittance times the upstream zero-sequence impedance exceeds one, and
    # so does a set of banks that share upstream impedance even when each bank alone would
    # not. So, as in the compensation method of Shirmohammadi et al. (1988), all c are
    # updated together after each forward sweep, by a Newton step on the zero-sequence
    # impedance matrix between the banks' primary buses. This changes the iteration only, not
    # the converged solution. With a floating neutral, c is zero.
    yd_edges = [edge for edge in edge_list
                if circuit_dict[edge]["element_type"] in ("YD", "YD_alt") and zero_sum[edge] is None]
    # Below a delta winding or a floating neutral, the sweep drops the zero-sequence current
    # delivered and fixes the zero-sequence voltage (to none). A grounded wye-delta bank that
    # is the only zero-sequence source of such a part (a grounding bank on an ungrounded
    # system) would instead have to carry that current and set that voltage.
    for buses, sources in zero_sequence_groups(circuit_data):
        if sources and all(kind in ("YD", "YD_alt") for kind, _ in sources):
            raise NotImplementedError(
                f"Wye-delta transformer(s) {[circuit_dict[edge].get('element_name', edge) for _, edge in sources]} "
                "ground a part of the network that nothing else supplies zero-sequence current to "
                "(it lies below a delta winding or a floating wye neutral); the power flow does not "
                "support such grounding banks")
    yd_index = {edge: i for i, edge in enumerate(yd_edges)}
    yd_circulating = np.zeros(len(yd_edges), dtype=complex)
    yd_I_internal = {}
    if yd_edges:
        z0_banks = _zero_sequence_impedance_matrix(yd_edges, traversal, parents, circuit_dict, z_hat_s, root)
        yd_newton = z0_banks + np.diag([np.sum(z_hat_s[edge]) for edge in yd_edges])

    # Variables
    I_bus = np.zeros((n_nodes, 3), dtype=complex)
    I_branch = np.zeros((len(circuit_data), 3), dtype=complex)
    converged = False

    I_bus_prev = I_bus.copy()
    I_bus = np.conj((p + 1j * q) / V)

    circulating_delta = 0.0  # no wye-delta banks: nothing else to converge
    i = -1
    voltage_delta = current_delta = np.inf
    for i in range(int(max_iter)):
        I_branch_internal = {}
        # Backward sweep: compute branch currents from leaf nodes to root
        for node in reversed(traversal):
            if node == root:
                continue
            parent = parents[node]
            node_idx = node2idx[node]
            parent_idx = node2idx[parent]
            y_jk_s, y_kj_s, y_jk_m, y_kj_m = branch_params[(parent, node)]
            # Sum currents and losses (17.43a)
            child_current = np.zeros(3, dtype=complex)
            for child in graph.successors(node):
                child_current += I_branch[edge_to_idx[(node, child)]]
            branch_data = circuit_data[edge_to_idx[(parent, node)]]
            if branch_data["element_type"] == "line":
                I_branch[edge_to_idx[(parent, node)]] = (
                    child_current
                    - I_bus[node_idx]
                    + y_jk_m @ V[parent_idx]
                    + y_kj_m @ V[node_idx]
                )
            # Note: For the special handling here, we assume (for simplicity), that no physical shunt is present.
            # With a floating wye neutral, the winding currents cannot have the part of the
            # current delivered below that does not sum to zero over the windings; as below a
            # delta winding, it is dropped (I = Y @ V then fails by it at the secondary).
            elif branch_data["element_type"] == "DY":
                I_jk_internal = (child_current - I_bus[node_idx]) / branch_data['turns_ratio']
                if zero_sum[(parent, node)] is not None:
                    I_jk_internal = zero_sum[(parent, node)] @ I_jk_internal
                I_branch[edge_to_idx[(parent, node)]] = Gamma.T @ I_jk_internal
                I_branch_internal[(parent, node)] = I_jk_internal
            elif branch_data["element_type"] == "DY_alt":
                I_jk_internal = (child_current - I_bus[node_idx]) / branch_data['turns_ratio']
                if zero_sum[(parent, node)] is not None:
                    I_jk_internal = zero_sum[(parent, node)] @ I_jk_internal
                I_branch[edge_to_idx[(parent, node)]] = Gamma @ I_jk_internal
                I_branch_internal[(parent, node)] = I_jk_internal
            elif branch_data["element_type"] in ("YY", "openY_openY"):
                I_jk_internal = (child_current - I_bus[node_idx]) / branch_data['turns_ratio']
                if zero_sum[(parent, node)] is not None:
                    I_jk_internal = zero_sum[(parent, node)] @ I_jk_internal
                I_branch[edge_to_idx[(parent, node)]] = I_jk_internal
                I_branch_internal[(parent, node)] = I_jk_internal
            elif branch_data["element_type"] == "DD":
                I_jk_internal = (1/3) * Gamma @ (child_current - I_bus[node_idx]) / branch_data['turns_ratio']
                I_branch[edge_to_idx[(parent, node)]] = Gamma.T @ I_jk_internal
                I_branch_internal[(parent, node)] = I_jk_internal
            elif branch_data["element_type"] in ("YD", "YD_alt"):
                # Wye primary, delta secondary. The secondary line currents fix the winding
                # currents only up to the current c circulating in the delta: for a grounded
                # wye, c is updated after the forward sweep (see above); for a floating one,
                # it is zero. As for DD, a zero-sequence part of the current delivered below
                # cannot flow in the delta and is dropped (I = Y @ V then fails by it at the
                # secondary).
                Gamma_w = Gamma if branch_data["element_type"] == "YD" else Gamma.T
                I_jk_internal = (1/3) * Gamma_w @ (child_current - I_bus[node_idx]) / branch_data['turns_ratio']
                if (parent, node) in yd_index:
                    yd_I_internal[(parent, node)] = I_jk_internal
                    I_jk_internal = I_jk_internal + yd_circulating[yd_index[(parent, node)]]
                I_branch[edge_to_idx[(parent, node)]] = I_jk_internal
                I_branch_internal[(parent, node)] = I_jk_internal
            else:
                raise ValueError(f"Un-implemented branch type {branch_data['element_type']} for branch {(parent, node)}")

        # Forward sweep: compute voltages from root to leaf nodes
        V_prev = V.copy()
        for node in traversal:
            if node == root:
                continue
            parent = parents[node]
            parent_idx = node2idx[parent]
            node_idx = node2idx[node]
            branch_data = circuit_data[edge_to_idx[(parent, node)]]
            if branch_data["element_type"] == "line":
                _, _, y_jk_m, _ = branch_params[(parent, node)]
                branch_current = I_branch[edge_to_idx[(parent, node)]]
                V[node_idx] = V[parent_idx] + z_ij_s[(parent, node)] @ (
                    y_jk_m @ V[parent_idx] - branch_current
                )
            # Note: Again, for internal variable method, we ignore any physical shunt.
            elif branch_data["element_type"] == "DY":
                V[node_idx] = (Gamma @ V[parent_idx] - z_hat_s[(parent, node)] @ I_branch_internal[(parent, node)]) / branch_data['turns_ratio']
            elif branch_data["element_type"] == "DY_alt":
                V[node_idx] = (Gamma.T @ V[parent_idx] - z_hat_s[(parent, node)] @ I_branch_internal[(parent, node)]) / branch_data['turns_ratio']
            elif branch_data["element_type"] in ("YY", "openY_openY"):
                V[node_idx] = (V[parent_idx] - z_hat_s[(parent, node)] @ I_branch_internal[(parent, node)]) / branch_data['turns_ratio']
            elif branch_data["element_type"] == "DD":
                V[node_idx] = (1/3) * Gamma.T @ (Gamma @ V[parent_idx] - z_hat_s[(parent, node)] @ I_branch_internal[(parent, node)]) / branch_data['turns_ratio']
            elif branch_data["element_type"] in ("YD", "YD_alt"):
                # turns_ratio * (Gamma or Gamma.T) @ V[node] = V_parent - z @ I_jk_internal; like
                # DD, the delta secondary has no zero-sequence reference and gets none.
                Gamma_w = Gamma if branch_data["element_type"] == "YD" else Gamma.T
                V[node_idx] = (1/3) * Gamma_w.T @ (V[parent_idx] - z_hat_s[(parent, node)] @ I_branch_internal[(parent, node)]) / branch_data['turns_ratio']
            else:
                raise ValueError(f"Un-implemented branch type {branch_data['element_type']} for branch {(parent, node)}")
            # A floating neutral leaves the zero-sequence voltage across the bank free, so like a
            # delta secondary, the secondary of a wye-wye or delta-wye bank with one gets none
            # (on the windings present). A wye-delta bank's secondary is a delta anyway.
            if branch_data["element_type"] in ("DY", "DY_alt", "YY", "openY_openY") and zero_sum[(parent, node)] is not None:
                V[node_idx] = zero_sum[(parent, node)] @ V[node_idx]

        if yd_edges:
            # c makes the primary windings absorb the primary zero-sequence voltage:
            # sum(V_parent - z @ (I_jk_internal + c)) = 0, as the delta voltages (the left-hand
            # side of the forward sweep) sum to zero. A step dc lowers sum(V_parent) of every
            # bank by z0_banks @ dc and raises sum(z) * c, so take the Newton step on both.
            residual = np.array([
                np.sum(V[node2idx[edge[0]]]) - np.sum(z_hat_s[edge] @ yd_I_internal[edge])
                - np.sum(z_hat_s[edge]) * yd_circulating[k]
                for k, edge in enumerate(yd_edges)
            ])
            circulating_step = np.linalg.solve(yd_newton, residual)
            yd_circulating = yd_circulating + circulating_step
            # Part of the stopping test: the sweep that set V and I_bus used the old c. Measured
            # as the voltage mismatch it leaves, which has the same roundoff floor as the voltage
            # test (the step itself does not, when c is zero).
            circulating_delta = np.max(np.abs(residual)) / np.max(np.abs(V))

        I_bus_prev = I_bus.copy()
        I_bus = np.conj((p + 1j * q) / V)

        voltage_delta = np.max(np.abs(V - V_prev) / np.maximum(1e-12, np.abs(V)))
        current_delta = np.max(np.abs(I_bus - I_bus_prev) / np.maximum(1e-12, np.abs(I_bus)))
        converged = (voltage_delta < rtol) and (current_delta < rtol) and (circulating_delta < rtol)
        # Lazy %-formatting: this fires up to max_iter (default 1e5) times, so the
        # message must not be built when the record is discarded.
        logger.debug("BFS iter %d: voltage delta=%.3e %%, current delta=%.3e %%, "
                     "wye-delta circulating delta=%.3e %%, converged=%s",
                     i, voltage_delta * 100, current_delta * 100, circulating_delta * 100, converged)
        if converged:
            break
        V_scale = np.max(np.abs(V))
        if not np.isfinite(V_scale):
            # NaN compares False against every threshold, so without this the sweep would
            # grind through all max_iter iterations and return NaNs with converged=False.
            raise PowerFlowDivergedError(
                f"Power flow diverged at iteration {i}: the voltages are no longer finite. "
                "Check p/q for NaN/Inf and the initial guess (V_init / v0)."
            )
        if V_scale / V_max > 1e5:
            raise PowerFlowDivergedError(
                f"Power flow diverged at iteration {i}: |V| grew by more than 1e5x relative to "
                "the initial guess. Check the initial guess (V_init / v0) and the load scaling."
            )

    if converged:
        logger.info("BFS power flow converged in %d iterations (rtol=%g)", i + 1, rtol)
    else:
        logger.warning(
            "BFS power flow did NOT converge in %d iterations: voltage delta=%.3e, "
            "current delta=%.3e, wye-delta circulating delta=%.3e, rtol=%g",
            int(max_iter), voltage_delta, current_delta, circulating_delta, rtol
        )

    # The slack current is set by the network rather than by p/q (whose slack row was zeroed
    # above), so I_bus at the root is the sum of the currents leaving it. This is what makes
    # the returned solution satisfy I_bus = Y @ V at the root as well.
    root_idx = node2idx[root]
    outgoing_sum = np.zeros(3, dtype=complex)
    for child in graph.successors(root):
        outgoing_sum += I_branch[edge_to_idx[(root, child)]]
    I_bus[root_idx] += outgoing_sum

    return V.reshape(-1), I_bus.reshape(-1), I_branch.reshape(-1), converged


def generate_dataset(circuit_data, p_samples, q_samples, slack_bus, node_list, V_init=None,
                     rtol=1e-6, max_iter=1e3, v0=None, on_failure="warn"):
    """
    Generate a synthetic dataset of (p, q, V, I) tuples.

    Parameters
    ----------
    circuit_data : list of branch dicts describing the (radial) circuit
    p_samples : (N, 3n) or (N, n, 3) array of active power injection scenarios
    q_samples : (N, 3n) or (N, n, 3) array of reactive power injection scenarios
    slack_bus, node_list, V_init, rtol, max_iter, v0 : passed through to
        :func:`solve_power_flow`; as there, exactly one of ``V_init`` / ``v0`` is
        required (passing both, or neither, raises ``ValueError``).
    on_failure : {'warn', 'raise', 'skip'}, default 'warn'
        What to do with scenarios that fail to converge (or diverge):
        ``'warn'`` drops them and logs a warning naming how many were dropped,
        ``'raise'`` re-raises / raises ``RuntimeError`` on the first failure,
        ``'skip'`` drops them quietly (logged at debug level only).

    Returns
    -------
    dict with arrays ``p``, ``q``, ``V``, ``I_bus``, ``I_branch`` holding only the
    scenarios that converged, plus:
    ``scenario_index`` : (K,) int indices into ``p_samples`` of the rows kept,
    ``failed_index``   : (N-K,) int indices of the rows dropped,
    ``n_failed``       : int, number of dropped scenarios.
    Because non-converged scenarios are dropped, the leading dimension K can be
    shorter than ``N``; use ``scenario_index`` to line the results back up with
    the inputs. ``p`` and ``q`` keep the row shape and dtype of ``p_samples`` /
    ``q_samples``; ``V`` and ``I_bus`` are (K, 3n) complex and ``I_branch`` is
    (K, 3m) complex, with n = len(node_list) and m = len(circuit_data). Those
    widths and dtypes hold for K = 0 too, so an all-failing call still stacks and
    reshapes like a successful one.
    """
    if on_failure not in ("warn", "raise", "skip"):
        raise ValueError(f"on_failure must be one of 'warn', 'raise', 'skip'; got {on_failure!r}")

    records = []
    kept = []
    failed = []
    for scenario, (p, q) in enumerate(zip(p_samples, q_samples)):
        try:
            V, I_bus, I_branch, ok = solve_power_flow(
                circuit_data, p, q, slack_bus, node_list, V_init, rtol, max_iter, v0=v0
            )
        except PowerFlowDivergedError as exc:
            if on_failure == "raise":
                raise
            logger.debug("Scenario %d diverged: %s", scenario, exc)
            ok = False

        if ok:
            kept.append(scenario)
            records.append({
                'p': p, 'q': q,
                'V': V, 'I_bus': I_bus, 'I_branch': I_branch
            })
            continue

        failed.append(scenario)
        if on_failure == "raise":
            raise RuntimeError(
                f"Power flow did not converge for scenario {scenario} "
                f"(rtol={rtol}, max_iter={int(max_iter)})."
            )

    if failed:
        n_total = len(kept) + len(failed)
        if on_failure == "warn":
            logger.warning(
                "generate_dataset dropped %d of %d scenarios that did not converge "
                "(first dropped indices: %s); see 'failed_index' in the returned dict",
                len(failed), n_total, failed[:10]
            )
        else:
            logger.debug("generate_dataset dropped %d of %d non-converged scenarios",
                         len(failed), n_total)

    if records:
        stacked = {key: np.array([record[key] for record in records])
                   for key in ('p', 'q', 'V', 'I_bus', 'I_branch')}
    else:
        # No scenario survived. np.array([]) would collapse to a (0,) float64 array, whose
        # rank and dtype differ from the non-empty case, so anything the caller does next
        # (result['V'].shape[1], reshape, np.concatenate over chunks) breaks. Build the
        # empty arrays with the shapes and dtypes the non-empty path would have produced.
        p_rows = np.asarray(p_samples)
        q_rows = np.asarray(q_samples)
        n_nodes = (len(node_list) if node_list is not None
                   else len(check_network_validity(circuit_data)[1]))
        stacked = {
            'p': np.empty((0, *p_rows.shape[1:]), dtype=p_rows.dtype),
            'q': np.empty((0, *q_rows.shape[1:]), dtype=q_rows.dtype),
            'V': np.empty((0, 3 * n_nodes), dtype=complex),
            'I_bus': np.empty((0, 3 * n_nodes), dtype=complex),
            'I_branch': np.empty((0, 3 * len(circuit_data)), dtype=complex),
        }

    dataset = {
        **stacked,
        'scenario_index': np.array(kept, dtype=int),
        'failed_index': np.array(failed, dtype=int),
        'n_failed': len(failed),
    }
    return dataset
