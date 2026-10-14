"""
Parse .dss files using the altdss python package.
Extract distribution network data (nodes, edges, attributes) from the DSS files.
Format the extracted data into our standard fomat for 3-phase LindistFlow.
"""

import csv
import logging
import os
import shutil
import tempfile
import warnings
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import re
from pathlib import Path
from difflib import SequenceMatcher
import copy
import numpy as np
import networkx as nx
from altdss import altdss

from .topology import build_node_data, to_edge_list
from .plotting import plot_nx_graph

logger = logging.getLogger(__name__)

# Suffix that some utility datasets append to secondary (low-voltage) conductor names in
# their WireData/CNData catalogs. Used only as a tie-breaker when matching a LineCode to a
# conductor; override for a dataset that uses a different naming convention.
SECONDARY_CONDUCTOR_SUFFIX = "_d1_sec"

# Name of the throwaway master script this parser synthesizes. It is written to a fresh
# temporary directory on every parse, never into the dataset directory.
AUTOGEN_MASTER_NAME = "autogen_master.dss"

# Where optional diagnostics (cycle reports, network plots) go when the caller does not
# specify an ``output_dir``. Nothing is written here unless there is something to write.
DEFAULT_OUTPUT_DIR = Path(tempfile.gettempdir()) / "three_phase_lindistflow"


def _enabled(element):
    """Whether the circuit element of this full name (Class.name) is enabled.

    OpenDSS leaves a disabled element out of the circuit, as if it were not defined, and so
    does DataLoader.
    """
    circuit = altdss.to_dss_python().ActiveCircuit
    circuit.SetActiveElement(element)
    return bool(circuit.ActiveCktElement.Enabled)


def _orient_from_source(circuit_data, root, source):
    """Turn around the lines below the feeder root that are defined from their load side.

    The feeder is the root's subtree, taken by following the branches from the root in their
    bus1 -> bus2 (winding 1 -> winding 2) direction. Seen from the voltage source (or from the
    root, if the source is not connected to it), the buses below the root are those whose
    path from the source passes through the root. A branch from a bus below the root that the
    direction does not reach into one that it does is defined from its load side, and would
    cut off what lies behind it: a line (or reactor) is turned around, which changes nothing
    else, since its impedance is the same either way; a transformer raises, since turning it
    around would swap its windings. Branches the direction already reaches, ties included,
    are left as they are.
    """
    graph = nx.Graph([(e['source'], e['target']) for e in circuit_data])
    if root not in graph:
        return
    origin = source if source in nx.node_connected_component(graph, root) else root
    below = {root} | nx.descendants(nx.bfs_tree(graph, origin), root)
    turned = 0
    while True:
        reached = set(nx.bfs_tree(nx.DiGraph([(e['source'], e['target']) for e in circuit_data]), root))
        reversed_branches = [e for e in circuit_data
                             if e['target'] in reached and e['source'] not in reached and e['source'] in below]
        if not reversed_branches:
            break
        for e in reversed_branches:
            if e['element_type'] != 'line':
                raise NotImplementedError(
                    f"Transformer {e['element_name']} has winding 1 on bus {e['source']}, which is on the "
                    f"load side of its other winding's bus {e['target']} (seen from the voltage source); "
                    "define winding 1 on the source side")
            e['source'], e['target'] = e['target'], e['source']
        turned += len(reversed_branches)
    if turned:
        logger.info("Turned around %d line(s) defined from their load side.", turned)


def _check_like_windings_on_like_nodes(xfmr):
    """Raise if a wye-wye or delta-delta winding pair has windings on different nodes.

    The models read the nodes of a transformer winding as a set, and connect phase p of one
    such winding to phase p of the other, with the same polarity. So the other winding must
    be on the same nodes in the same order: written the other way round (bus.0.1 against
    bus.1.0, bus.2.1 against bus.1.2), it is reversed, which shifts it by 180 degrees; on
    other phases, or in another order (bus.2.3.1 against bus.1.2.3), it shifts the phases.
    The exception is a split-phase (center-tap) unit: a single-phase transformer of three
    wye windings, whose second winding is the first half of the secondary, and its third
    winding the second half, reversed (bus.1.0 and bus.0.2). Which half is in phase with the
    primary is not checked: like the orientation of a delta winding in an open bank, it is a
    sign the open and split-phase models ignore (see build_permutation_matrix_open_split).
    Delta-wye pairs are checked in to_circuit_data.

    Neither shift is modeled, so it is rejected rather than ignored. These checks can be
    removed once the models take it: the phases moved by a permutation of the secondary
    side, a reversed winding by a negative turns ratio.
    """
    n, conns = xfmr['num_phases'], xfmr['winding_connections']

    def conductors(w):
        # As OpenDSS connects them: a node left out takes its position's default, phase k
        # or, for the last conductor, node 0 (a bare bus.2 is bus.2.0, a bare bus 1.2.3.0).
        # The fourth conductor of a three-phase delta winding connects nothing.
        nodes = list(range(1, n + 1)) + [0]
        nodes[:len(xfmr['phases'][w])] = xfmr['phases'][w]
        return tuple(nodes[:3] if conns[w] == 'delta' and n == 3 else nodes)

    def reversed_(w):
        return conductors(w)[0] == 0

    if n == 1 and xfmr['windings'] == 3 and set(conns) == {'wye'}:
        if reversed_(1) != reversed_(0) or reversed_(2) == reversed_(1):
            raise ValueError(
                f"Transformer {xfmr['name']}: a split-phase unit must have its second winding of the "
                f"polarity of its first and its third winding reversed (as bus.1.0 and bus.0.2), got "
                f"windings on nodes {[conductors(w) for w in range(3)]}, which is not supported")
        return
    for t in range(1, xfmr['windings']):
        first, other = conductors(0), conductors(t)
        if conns[t] != conns[0] or other == first:
            continue
        if other == first[::-1] or [p for p in other if p] == [p for p in first if p]:
            problem = "opposite polarity"
        else:
            problem = "different phases or a different phase order"
        raise ValueError(
            f"Transformer {xfmr['name']}: its {conns[0]} windings on nodes {first} and {other} "
            f"have {problem}, which is not supported")


class DataLoader:
    def __init__(self, base_path: str, output_dir: Optional[str] = None):
        """
        Initialize DataLoader for a specific feeder using altdss-python

        Args:
            base_path: Path to the base directory of the DSS files. This directory is
                only ever read from; parsing never writes to it.
            output_dir: Directory for optional diagnostics (cycle reports, network
                plots). Defaults to :data:`DEFAULT_OUTPUT_DIR` under the system temp
                directory. The directory is created lazily, only if something is written.
        """
        self.base_path = Path(base_path).resolve()
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        self.name = None
        self.dss_data = {}
        self.circuit_data = None
        self.node_data = None
        self.load_profiles = {}
        self.pv_profiles = {}

        if not os.path.exists(self.base_path):
            raise ValueError(f"Invalid input data path: {self.base_path}")

    def parse(self) -> Dict[str, Any]:
        """
        Execute .dss files using Master.dss in self.base_path.
        Parse the feeder data using altdss-python and extract relevant parameters
        Returns:
            Dictionary containing parsed network data
        """
        # Check folder/file existence
        if not self.base_path.exists():
            raise FileNotFoundError(f"DSS file folder not found at {self.base_path}")
        # Accept several casing variants for Master.dss (some datasets use Master.DSS)
        master_file = None
        for name in ("Master.dss", "Master.DSS", "master.dss", "master.DSS"):
            p = self.base_path / name
            if p.exists():
                master_file = p
                break
        if master_file is None:
            # Try a case-insensitive search as a last resort
            for p in self.base_path.iterdir():
                if p.name.lower() == "master.dss":
                    master_file = p
                    break
        if master_file is None:
            raise FileNotFoundError(f"Master.dss not found in {self.base_path}")
        # node coordinates may be provided as nodecoords.csv or BusCoords.dss (case-insensitive)
        csv_path = self.base_path / "nodecoords.csv"
        buscoords_path = None
        for name in ("BusCoords.dss", "Buscoords.dss", "buscoords.dss"):
            p = self.base_path / name
            if p.exists():
                buscoords_path = p
                break
        if not csv_path.exists() and (buscoords_path is None):
            # don't fail here; coordinates are optional. Warn and continue.
            logger.warning(
                "No node coordinate file found at %s (expected nodecoords.csv or "
                "BusCoords.dss). Continuing without coordinates.",
                self.base_path,
            )
        self.name = master_file.name

        # Build a minimal autogenerated master file that redirects the dataset's DSS files
        # by ABSOLUTE path. Using absolute paths we resolve ourselves means we never have to
        # patch up the filename casing of the dataset's own Redirect statements, and the
        # master can live outside the dataset directory.
        # (This also avoids executing dataset-specific experimental commands present in the
        # shipped Master.DSS.)
        dss_files_raw = []
        # Collect both lowercase and uppercase extensions to handle datasets
        # that use inconsistent casing (e.g., .dss and .DSS)
        # Collect DSS script files but skip dataset run scripts (Run_*.dss)
        p_list = [p for p in sorted(self.base_path.rglob("*.dss")) + sorted(self.base_path.rglob("*.DSS"))
              if not p.name.lower().startswith("run_")]
        # Deduplicate while preserving order
        seen = set()
        p_list_uniq = []
        for p in p_list:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            p_list_uniq.append(p)
        for p in p_list_uniq:
            # skip the original master and any stale autogen file from an older version
            if p.name.lower() == master_file.name.lower():
                continue
            if p.name.lower() == AUTOGEN_MASTER_NAME:
                continue
            # Skip device/control helper files that may rely on dataset-specific ordering
            if any(x in p.name.lower() for x in ("device", "regcontrol", "swt", "energy")):
                continue
            # Skip explicit BusCoords files; we'll parse them separately and they
            # are not valid DSS command scripts for Redirect.
            if buscoords_path is not None and p.name.lower() == buscoords_path.name.lower():
                continue
            # store the resolved absolute path (this is what fixes filename-case
            # mismatches without touching the dataset directory)
            dss_files_raw.append(p.resolve())

        # Ensure definition files (linecodes, wiredata, cndata, bus voltage bases, coords) come first
        priority_names = ("linecode.dss", "linegeometry.dss", "wiredata.dss", "cndata.dss", "busvoltagebases.dss", "buscoords.dss", "setbasekv.dss")
        dss_files = [q for q in dss_files_raw if q.name.lower() in priority_names] + [q for q in dss_files_raw if q.name.lower() not in priority_names]

        # Write the autogen master to a throwaway temporary directory so that parsing has no
        # side effects on the user's data directory.
        original_dir = os.getcwd()
        tmp_dir = tempfile.mkdtemp(prefix="three_phase_lindistflow_")
        try:
            autogen_master = Path(tmp_dir) / AUTOGEN_MASTER_NAME
            with open(autogen_master, "w") as f:
                f.write("Clear\n")
                f.write("New Circuit.autogen\n")
                for q in dss_files:
                    f.write(f'Redirect "{q.as_posix()}"\n')
                # Do not inject Buscoords here; some datasets provide comma-separated
                # coordinate files that OpenDSS treats as commands and will error.
                f.write("CalcVoltageBases\n")
                f.write("calcv\n")
                f.write("calcincmatrix_o\n")

            # Compile from the feeder directory so relative paths inside the dataset's own
            # scripts resolve the way they natively would. OpenDSS itself moves the process
            # working directory while compiling, so the restore below must be unconditional.
            os.chdir(self.base_path)
            # Clear any existing circuit and compile the autogenerated master file
            altdss("Clear")
            altdss(f'Compile "{autogen_master.as_posix()}"')
            # OpenDSS lists disabled elements as isolated too; they are left out below.
            isolated = [name for name in altdss.Topology.AllIsolatedBranches if name and name != 'NONE']
            assert all(not _enabled(name) for name in isolated), isolated
            # Note that altdss.Line[0].NormAmps is not correct.

            # Check if compilation was successful
            if altdss.Error.Number != 0:
                raise RuntimeError(f"AltDSS compilation error: {altdss.Error.Description}")
            # Use the ideal source model so SystemY comparisons reflect only the network elements.
            if len(altdss.Vsource) > 0:
                altdss.Vsource[0].edit(Model=1)
                logger.info("Set Vsource[0] to ideal source model for accurate SystemY extraction.")
        finally:
            # Compiling from the temporary directory leaves OpenDSS's DataPath pointing at
            # tmp_dir, which is about to be deleted. Any later file-writing DSS command
            # (Export/Show) would then hard-fail, and Save would silently re-create the
            # deleted temp directory and leak it. Point DataPath back at the caller's
            # working directory: it exists, it is where a plain OpenDSS run would write,
            # and it is never the dataset directory (which we only ever read from).
            try:
                altdss(f'Set DataPath="{Path(original_dir).as_posix()}"')
            except Exception as exc:  # pragma: no cover - diagnostics only
                logger.debug("Could not reset OpenDSS DataPath to %s: %s", original_dir, exc)
            # Always restore the caller's working directory, then drop the temporary master.
            os.chdir(original_dir)
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # Extract all network data using AltDSS interface
        dss_data = {
            "base_path": self.base_path,
            "circuit_name": altdss.Name,
            "buses": self._extract_buses(),
            "lines": self._extract_lines(),
            "transformers": self._extract_transformers(),
            "reactors": self._extract_reactors(),
            "swt_control": self._extract_swtcontrol(),
            "loads": self._extract_loads(),
            "capacitors": self._extract_capacitors(),
            "wire_data": self._extract_wire_data(),
            "cn_data": self._extract_cn_data(),
            # "line_geometry": self._extract_line_geometry(),
            "voltage_source": altdss.Vsource.BasekV.to_array()[0],
            # The bus of the circuit's voltage source, which to_circuit_data orients branches from.
            "source_bus": altdss.Vsource[0].Bus1.split(".")[0].lower(),
        }

        # Parse bus coordinates if available (support CSV or OpenDSS BusCoords.dss)
        if csv_path.exists():
            dss_data["bus_coordinates"] = self._parse_coordinates(csv_path)
        elif buscoords_path is not None and buscoords_path.exists():
            dss_data["bus_coordinates"] = self._parse_coordinates(buscoords_path)

        # Store the parsed data
        self.dss_data = dss_data
        return dss_data

    def _extract_buses(self) -> Dict[str, Any]:
        """Extract bus information using AltDSS interface"""
        buses = {}

        # Get all bus names
        logger.debug("Loading %d buses", len(altdss.Bus))
        for bus in altdss.Bus:
            buses[bus.Name] = {
                "name": bus.Name,
                "base_kv": bus.kVBase,
                "nodes": bus.Nodes.tolist(),
                "coordinate": (bus.X, bus.Y),
            }
        return buses

    def _split_bus_and_phases(self, bus_str, element=None):
        """
        Split an OpenDSS bus name like 'BusName.1.2.3' into:
        (base_name, [phase_numbers]), e.g. ('BusName', (1, 2, 3))

        Nodes are read as OpenDSS reads them (04 is node 4). ``element`` (e.g. 'Line.l1')
        names the element in the error raised for a node other than 0-3.
        """
        base_name, *nodes = bus_str.split(".")
        if not base_name or re.search(r"\s", base_name) or not all(re.fullmatch(r"[0-9]+", n) for n in nodes):
            raise ValueError(f"Invalid bus string format: {bus_str}")
        phases = tuple(int(n) for n in nodes)
        spare = [p for p in phases if p > 3]
        if spare:
            # The models have three phases and ground: a fourth wire (a neutral run between
            # buses, say) is not modeled. A transformer's floating wye neutral is taken off
            # before this (see _extract_transformers).
            raise NotImplementedError(
                f"{element + ' on ' if element else ''}{bus_str}: a conductor on node {spare[0]} (only "
                "nodes 1-3 and 0, ground, are modeled; a spare node only as the floating neutral of a wye winding)")
        return base_name, phases

    def _extract_lines(self) -> Dict[str, Any]:
        """Extract line information using AltDSS interface"""
        lines = {}

        logger.debug("Loading %d lines", len(altdss.Line))
        for line in altdss.Line:
            if not line.Enabled:
                continue
            # LineCode may be None for some legacy or custom lines; guard access.
            lc = line.LineCode if hasattr(line, 'LineCode') else None
            line_code_name = None if lc is None else lc.Name
            # Read the matrices from the Line even when it has a LineCode: they are what
            # OpenDSS builds the line's YPrim from, per unit of the Line's own length units.
            # The LineCode's are per unit of the LineCode's units, which need not match
            # (IEEE 13-node: ohm/mi linecodes on lines given in ft), and for a single-phase
            # line given by sequence impedances OpenDSS uses Z1, not the LineCode's (2*Z1 + Z0)/3.
            RMatrix = getattr(line, 'RMatrix', None)
            XMatrix = getattr(line, 'XMatrix', None)
            CMatrix = getattr(line, 'CMatrix', None)
            # Normalize numpy scalar to array if necessary
            try:
                if RMatrix is not None and np.isscalar(RMatrix):
                    RMatrix = np.array([RMatrix])
                if XMatrix is not None and np.isscalar(XMatrix):
                    XMatrix = np.array([XMatrix])
            except Exception:
                pass

            # Try to capture a normative ampacity reported by DSS for the line
            norm_amps = None
            if hasattr(line, 'NormAmps'):
                try:
                    norm_amps = float(line.NormAmps)
                except Exception:
                    norm_amps = None

            lines[line.Name] = {
                "name": line.Name,
                "length": line.Length,
                "units": line.Units,
                "num_phases": line.Phases,
                "line_type": line.LineType_str,
                "line_code": line_code_name,
                "RMatrix": RMatrix,
                "XMatrix": XMatrix,
                "CMatrix": CMatrix,
            }
            bus1, phases1 = self._split_bus_and_phases(line.Bus1, f"Line.{line.Name}")
            bus2, phases2 = self._split_bus_and_phases(line.Bus2, f"Line.{line.Name}")
            # If bus strings do not include explicit phase suffixes, infer
            # the phases from the declared number of phases on the line.
            if len(phases1) == 0:
                phases1 = tuple(range(1, lines[line.Name]["num_phases"] + 1))
            if len(phases2) == 0:
                phases2 = tuple(range(1, lines[line.Name]["num_phases"] + 1))
            assert (len(phases1) == lines[line.Name]["num_phases"]) and (phases1 == phases2), \
                (phases1, phases2, lines[line.Name]["num_phases"])
            lines[line.Name]["bus1"] = bus1
            lines[line.Name]["bus2"] = bus2
            lines[line.Name]["phases"] = phases1
            lines[line.Name]["norm_amps"] = norm_amps
        return lines

    def _extract_transformers(self) -> Dict[str, Any]:
        """Extract transformer information using AltDSS interface"""
        transformers = {}
        grounded_spare_nodes = set()  # (bus, node) of the wye neutrals on a spare node that Rneut=0 grounds
        logger.debug("Loading %d transformers", len(altdss.Transformer))
        for xfmr in altdss.Transformer:  # iterable, like `for bus in altdss.Bus`
            if not xfmr.Enabled:
                continue
            transformers[xfmr.Name] = {
                "name": xfmr.Name,
                "xhl": xfmr.XHL,          # leakage reactance b/t high and low windings
                "windings": xfmr.Windings,
                "winding_connections": [c.name for c in xfmr.Conns],
                "num_phases": xfmr.Phases,
                # These are bundled per‑winding in AltDSS/DSS-Extensions
                "pcrRs": xfmr.pctRs.tolist(),       # %R per winding
                "kvs": xfmr.kVs.tolist(),
                "kvas": xfmr.kVAs.tolist(),
                # Per-unit tap of each winding. Not modeled: to_circuit_data warns if not all 1.
                "taps": xfmr.Taps.tolist(),
                # 'lag' (the ANSI default) or 'lead': the phase of the low-voltage side
                # relative to the high-voltage side in a three-phase delta-wye bank.
                "lead_lag": xfmr.LeadLag.name.lower(),
                "buses": [],
                "phases": [],
                # Per winding: whether it is a wye winding whose neutral floats, and the
                # (bus, node) that neutral is on (None if grounded, or for a delta winding).
                "floating_neutral": [],
                "neutral_nodes": [],
            }
            r_neutral, x_neutral = xfmr.RNeut.tolist(), xfmr.XNeut.tolist()
            for w, (bus, conn) in enumerate(zip(list(xfmr.Buses), transformers[xfmr.Name]["winding_connections"])):
                # A wye winding connects its neutral to node 0 (ground), unless its bus puts it
                # on a spare node (bus.1.2.3.4). That node floats: with the default Rneut=-1,
                # OpenDSS connects nothing else to it. Rneut=0 (and Xneut=0) grounds it
                # solidly, and any other Rneut/Xneut through an impedance, which is not
                # supported. Another element on the node is checked for below. A single-phase
                # unit of two windings may also be written the other way round (bus.4.1, like
                # bus.0.1), which reverses the winding (to_circuit_data checks its polarity
                # against the other winding's); its neutral conductor is then on the phase.
                bus_name, *nodes = bus.split(".")
                neutral = None
                spare = [k for k, node in enumerate(nodes) if node.isdigit() and int(node) > 3]
                neutral_positions = [len(nodes) - 1] + ([0] if xfmr.Phases == 1 and xfmr.Windings == 2 else [])
                on_phases = all(node.isdigit() and 1 <= int(node) <= 3 for k, node in enumerate(nodes) if k not in spare)
                if conn == "wye" and len(nodes) == xfmr.Phases + 1 and len(spare) == 1 and spare[0] in neutral_positions \
                        and on_phases:
                    spare_node = (bus_name.lower(), int(nodes[spare[0]]))
                    if r_neutral[w] < 0:
                        neutral = spare_node
                    elif spare[0] == 0:
                        raise NotImplementedError(
                            f"Transformer {xfmr.Name}: winding {w + 1}, written neutral first on bus {bus}, "
                            "has an Rneut, which would connect its phase conductor to ground")
                    elif r_neutral[w] != 0 or x_neutral[w] != 0:
                        raise NotImplementedError(
                            f"Transformer {xfmr.Name}: winding {w + 1} grounds its wye neutral through "
                            f"Rneut={r_neutral[w]:g}, Xneut={x_neutral[w]:g} ohm; only a solidly grounded "
                            "(node 0, or Rneut=0 and Xneut=0) or floating neutral is supported")
                    else:
                        grounded_spare_nodes.add(spare_node)
                    nodes[spare[0]] = "0"
                    bus = ".".join([bus_name] + nodes)
                bus_name, phases = self._split_bus_and_phases(bus, f"Transformer.{xfmr.Name}")
                transformers[xfmr.Name]["buses"].append(bus_name)
                transformers[xfmr.Name]["phases"].append(phases)
                transformers[xfmr.Name]["floating_neutral"].append(neutral is not None)
                transformers[xfmr.Name]["neutral_nodes"].append(neutral)

        # A spare node that one winding grounds (Rneut=0) grounds every neutral on it.
        for xfmr in transformers.values():
            for w, node in enumerate(xfmr["neutral_nodes"]):
                if node in grounded_spare_nodes:
                    xfmr["floating_neutral"][w], xfmr["neutral_nodes"][w] = False, None
        self._check_floating_neutral_nodes(transformers)
        return transformers

    def _check_floating_neutral_nodes(self, transformers):
        """Raise if anything but wye neutrals connects to the node of a floating neutral.

        A line or a reactor there would carry the neutral current elsewhere or ground it, a
        load or capacitor would put its own neutral on it, and a transformer winding that has
        a phase on it would ground it. Several neutrals may share the node: those of the units
        of a bank, which is checked when the units are consolidated into the bank. Meters,
        controls and fuses name the terminal they watch, but connect nothing, so only the
        elements that carry current count: power delivery and conversion elements, and the
        rest of OpenDSS's (sources, and faults, which can ground a neutral too), if enabled.
        """
        neutral_nodes = {node for xfmr in transformers.values() for node in xfmr["neutral_nodes"] if node}
        if not neutral_nodes:
            return
        own_neutrals = {f"transformer.{name.lower()}": xfmr["neutral_nodes"] for name, xfmr in transformers.items()}
        circuit = altdss.to_dss_python().ActiveCircuit
        conducting = [element.FullName() for elements in (altdss.PDElement, altdss.PCElement, altdss.Vsource,
                                                          altdss.Isource, altdss.Fault, altdss.GICsource)
                      for element in elements]
        for element in conducting:
            circuit.SetActiveElement(element)
            if not circuit.ActiveCktElement.Enabled:
                continue
            own = own_neutrals.get(element.lower())
            for terminal, bus in enumerate(circuit.ActiveCktElement.BusNames):
                bus_name, *nodes = bus.lower().split(".")
                numbers = {int(n) for n in nodes if n.isdigit()}  # (as OpenDSS reads them: 04 is 4)
                if own and own[terminal]:
                    numbers.discard(own[terminal][1])  # the winding's own neutral, on a spare node
                shared = sorted(node for node in neutral_nodes if node[0] == bus_name and node[1] in numbers)
                if shared:
                    raise NotImplementedError(
                        f"{element} connects to node {shared[0][1]} of bus {bus_name}, the floating "
                        "neutral of a wye winding; only a neutral that nothing but other wye neutrals "
                        "connects to is supported")

    def _extract_reactors(self) -> Dict[str, Any]:
        """Extract reactor information using AltDSS interface"""
        reactors = {}
        logger.debug("Loading %d reactors", len(altdss.Reactor))
        if len(altdss.Reactor):
            # The primitive admittances below are only computed with the system matrix.
            altdss.Solution.BuildYMatrix(1, 0)
        for reactor in altdss.Reactor:
            if not reactor.Enabled:
                continue
            reactors[reactor.Name] = {
                "name": reactor.Name,
                "num_phases": reactor.Phases,
                "R": reactor.R,
                "X": reactor.X,
                "enabled": reactor.Enabled,
            }
            bus1, phases1 = self._split_bus_and_phases(reactor.Bus1, f"Reactor.{reactor.Name}")
            bus2, phases2 = self._split_bus_and_phases(reactor.Bus2, f"Reactor.{reactor.Name}")
            # A bare bus name connects nodes 1..phases, as for lines.
            phases1 = phases1 or tuple(range(1, reactor.Phases + 1))
            phases2 = phases2 or tuple(range(1, reactor.Phases + 1))
            if bus2 == bus1 or set(phases2) == {0}:
                raise ValueError(f"Reactor {reactor.Name} is a shunt reactor; only series reactors are supported")
            assert (len(phases1) == reactors[reactor.Name]["num_phases"]) and (phases1 == phases2), \
                (phases1, phases2, reactors[reactor.Name]["num_phases"])
            reactors[reactor.Name]["bus1"] = bus1
            reactors[reactor.Name]["bus2"] = bus2
            reactors[reactor.Name]["phases"] = phases1
            # Series impedance matrix over its phases, read off the primitive admittance: R and X
            # describe only a reactor given by R/X (or Z), and report a default reactance for one
            # given by Z1/Z0 or Rmatrix/Xmatrix.
            n = reactor.Phases
            # YPrim comes flattened column by column.
            y_series = -np.asarray(reactor.YPrim(), dtype=complex).reshape(2 * n, 2 * n, order="F")[:n, n:]
            if not np.all(np.isfinite(y_series)):
                # OpenDSS has no admittance for a zero impedance; keep it zero, as given.
                reactors[reactor.Name]["Z"] = np.eye(n) * (reactor.R + 1j * reactor.X)
            else:
                try:
                    reactors[reactor.Name]["Z"] = np.linalg.inv(y_series)
                except np.linalg.LinAlgError:
                    raise ValueError(
                        f"Reactor {reactor.Name} has a singular series admittance, i.e. an open "
                        "conductor, which is not supported"
                    ) from None
        return reactors

    def _extract_swtcontrol(self) -> Dict[str, Any]:
        """Extract device information using AltDSS interface"""
        swt_ctrls = {}
        logger.debug("Loading %d switch controllers", len(altdss.SwtControl))
        for swt in altdss.SwtControl:
            # A disabled control does not switch its element; a disabled element is left out.
            if not (swt.Enabled and swt.SwitchedObj.Enabled):
                continue
            swt_ctrls[swt.Name] = {
                "name": swt.Name,
                "swt_obj": swt.SwitchedObj,
                "normal": swt.Normal
            }
        return swt_ctrls

    def _extract_loads(self) -> Dict[str, Any]:
        """Extract load information using AltDSS interface"""
        loads = {}

        logger.debug("Loading %d loads", len(altdss.Load))
        for load in altdss.Load:
            if not load.Enabled:
                continue
            loads[load.Name] = {
                "name": load.Name,
                "num_phases": load.Phases,
                "kw": load.kW,
                "kvar": load.kvar,
                "kv": load.kV,
                "pf": load.PF,
                "connection": load.Conn.name,
                "vmaxpu": load.VMaxpu,
                "vminpu": load.VMinpu,
            }
            bus_name, phases = self._split_bus_and_phases(load.Bus1, f"Load.{load.Name}")
            # Assumption: Delta-connected loads are AB, BC, CA in polarity
            phases = list(sorted(phases))
            if phases == [1, 3]:
                phases = [3, 1]
            loads[load.Name]["bus"] = bus_name
            loads[load.Name]["phases"] = phases

        return loads

    def _extract_capacitors(self) -> Dict[str, Any]:
        """Extract capacitor information using AltDSS interface"""
        capacitors = {}

        logger.debug("Loading %d capacitors", len(altdss.Capacitor))
        for cap in altdss.Capacitor:
            if not cap.Enabled:
                continue
            capacitors[cap.Name] = {
                "name": cap.Name,
                "num_phases": cap.Phases,
                "kvar": cap.kvar,
                "kv": cap.kV,
                "enabled": cap.Enabled,
                "connection": cap.Conn.name,
            }
            bus_name, phases = self._split_bus_and_phases(cap.Bus1, f"Capacitor.{cap.Name}")
            capacitors[cap.Name]["bus"] = bus_name
            capacitors[cap.Name]["phases"] = phases
            
        return capacitors

    def _extract_wire_data(self) -> Dict[str, Any]:
        """Extract wire data information"""
        wire_data = {}
        logger.debug("Loading %d wire data", len(altdss.WireData))
        for wire in altdss.WireData:
            wire_data[wire.Name] = {
                "name": wire.Name,
                "norm_amps": wire.NormAmps,
                "emerg_amps": wire.EmergAmps,
            }

        return wire_data
    
    def _extract_cn_data(self) -> Dict[str, Any]:
        """Extract circuit node information using AltDSS interface"""
        cn_data = {}

        logger.debug("Loading %d concentric neutral data", len(altdss.CNData))
        for cn in altdss.CNData:
            cn_data[cn.Name] = {
                "name": cn.Name,
                "norm_amps": cn.NormAmps,
            }

        return cn_data

    def _extract_line_geometry(self) -> Dict[str, Any]:
        """Extract line geometry information"""
        geometries = {}
        logger.debug("Loading %d line geometries", len(altdss.LineCode.Name))
        for data in altdss.LineCode:
            geometries[data.Name] = {
                "RMatrix": data.RMatrix,
                "XMatrix": data.XMatrix,
                "NPhases": data.NPhases
            }
        return geometries

    def _parse_coordinates(self, file_path: Path) -> Dict[str, Any]:
        """Parse node coordinates from CSV file"""
        try:
            # Support two formats: CSV with (name,x,y) or OpenDSS BusCoords.dss with lines like "NAME,x,y"
            if file_path.suffix.lower() in {".dss", ".txt"}:
                buses = {}
                with open(file_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("//") or line.startswith("#"): 
                            continue
                        parts = [p.strip() for p in re.split(r"[,\s]+", line) if p.strip()]
                        if len(parts) < 3:
                            continue
                        name = parts[0]
                        try:
                            x_coord = float(parts[1])
                            y_coord = float(parts[2])
                        except ValueError:
                            # Some BusCoords.dss use different separators; try splitting on comma only
                            parts = [p.strip() for p in line.split(",")]
                            if len(parts) >= 3:
                                name = parts[0]
                                x_coord = float(parts[1])
                                y_coord = float(parts[2])
                            else:
                                continue
                        buses[name.lower()] = {"x": x_coord, "y": y_coord}
                return buses
            else:
                # Headerless CSV: name, x, y
                buses = {}
                with open(file_path, "r", newline="") as f:
                    for row in csv.reader(f):
                        if len(row) < 3:
                            continue
                        buses[str(row[0]).strip().lower()] = {
                            "x": float(row[1]),
                            "y": float(row[2]),
                        }
                return buses
        except Exception as e:
            logger.warning("Could not parse coordinates: %s", e)
            return {}

    def get_load_info(self,):
        """
        Flatten the parsed loads into one entry per load connection (leg).

        Returns:
            list of dicts with keys:
                - bus: str, the bus the load is connected to
                - phases: tuple of two ints, the pair of nodes the leg is connected
                  across: ``(p, 0)`` for a phase-to-neutral (wye) leg and ``(p, q)``
                  for a phase-to-phase (delta) leg.
            A single-phase load contributes one entry; a three-phase load contributes
            three (assumed balanced, since the DSS files do not specify otherwise).
        """
        if not self.dss_data:
            self.dss_data = self.parse()
        load_info = []
        for load in self.dss_data['loads'].values():
            num_phases = load['num_phases']
            phases = [p for p in load['phases'] if p != 0]
            # A bus string without an explicit node suffix (e.g. "bus3" rather than
            # "bus3.1.2.3") means OpenDSS connects the load to nodes 1..num_phases.
            if not phases:
                phases = list(range(1, num_phases + 1))
            if num_phases == 1:
                if len(phases) == 1:
                    p = phases[0]
                    if load['connection'] in ('wye', 'delta'):
                        # A single-phase delta load given one node is phase-to-ground, like a
                        # wye load: OpenDSS fills its unspecified second conductor with node 0
                        # (NodeOrder [1, 0] for IEEE 34-bus "Bus1=832.1 Phases=1 Conn=Delta").
                        load_info.append({'bus': load['bus'], 'phases': (p, 0)})
                    else:
                        raise ValueError(f"Unsupported connection type: {load['connection']} for {load}")
                elif len(phases) == 2:
                    assert load['connection'] == 'delta', load
                    load_info.append({'bus': load['bus'], 'phases': tuple(phases)})
                else:
                    raise ValueError(f"Unsupported phase configuration: {load['phases']} for {load}")
            elif num_phases == 3:
                assert len(phases) == 3, load['phases']
                if load['connection'] == 'wye':
                    for p in (1, 2, 3):
                        load_info.append({'bus': load['bus'], 'phases': (p, 0)})
                elif load['connection'] == 'delta':
                    for leg in ((1, 2), (2, 3), (3, 1)):
                        load_info.append({'bus': load['bus'], 'phases': leg})
                else:
                    raise ValueError(f"Unsupported connection type: {load['connection']} for {load}")
            else:
                raise ValueError(f"Unsupported number of phases: {num_phases} for {load}")
        return load_info

    def to_circuit_data(self, feeder_root, plot_circuit=False, output_dir=None) -> list:
        """
        Map dss data into 3-phase LindistFlow format

        Args:
            feeder_root: name of the bus to treat as the feeder head; the returned
                circuit is the subtree reachable from it.
            plot_circuit: if True, render network diagrams into the output directory.
                Off by default because it requires pygraphviz and takes seconds to
                minutes on a full feeder.
            output_dir: overrides the instance's ``output_dir`` for this call. Used for
                the cycle report and the network plots; created only if written to.

        returns:
            circuit_data: list of dicts, one per branch, with keys:
                - element_name: str, the OpenDSS element this came from
                - element_type: str, one of 'line', 'YY', 'DY', 'DY_alt', 'YD', 'DD',
                  'openY_openY', 'openY_openD_split', 'openD_openD', 'openD_openD_split'
                - source: str, source bus name
                - target: str, target bus name
                - phases: tuple of int, e.g. (1, 2) for AB, (1,) for A only; for a
                  non-consolidated transformer winding, a list of two such tuples
                - series_impedance: np.ndarray, shape (3, 3), complex ohms
                - turns_ratio: float, primary:secondary internal voltage (a_jk)
                - S_rating: float, apparent power rating if one could be determined.
                  Note the unit differs by element: transformer winding kVA vs. a line's
                  ampacity times base kV.
                - winding_connections: list of str, transformers only
                - floating_neutral: [bool, bool], transformers only: whether the primary
                  and the secondary winding is a wye whose neutral floats (it is on a spare
                  node of the bus that only the neutrals of the bank's units connect to)
                  rather than being grounded. False for a delta winding.
                - split_phase / winding_pairing / open_phase: open and split-phase
                  transformers only
        """
        def _find_duplicate_edges(circuit_data):
            edges = set()
            duplicates, dup_list = {}, []
            for edge in circuit_data:
                identifier = (edge['source'], edge['target'])
                if identifier in edges:
                    dup_list.append(identifier)
                else:
                    edges.add(identifier)
                if identifier not in duplicates:
                    duplicates[identifier] = []
                duplicates[identifier].append(edge)
            duplicates = {k: v for k, v in duplicates.items() if k in dup_list}
            return duplicates

        def match_linecode_to_conductor(linecode: str, candidate_lists: List[List[str]]) -> Tuple[Optional[str], float, Optional[int]]:
            """
            Match a single OpenDSS LineCode to the best conductor name, across multiple catalogs.

            Args
            -----
            linecode : str
                e.g. "oh_4/0_al_xlpe_abc" (utility linecode names are often decorated
                with feeder/phase/construction suffixes that must be matched loosely)
            candidate_lists : List[List[str]]
                Outer list = different catalogs (e.g., [wiredata_names, cndata_names]).
                Each inner list = candidate names from that catalog.

            Returns
            -------
            (matched_name, score, source_index)
                matched_name : str or None
                score : float confidence
                source_index : int index of which candidate list the match came from
                            (0 for the first list, 1 for the second, etc.)

            Notes
            -----
            - Uses the same 3-tier scoring scheme as before (EXACT, PREFIX+env, FUZZY),
            with a strong bonus (+0.75) for matching the leading gauge/size token.
            - Treat score < 1.25 as low confidence.
            """

            def _extract_stem_and_suffix(code: str):
                parts = code.strip().split("__", 1)
                stem = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""
                return stem, suffix

            def _strip_pn_prefix(stem: str) -> str:
                return re.sub(r'^[pn_]+', '', stem, flags=re.IGNORECASE)

            def _tokens(s: str):
                return [t for t in re.split(r'[^A-Za-z0-9#/]+', s.lower()) if t]

            def _leading_gauge(s: str) -> str:
                s = s.lstrip()
                m = re.match(r'^(#\d+)', s)
                if m: return m.group(1)
                m = re.match(r'^(\d+/\d+)', s)
                if m: return m.group(1)
                m = re.match(r'^(\d+)', s)
                if m: return m.group(1)
                return ""

            def _gauge_bonus(g1: str, g2: str) -> float:
                return 0.75 if (g1 and g2 and g1 == g2) else 0.0

            # Flatten candidate lists with index tracking
            candidates = []
            for idx, clist in enumerate(candidate_lists):
                for c in clist:
                    candidates.append((c, idx))

            if not candidates:
                return None, 0.0, None

            stem, suffix = _extract_stem_and_suffix(linecode)
            stem_norm = _strip_pn_prefix(stem).lower()
            stem_gauge = _leading_gauge(stem_norm)

            exact_forms = {stem_norm, stem_norm + SECONDARY_CONDUCTOR_SUFFIX}
            cand_lc = [c[0].lower() for c in candidates]
            cand_gauges = [_leading_gauge(c[0]) for c in candidates]

            # Tier 1: EXACT
            for (cand, source), cl, cg in zip(candidates, cand_lc, cand_gauges):
                if cl in exact_forms:
                    return cand, 3.0, source

            # Tier 2: PREFIX
            starts = [(cand, cl, cg, source) for (cand, source), cl, cg in zip(candidates, cand_lc, cand_gauges)
                    if cl.startswith(stem_norm) or stem_norm.startswith(cl)]
            if starts:
                suffix_l = suffix.lower()
                env = None
                if "ugconst" in suffix_l or "_ug_" in stem_norm:
                    env = "ug"
                elif "ohsecconst" in suffix_l or "secconst" in suffix_l:
                    env = "sec"
                elif "ohconst" in suffix_l:
                    env = "oh"

                filtered = starts
                if env == "ug":
                    ug_pref = [t for t in starts if "_ug_" in t[1] or SECONDARY_CONDUCTOR_SUFFIX not in t[1]]
                    if ug_pref:
                        filtered = ug_pref
                    else:
                        sec_like = [t for t in starts if SECONDARY_CONDUCTOR_SUFFIX in t[1]]
                        if sec_like:
                            filtered = sec_like
                elif env == "sec":
                    sec_like = [t for t in starts if SECONDARY_CONDUCTOR_SUFFIX in t[1]]
                    if sec_like:
                        filtered = sec_like

                best_cand, best_cl, best_cg, best_src = max(filtered, key=lambda t: len(t[1]))
                coverage = len(stem_norm) / max(1, len(best_cl))
                score = 2.0 + coverage + _gauge_bonus(stem_gauge, best_cg)
                return best_cand, float(score), best_src

            # Tier 3: FUZZY
            stem_toks = set(_tokens(stem_norm))
            best, best_score, best_src = None, -1.0, None
            for (cand, source), cl, cg in zip(candidates, cand_lc, cand_gauges):
                if cl.startswith(stem_norm):
                    prefix_cov = len(stem_norm) / max(1, len(cl))
                elif stem_norm.startswith(cl):
                    prefix_cov = len(cl) / max(1, len(stem_norm))
                else:
                    prefix_cov = 0.0
                cand_toks = set(_tokens(cl))
                inter = len(stem_toks & cand_toks)
                union = len(stem_toks | cand_toks) or 1
                jacc = inter / union
                seq = SequenceMatcher(a=stem_norm, b=cl).ratio()
                gbonus = _gauge_bonus(stem_gauge, cg)
                score = gbonus + (2.0 * prefix_cov) + (1.0 * jacc) + (0.5 * seq)
                if score > best_score:
                    best_score, best, best_src = score, cand, source
            return best, float(best_score), best_src

        def _sanitize_small_impedance(edge_data, label):
            """Detect very small non-zero impedance entries, warn and replace with a small floor."""
            z = edge_data.get('series_impedance')
            if z is None:
                return
            tiny_thresh = 1e-12
            tiny_floor = 1e-9
            # mask of entries that are non-zero but extremely small
            small_mask = (~np.isclose(z, 0)) & (np.abs(z) < tiny_thresh)
            if np.any(small_mask):
                warnings.warn(
                    f"Very small impedance entries detected in branch {label}; replacing with {tiny_floor}",
                    RuntimeWarning,
                )
                vals = z[small_mask]
                # preserve phase of complex value, set magnitude to tiny_floor
                z[small_mask] = (vals / np.abs(vals)) * tiny_floor
                edge_data['series_impedance'] = z

        def _reactor_series_impedance(reactor):
            """Series impedance of a Reactor on its own phases, zero on the others (as for a line)."""
            phases = [p - 1 for p in reactor['phases']]
            if sorted(phases) != sorted(set(phases)) or not set(phases) <= {0, 1, 2}:
                raise ValueError(f"Reactor {reactor['name']} is connected to nodes {reactor['phases']}; "
                                 "only distinct phase nodes 1, 2, 3 are supported")
            z = np.zeros((3, 3), dtype=complex)
            z[np.ix_(phases, phases)] = reactor['Z']
            return z

        # Begin main function content
        assert feeder_root, "feeder_root must be specified to format circuit data."
        # Destination for optional diagnostics. Created lazily, only when written to.
        base_out = Path(output_dir) if output_dir is not None else self.output_dir
        # Name the subdirectory after the feeder directory, not after the master file:
        # every dataset's master is called "Master.dss", so naming it after the file makes
        # all feeders collide in (and overwrite each other's diagnostics in) one directory.
        report_dir = Path(base_out) / self.base_path.name
        winding_connection_mapping = {
            ("wye", "wye"): "YY",
            ("delta", "delta"): "DD",
            ("wye", "delta"): "YD",
            ("delta", "wye"): "DY"
        }
        # Ignore fictitious nodes
        remove_nodes_file = os.path.join(self.base_path, "remove_nodes.txt")
        remove_nodes = set()
        if os.path.exists(remove_nodes_file):
            with open(remove_nodes_file, "r") as f:
                remove_nodes = set([line.strip() for line in f])
            
        # Circuit data is a list of edges: lines, transformers, switches
        circuit_data = []
        wiredata_names = list(self.dss_data['wire_data'].keys())
        cndata_names = list(self.dss_data['cn_data'].keys())
        for line in self.dss_data['lines'].values():
            if (line['bus1'] in remove_nodes)  or (line['bus2'] in remove_nodes):
                continue
            edge_data = {
                'element_name': line['name'],
                'source': line['bus1'],
                'target': line['bus2'],
                'turns_ratio': 1.0,
                'element_type': 'line',
                'phases': line['phases'],
            }
            # DSS R/X matrices are per-unit-length; multiply by line length to get total impedance.
            try:
                line['RMatrix'] = np.array(line['RMatrix']) * line['length']
                line['XMatrix'] = np.array(line['XMatrix']) * line['length']
            except Exception:
                # Fallback: use a small default per-phase impedance to avoid crashing
                Z_default = 0.001 + 1j * 0.005
                line['RMatrix'] = np.array([Z_default.real])
                line['XMatrix'] = np.array([Z_default.imag])
                logger.warning(
                    "Could not process R/X matrices for line %s; using default Z=%s per phase.",
                    line['name'],
                    Z_default,
                )
            if len(line['RMatrix']) == 9:
                edge_data['series_impedance'] = line['RMatrix'].reshape(3,3) + 1j * line['XMatrix'].reshape(3,3)
            elif len(line['RMatrix']) == 4:
                assert len(line['phases']) == 2, (line['phases'], line['name'])
                # For non-existent lines, make the self-impedance zero
                edge_data['series_impedance'] = np.zeros((3, 3), dtype=complex)
                R, X = line['RMatrix'].reshape(2,2), line['XMatrix'].reshape(2,2)
                p1, p2 = line['phases'][0] - 1, line['phases'][1] - 1
                edge_data['series_impedance'][p1, p1] = R[0, 0] + 1j * X[0, 0]
                edge_data['series_impedance'][p2, p2] = R[1, 1] + 1j * X[1, 1]
                edge_data['series_impedance'][p1, p2] = R[0, 1] + 1j * X[0, 1]
                edge_data['series_impedance'][p2, p1] = R[1, 0] + 1j * X[1, 0]
            elif len(line['RMatrix']) == 1:
                Z = complex(line['RMatrix'][0]) + 1j * complex(line['XMatrix'][0])
                # For non-existent lines, make the self-impedance zero
                edge_data['series_impedance'] = np.zeros((3, 3), dtype=complex)
                for p in line['phases']:
                    if p == 0: continue
                    edge_data['series_impedance'][p-1, p-1] = Z
            else:
                raise ValueError(f"Unexpected RMatrix length: {len(line['RMatrix'])}")
            if (edge_data['series_impedance'] == 0).all():
                logger.warning("Zero impedance detected in branch: %s.", line['name'])
                
            # To obtain line power ratings, match line code to WireData or CNData. 
            # This is a bit hacky, but some DSS files do not explicitly link LineCodes to conductor ampacities.
            matched_name, score, which = match_linecode_to_conductor(line['line_code'], [wiredata_names, cndata_names])
            if score < 1:
                logger.warning(
                    "Line code '%s' match score is low: %s, matched to %s.",
                    line['line_code'],
                    score,
                    matched_name,
                )
            # The norm_amps field is per-phase, and the constraints are instantiated per-phase.
            if which == 0:
                I_rating = self.dss_data['wire_data'][matched_name]['norm_amps']
            elif which == 1:
                I_rating = self.dss_data['cn_data'][matched_name]['norm_amps']
            else:
                # Fallback: use any normative ampacity reported by DSS for the line
                I_rating = line.get('norm_amps')
                if I_rating is None:
                    # No good conductor match or DSS norm amps; use conservative default
                    I_rating = 100.0
                    logger.warning(
                        "No ampacity match for line %s; using default I_rating=%s",
                        line['name'],
                        I_rating,
                    )
            edge_data['S_rating'] = I_rating * self.dss_data['buses'][line['bus1']]['base_kv']
            _sanitize_small_impedance(edge_data, line['name'])
            circuit_data.append(edge_data)
            
        # Process transformers
        # Standard winding pairing (PSA textbook): AB – AN, BC – BN, CA – CN
        # Alternative winding pairing: AC – AN, BA – BN, CB – CN
        xfmr_lookup = {
            'DY': {
                ((1, 2), (1,)): 'DY',
                ((2, 3), (2,)): 'DY',
                ((1, 3), (3,)): 'DY',
                ((1, 3), (1,)): 'DY_alt',
                ((1, 2), (2,)): 'DY_alt',
                ((2, 3), (3,)): 'DY_alt',
            },
            'YD': {
                ((1,), (1, 2)): 'YD',
                ((2,), (2, 3)): 'YD',
                ((3,), (1, 3)): 'YD',
                ((1,), (1, 3)): 'YD_alt',
                ((2,), (1, 2)): 'YD_alt',
                ((3,), (2, 3)): 'YD_alt',
            }
        }
        # Per transformer, its taps other than 1 (such as the saved position of a regulator),
        # which the turns ratios below, from the rated kV, leave out.
        ignored_taps = []
        for xfmr in self.dss_data['transformers'].values():
            # Make sure the transformer windings buses are in the form of src, tgt, tgt, ...
            bus_set = set(xfmr['buses'])
            assert len(bus_set) == 2, (xfmr['buses'], xfmr['name'])
            src, tgt = xfmr['buses'][0], (bus_set - {xfmr['buses'][0]}).pop()
            if (src in remove_nodes)  or (tgt in remove_nodes):
                continue
            if any(abs(tap - 1) > 1e-9 for tap in xfmr['taps']):
                ignored_taps.append([tap for tap in xfmr['taps'] if abs(tap - 1) > 1e-9])
            assert all([b != src for b in xfmr['buses'][1:]]), (xfmr['buses'], src, xfmr['name'])
            if xfmr['num_phases'] == 2 and 'delta' in xfmr['winding_connections']:
                # OpenDSS gives it three conductors, the third grounded: one winding from
                # phase to phase, the other from phase to ground.
                raise NotImplementedError(
                    f"Transformer {xfmr['name']}: a two-phase transformer with a delta winding is not supported")
            _check_like_windings_on_like_nodes(xfmr)
            # Add each winding pair as a separate edge. Let the consolidation step handle it.
            for t in range(1, xfmr['windings']):
                edge_data = {
                    'element_name': xfmr['name'],
                    'source': src,
                    'target': tgt,
                    'turns_ratio': xfmr['kvs'][0] / xfmr['kvs'][t],
                    'phases': [xfmr['phases'][0], xfmr['phases'][t]],
                    'S_rating': xfmr['kvas'][t],
                    'winding_connections': [xfmr['winding_connections'][0], xfmr['winding_connections'][t]],
                    'floating_neutral': [xfmr['floating_neutral'][0], xfmr['floating_neutral'][t]],
                    # Checked (and dropped) once the units of banks are consolidated, below.
                    '_neutral_nodes': [(xfmr['neutral_nodes'][0], xfmr['neutral_nodes'][t])],
                }
                # Type of transformer, i.e. 'element_type'
                if tuple(edge_data["winding_connections"]) in winding_connection_mapping:
                    edge_data['element_type'] = winding_connection_mapping[tuple(edge_data["winding_connections"])]
                    if edge_data['element_type'] in ('DY', 'YD'):
                        raw_nodes = [tuple(xfmr['phases'][w]) for w in (0, t)]
                        nodes = [tuple(p for p in n if p != 0) for n in raw_nodes]
                        winding_pair = tuple(tuple(sorted(set(n))) for n in nodes)
                        # A three-phase bank does not name its winding pairs: OpenDSS connects the
                        # delta so that the low-voltage side lags the high-voltage side by 30 degrees
                        # (LeadLag=Lag, the default) or leads it (LeadLag=Lead), taking winding 1 as
                        # the high-voltage side when the kV are equal. Of the two pairings, DY_alt
                        # and YD make the secondary lag the primary. A bare bus name connects
                        # nodes 1, 2, 3. The same node order on both windings relabels the
                        # phases: a cyclic one (2.3.1) keeps the pairing, an anti-cyclic one
                        # (1.3.2) reverses the phase sequence and so swaps the two pairings.
                        # Different orders on the two windings shift the phases in a way
                        # neither pairing represents.
                        if xfmr['num_phases'] == 3:
                            cores = [n[:3] or (1, 2, 3) for n in raw_nodes]
                            cyclic, anticyclic = ((1, 2, 3), (2, 3, 1), (3, 1, 2)), ((1, 3, 2), (3, 2, 1), (2, 1, 3))
                            if cores[0] != cores[1] or cores[0] not in cyclic + anticyclic \
                                    or any(n[3:] not in ((), (0,)) for n in raw_nodes):
                                raise ValueError(
                                    f"Transformer {xfmr['name']}: a three-phase delta-wye bank must connect "
                                    f"both windings to nodes 1, 2, 3 in the same order, got {raw_nodes}")
                            secondary_lags = (xfmr['kvs'][0] >= xfmr['kvs'][t]) == (xfmr['lead_lag'] == 'lag')
                            if cores[0] in anticyclic:
                                secondary_lags = not secondary_lags
                            if secondary_lags == (edge_data['element_type'] == 'DY'):
                                edge_data['element_type'] = edge_data['element_type'] + '_alt'
                        # Otherwise, for a single pair of windings, use the lookup table.
                        elif winding_pair in xfmr_lookup[edge_data['element_type']]:
                            edge_data['element_type'] = xfmr_lookup[edge_data['element_type']][winding_pair]
                            # The pairings take the wye winding from phase p to neutral and the
                            # delta winding from node p to node p+1 (p-1 for _alt). Reversing
                            # either one (neutral first, or the delta the other way round) shifts
                            # the secondary by 180 degrees, which neither pairing represents.
                            # Reversing both does not.
                            delta, wye = (raw_nodes[0], raw_nodes[1]) if edge_data['element_type'].startswith('DY') \
                                else (raw_nodes[1], raw_nodes[0])
                            p = next(node for node in wye if node != 0)
                            step = 1 if edge_data['element_type'] in ('DY', 'YD') else -1
                            delta_reversed = delta != (p, (p - 1 + step) % 3 + 1)
                            wye_reversed = wye[0] == 0
                            if delta_reversed != wye_reversed:
                                raise ValueError(
                                    f"Transformer {xfmr['name']}: its delta winding on nodes {delta} and wye "
                                    f"winding on nodes {wye} have opposite polarity, which is not supported")
                        else:
                            raise ValueError(f"Unknown transformer winding connection {winding_pair} for {xfmr}")
                    # Wye-wye and delta-delta pairs were checked by _check_like_windings_on_like_nodes.
                else:
                    raise ValueError(f"Unknown transformer winding connection: {xfmr['winding_connections']}")  
                # Calculate series impedance
                z_pu = (xfmr['pcrRs'][0] + xfmr['pcrRs'][t] + 1j * xfmr['xhl']) * 1e-2
                if xfmr['kvas'][0] == 0:
                    z_base = 0
                    logger.warning(
                        "Transformer %s has zero kVA rating. Assuming series impedance is zero, "
                        "which the admittance models (BIM, power flow) reject until it is given one.",
                        xfmr['name'],
                    )
                else:
                    z_base = xfmr['kvs'][0] ** 2 / xfmr['kvas'][0] * 1e3     # Normalize to primary winding voltage
                    # The branch models take the impedance of each primary winding. A two- or
                    # three-phase transformer is rated in line-to-line kV and total kVA, so each
                    # winding has 1/phases of the kVA, and a wye winding a third of the kV**2:
                    # a delta winding has three times the wye base, and a two-phase wye 2/3.
                    if xfmr['num_phases'] == 3 and xfmr['winding_connections'][0] == 'delta':
                        z_base *= 3
                    elif xfmr['num_phases'] == 2:
                        z_base *= 2 / 3
                # For single-phase transformers, we still use the diagonal impedance matrix.
                # This is not physically correct, but does not affect the solution, since the fictitious
                # (non-existent) winding ports will always have zero current injection (constrained by the load injections).
                edge_data['series_impedance'] = np.eye(3) * z_pu * z_base
                # Turns ratio. Note that DSS kV ratings are always line-to-line for three-phase transformers.
                # Thus, for 3-phase transformers, the turns ratio needs to be adjusted based on the winding connection.
                # https://dss-extensions.org/dss-format/Transformer.html
                if (edge_data['element_type'] in ('DY', 'DY_alt')) and (xfmr['num_phases'] == 3):
                    edge_data['turns_ratio'] *= np.sqrt(3)
                elif (edge_data['element_type'] in ('YD', 'YD_alt')) and (xfmr['num_phases'] == 3):
                    edge_data['turns_ratio'] /= np.sqrt(3)
                _sanitize_small_impedance(edge_data, xfmr['name'])
                circuit_data.append(edge_data)
        if ignored_taps:
            # Without element names, which must not be shown for private feeders.
            taps = [tap for xfmr_taps in ignored_taps for tap in xfmr_taps]
            warnings.warn(
                f"{len(ignored_taps)} transformer(s) have taps other than 1 (from {min(taps):g} to "
                f"{max(taps):g}), which are ignored: turns ratios are taken from the rated kV only.",
                stacklevel=2,
            )

        # Switched reactors (treated as lines when closed)
        _reactors = copy.deepcopy(self.dss_data['reactors'])
        for swt in self.dss_data["swt_control"].values():
            swt_obj = _reactors.pop(swt["swt_obj"].Name)
            if swt["normal"].value == 1: continue
            assert swt["normal"].value == 2, (swt["normal"], swt["name"])
            if (swt_obj['bus1'] in remove_nodes)  or (swt_obj['bus2'] in remove_nodes):
                continue
            edge_data = {
                'element_name': swt_obj['name'],
                'source': swt_obj['bus1'],
                'target': swt_obj['bus2'],
                'turns_ratio': 1.0,
                'element_type': 'line',
                'phases': swt_obj['phases'],
                'S_rating': None,
            }
            edge_data['series_impedance'] = _reactor_series_impedance(swt_obj)
            _sanitize_small_impedance(edge_data, swt_obj['name'])
            circuit_data.append(edge_data)

        # Include remaining reactors (non-switched) as line-like edges so the
        # network graph remains connected. These are treated as series impedances
        # between their two terminal buses.
        for r_name, r_obj in _reactors.items():
            if (r_obj['bus1'] in remove_nodes) or (r_obj['bus2'] in remove_nodes):
                continue
            edge_data = {
                'element_name': r_obj['name'] if 'name' in r_obj else r_name,
                'source': r_obj['bus1'],
                'target': r_obj['bus2'],
                'turns_ratio': 1.0,
                'element_type': 'line',
                'phases': r_obj['phases'],
                'S_rating': None,
            }
            edge_data['series_impedance'] = _reactor_series_impedance(r_obj)
            _sanitize_small_impedance(edge_data, edge_data['element_name'])
            circuit_data.append(edge_data)

        """Consolidate duplicate edges (parallel edges)"""
        duplicates = _find_duplicate_edges(circuit_data)
        for l in duplicates.values():
            for edge_data in l:
                circuit_data.remove(edge_data)
            for k in ('source', 'target'):
                if len(set([edge_data[k] for edge_data in l])) != 1:
                    raise ValueError(f"All {k} should be the same for {[ele['element_name'] for ele in l]}")
            for k in ('S_rating',):
                if len(set([edge_data[k] for edge_data in l])) != 1:
                    logger.warning(
                        "All %s should be the same for %s. Picking the first arbitrarily.",
                        k,
                        [ele['element_name'] for ele in l],
                    )
            if l[0]['element_type'] in ('line',):
                raise NotImplementedError("Consolidation of line elements is not implemented.")
            # Combine parallel edges into a single element
            consolidated_edge_data = {
                "element_name": "|".join([edge_data['element_name'] for edge_data in l]),
                "source": l[0]['source'],
                "target": l[0]['target'],
                'S_rating': l[0]['S_rating'],
                '_neutral_nodes': [nodes for edge_data in l for nodes in edge_data['_neutral_nodes']],
            }
            if len({tuple(edge_data['floating_neutral']) for edge_data in l}) != 1:
                raise ValueError(
                    f"Transformers {[ele['element_name'] for ele in l]}: the units of a bank must all "
                    "ground the neutral of a wye winding, or all leave it floating")
            consolidated_edge_data['floating_neutral'] = list(l[0]['floating_neutral'])
            # Here we only handle open type transformers. Other types are not implemented yet and will throw an error.
            # Single-phase and standard three-phase transformers are handled above.
            # Primary conection type (delta, wye)
            assert len(set([edge_data['winding_connections'][0] for edge_data in l])) == 1, \
                f"All primary winding connections should be the same for {[ele['element_name'] for ele in l]}"
            conn_pri = l[0]['winding_connections'][0]
            # Store winding pairings in a dictionary
            # For each dictionary, the key entry structure is: primary phase -> list (secondary phase(s)) --> array or value
            winding_pairing, turns_ratios, series_impedance = OrderedDict(), OrderedDict(), OrderedDict()
            for edge_data in l:
                pri_phases = tuple(sorted(set(edge_data['phases'][0]) - {0}))
                sec_phases = tuple(sorted(set(edge_data['phases'][1]) - {0}))
                if pri_phases in winding_pairing:
                    winding_pairing[pri_phases].append(sec_phases)
                    turns_ratios[pri_phases].append(edge_data['turns_ratio'])
                    series_impedance[pri_phases].append(edge_data['series_impedance'])
                else:
                    winding_pairing[pri_phases] = [sec_phases]
                    turns_ratios[pri_phases] = [edge_data['turns_ratio']]
                    series_impedance[pri_phases] = [edge_data['series_impedance']]
            # If we have a standard 3-phase transformer (no open/split)
            if len(winding_pairing) == 3:
                assert len({edge_data['element_type'] for edge_data in l}) == 1, \
                    f"All element types should be the same for {[ele['element_name'] for ele in l]}"
                consolidated_edge_data['element_type'] = l[0]['element_type']
                if consolidated_edge_data['element_type'] in ('YY', 'YD', 'YD_alt'):
                    series_impedance_list = [series_impedance[p][0][0,0] for p in ((1,), (2,), (3,))]
                elif consolidated_edge_data['element_type'] in ('DD', 'DY', ):
                    series_impedance_list = [series_impedance[p][0][0,0] for p in ((1,2), (2,3), (1,3))]
                elif consolidated_edge_data['element_type'] in ('DY_alt'):
                    series_impedance_list = [series_impedance[p][0][0,0] for p in ((1,3), (1,2), (2,3))]
                consolidated_edge_data["series_impedance"] = np.diag(series_impedance_list)
                assert len({edge_data['turns_ratio'] for edge_data in l}) == 1, \
                    f"Non-equal turns_ratio is not implemented yet."
                consolidated_edge_data['turns_ratio'] = l[0]['turns_ratio']
                all_pri = sorted({p for pri in winding_pairing.keys() for p in pri})
                all_sec = sorted({p for sec_list in winding_pairing.values() for sec in sec_list for p in sec})
                consolidated_edge_data['phases'] = [all_pri, all_sec]
                circuit_data.append(consolidated_edge_data)
                continue
            # Detect split phase. Treat all three-winding transformers as split-phase
            split_phase = None
            turns_ratios_list, series_impedance_list = [], []
            # A wye primary's impedances go on the diagonal by phase (the missing phase gets a
            # zero, below), so take its units in phase order, not in the order they are defined.
            pairings = sorted(winding_pairing.items()) if conn_pri == 'wye' else list(winding_pairing.items())
            for pri, sec_list in pairings:
                if len(sec_list) == 1:
                    turns_ratios_list.append(turns_ratios[pri][0])
                    series_impedance_list.append(series_impedance[pri][0][0, 0])    # Assumes: edge_data['series_impedance'] = np.eye(3) * z_pu * z_base
                # If split phase
                elif len(sec_list) == 2:
                    assert len(sec_list[0]) == len(sec_list[1]) == 1, \
                        f"{edge_data['element_name']} winding pairing style is not handled."
                    assert split_phase is None, f"Multiple split-phase windings detected in {edge_data['element_name']}"
                    winding_pairing[pri] = [(sec_list[0][0], sec_list[1][0])]
                    split_phase = (pri, tuple(sorted([sec_list[0][0], sec_list[1][0]])))
                    # Reconcile turns ratio (equivalent turns ratio: add the 2 split phases)
                    assert turns_ratios[pri][0] == turns_ratios[pri][1], \
                        f"{edge_data['element_name']} split-phase winding turns ratio mismatch."
                    turns_ratios_list.append(turns_ratios[pri][0] / 2)
                    # Reconcile series impedance
                    assert np.allclose(series_impedance[pri][0], series_impedance[pri][1]), \
                        f"{edge_data['element_name']} split-phase winding series impedance mismatch."
                    r, x = series_impedance[pri][0].real, series_impedance[pri][0].imag
                    series_impedance_list = [r[0, 0] * 2 / 3 + 1j * x[0, 0]] + series_impedance_list    # Assumes: edge_data['series_impedance'] = np.eye(3) * z_pu * z_base
                elif len(sec_list) > 2:
                    raise ValueError(f"{edge_data['element_name']} winding pairing style is not handled.")
            consolidated_edge_data['split_phase'] = split_phase
            assert len(set(turns_ratios_list)) == 1, \
                f"All equivalent turns ratios should be the same for {[ele['element_name'] for ele in l]}: {turns_ratios_list}"
            consolidated_edge_data['turns_ratio'] = turns_ratios_list[0]
            # Detect open connection.
            if len(winding_pairing) == 2:
                if all([len(pri) == 1 for pri in winding_pairing.keys()]):
                    open_pri = (({1, 2, 3} - {p[0] for p in winding_pairing.keys()}).pop(),)
                    assert conn_pri == 'wye', conn_pri
                    conn_pri = 'Y'
                elif all([len(pri) == 2 for pri in winding_pairing.keys()]):
                    open_pri = ({(1,2), (2,3), (1,3)} - {tuple(sorted(p)) for p in winding_pairing.keys()}).pop()
                    assert conn_pri == 'delta', conn_pri
                    conn_pri = 'D'
                else:
                    raise ValueError(f"{edge_data['element_name']} winding pairing style is not handled.")
                if all([len(sec_list[0]) == 1 for sec_list in winding_pairing.values()]):
                    open_sec = (({1, 2, 3} - {p[0] for sec_list in winding_pairing.values() for p in sec_list}).pop(),)
                    conn_sec = 'Y'
                elif all([len(sec_list[0]) == 2 for sec_list in winding_pairing.values()]):
                    open_sec = ({(1,2), (2,3), (1,3)} - {tuple(sorted(sec_list[0])) for sec_list in winding_pairing.values()}).pop()
                    conn_sec = 'D'
                else:
                    raise ValueError(f"{edge_data['element_name']} winding pairing style is not handled:{winding_pairing}")
            else:
                raise ValueError(f"{edge_data['element_name']} winding pairing style is not handled:{winding_pairing}")
            assert all([len(sec_list) == 1 for sec_list in winding_pairing.values()]), winding_pairing
            winding_pairing = tuple([(tuple(pri), tuple(sorted(winding_pairing[pri][0]))) for pri in sorted(winding_pairing.keys())])
            consolidated_edge_data['winding_pairing'] = winding_pairing
            consolidated_edge_data['element_type'] = f"open{conn_pri}_open{conn_sec}{'_split' if split_phase else ''}"
            consolidated_edge_data['open_phase'] = (open_pri, open_sec)
            # Calculate the series impedance for each winding pair
            assert len(series_impedance_list) == 2, series_impedance_list
            if conn_pri == 'Y':
                series_impedance_list.insert(open_pri[0] - 1, 0)
            elif conn_pri == 'D':
                i_insert = ((1,2), (2,3), (1,3)).index(open_pri[0])
                series_impedance_list.insert(i_insert, 0)
            consolidated_edge_data["series_impedance"] = np.diag(series_impedance_list)
            all_pri = sorted({p for pri, _sec in winding_pairing for p in pri})
            all_sec = sorted({p for _pri, sec in winding_pairing for p in sec})
            consolidated_edge_data['phases'] = [all_pri, all_sec]
            circuit_data.append(consolidated_edge_data)

        # A floating neutral returns the current of each winding on it through the others. So
        # the windings sharing one must be exactly those of one bank, and more than one.
        neutral_users = {}
        for xfmr in self.dss_data['transformers'].values():
            for node in xfmr['neutral_nodes']:
                if node is not None:
                    neutral_users.setdefault(node, set()).add(xfmr['name'])
        for edge_data in circuit_data:
            unit_neutral_nodes = edge_data.pop('_neutral_nodes', None)
            if unit_neutral_nodes is None or not any(edge_data['floating_neutral']):
                continue
            units = set(edge_data['element_name'].split('|'))
            for w in (0, 1):
                if not edge_data['floating_neutral'][w]:
                    continue
                nodes = {unit_nodes[w] for unit_nodes in unit_neutral_nodes}
                if len(nodes) != 1 or neutral_users[next(iter(nodes))] != units:
                    raise NotImplementedError(
                        f"Transformer(s) {sorted(units)}: winding {w + 1} has a floating wye neutral "
                        f"on {sorted(nodes)}, shared with {sorted(set().union(*(neutral_users[n] for n in nodes)))}; "
                        "the units sharing a floating neutral must be exactly the units of one bank")
                phases = {p for p in edge_data['phases'][w] if p in (1, 2, 3)}
                if len(phases) == 1:
                    raise NotImplementedError(
                        f"Transformer {edge_data['element_name']}: the floating neutral of its "
                        f"single-phase winding {w + 1} gives the winding no return path")
                # A single transformer is given windings on all three phases (see its series
                # impedance above), which a floating neutral would return current through.
                if len(units) == 1 and len(phases) == 2:
                    raise NotImplementedError(
                        f"Transformer {edge_data['element_name']}: a floating neutral on a two-phase "
                        "transformer is not supported; write it as one single-phase unit per phase")

        # check script: map all other transformers to YY
        # for e in circuit_data:
        #     if e['element_type'] not in {'line', 'YY'}:
        #         e['element_type'] = 'YY'

        _orient_from_source(circuit_data, feeder_root.lower(), self.dss_data.get('source_bus'))

        """Network check scripts"""
        # Check number of connected components    
        edge2name = to_edge_list(circuit_data, include_name=True)    
        edge_list = list(edge2name.keys())
        edge2index = {e: i for i, e in enumerate(edge_list)}
        graph = nx.DiGraph(edge_list)
        isolated_nodes = list(nx.isolates(graph))
        unknown_buses = [n for n in graph.nodes if n not in self.dss_data['buses']]
        # Consolidate single-phase transformers up when loading transformers
        if len(circuit_data) != len(graph.edges):
            # print(f"[Warning]: {len(circuit_data) - len(graph.edges)} duplicate (parallel) edges in circuit_data.")
            raise RuntimeError(f"Found {len(circuit_data) - len(graph.edges)} duplicate (parallel) edges.")
        logger.info("Number of edges, nodes: %d, %d", len(graph.edges), len(graph.nodes))
        if unknown_buses:
            logger.warning("Unknown buses detected: %s.", unknown_buses)
        logger.info("Number of isolated nodes: %d", len(isolated_nodes))
        graph.remove_nodes_from(isolated_nodes)
        components = list(nx.connected_components(graph.to_undirected()))
        logger.info("Number of connected components: %d", len(components))
        logger.info("Component sizes: %s", [len(c) for c in components])
        cycles = nx.cycle_basis(graph.to_undirected())
        logger.info("Number of cycles: %d.", len(cycles))
        if len(cycles):
            report_dir.mkdir(parents=True, exist_ok=True)
            cycle_path = report_dir / "cycle.txt"
            with open(cycle_path, "w") as f:
                for c, cycle in enumerate(cycles):
                    if c > 0:
                        f.write('\n')
                    f.write(f"Cycle {c}:\n")
                    f.write(",".join(cycle))
            logger.info("Wrote cycle report to %s", cycle_path)


        # Take the subgraph at feeder_root
        try:
            nodes_to_keep = list(nx.bfs_tree(graph, source=feeder_root.lower()).nodes)
        except Exception as e:
            logger.error("Failed to find feeder root: %s", e)
            raise e
        graph_pruned = graph.subgraph(nodes_to_keep)
        circuit_data = [circuit_data[edge2index[e]] for e in graph_pruned.edges]

        # Plot circuit
        if plot_circuit:
            # Use the dataset's geographic bus coordinates (parsed from nodecoords.csv /
            # BusCoords.dss into dss_data['bus_coordinates'], keyed by lowercased bus name).
            # dss_data['buses'][*]['coordinate'] is not usable here: the autogen master never
            # feeds the coordinate file to OpenDSS, so altdss reports (0, 0) for every bus.
            # plot_nx_graph needs a position for every node it draws, so fall back to its
            # automatic layout (pos=None) unless the coordinates cover the whole graph.
            bus_coords = self.dss_data.get('bus_coordinates') or {}
            pos = {}
            for n in graph.nodes:
                coord = bus_coords.get(str(n).lower())
                if coord is None:
                    pos = None
                    break
                pos[n] = (coord['x'], coord['y'])
            if pos is None and bus_coords:
                logger.warning(
                    "Bus coordinates are missing for some of the %d plotted buses; "
                    "falling back to an automatic graph layout.",
                    graph.number_of_nodes(),
                )
            plot_nx_graph(
                graph, 
                outpath=str(report_dir / "network.png"),
                figsize=(24, 24),
                pos=pos,
                highlight_cycles=True,
                plot_cycles=True,
                show=False,
            )
            # Plot the smaller graph of the feeder head
            N_head = 15
            root = [n for n, d in graph.in_degree() if d == 0][0]
            nodes_to_plot = list(nx.bfs_tree(graph, source=root).nodes)[:N_head]
            # Create a new subgraph with those nodes
            subgraph = graph.subgraph(nodes_to_plot)
            plot_nx_graph(
                subgraph,
                outpath=str(report_dir / "network_feeder_head.png"),
                figsize=(12, 12),
                highlight_cycles=True,
                edge_info=edge2name,
                node_color=['blue' if n in nodes_to_keep else 'gray' for n in subgraph.nodes],
                show=False,
            )

        # Find the closest connected transformer upstream
        circuit_dict, node_data = build_node_data(
            feeder_root.lower(), graph=graph, circuit_data=circuit_data, node_list=nodes_to_keep
        )
        # Save results
        self.circuit_data = circuit_data
        self.circuit_dict = circuit_dict
        self.node_data = node_data
        self.graph = graph_pruned
        return circuit_data

    def compute_nominal_voltages_from_turns_ratio(self, node_list, root_node_voltage):
        """
        Traverse the tree down and compute the nominal voltages at each node from the turns ratio.
        For DY and YD transformer we divide and multiply the turns ratio by sqrt(3) respectively.
        """
        to_visit = list(self.graph.successors(node_list[0]))
        self.node_data[node_list[0]]['nominal_voltage'] = np.array([root_node_voltage] * 3)
        while to_visit:
            k = to_visit.pop(0)
            j = next(self.graph.predecessors(k))
            a_jk = self.circuit_dict[(j, k)]['turns_ratio']
            if self.circuit_dict[(j, k)]['element_type'] in ('DY', 'DY_alt'):
                a_jk /= np.sqrt(3)
            elif self.circuit_dict[(j, k)]['element_type'] in ('YD', 'YD_alt'):
                a_jk *= np.sqrt(3)
            elif self.circuit_dict[(j, k)]['element_type'] in ('openD_openD_split', 'openD_openD', 'openY_openD_split',):
                if 'split_phase' in self.circuit_dict[(j, k)] and self.circuit_dict[(j, k)]['split_phase']:
                    a_jk = np.array([self.circuit_dict[(j, k)]['turns_ratio']] * 3)
                    wp = list(self.circuit_dict[(j, k)]['winding_pairing'])
                    if 'split_phase' in self.circuit_dict[(j, k)] and self.circuit_dict[(j, k)]['split_phase']:
                        (p_pri_sp, p_sec_sp) = self.circuit_dict[(j, k)]['split_phase']
                        for p in p_sec_sp:
                            a_jk[p - 1] *= 2
                        wp.remove((p_pri_sp, p_sec_sp))
                    # This one is subtle. The phase C voltage needs to consider that the ground 
                    # reference is now in the middle of AB leg. Draw a diagram and use trig to see why. 
                    for _, p_sec in wp:
                        for p in p_sec:
                            if p not in p_sec_sp:
                                a_jk[p - 1] *= 2 / np.sqrt(3)
            self.node_data[k]['nominal_voltage'] = self.node_data[j]['nominal_voltage'] / a_jk
            to_visit += list(self.graph.successors(k))
        nominal_voltages_list = [self.node_data[k]['nominal_voltage'] for k in node_list[1:]]
        return np.concatenate(nominal_voltages_list)



    # ------------------------------------------------------------------
    # Deprecated aliases for the pre-0.1 private-looking names.
    # ------------------------------------------------------------------
    def _dssParser(self) -> Dict[str, Any]:
        """Deprecated alias for :meth:`parse`."""
        warnings.warn(
            "DataLoader._dssParser() is deprecated; use DataLoader.parse() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.parse()

    def _format_circuit_data(self, feeder_root, plot_circuit=False, output_dir=None) -> list:
        """Deprecated alias for :meth:`to_circuit_data`."""
        warnings.warn(
            "DataLoader._format_circuit_data() is deprecated; use "
            "DataLoader.to_circuit_data() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_circuit_data(feeder_root, plot_circuit=plot_circuit, output_dir=output_dir)

def determine_valid_phases(circuit_data, node_list):
    """
    Determine which phases are physically present at each bus.

    Unlike the models, this function *requires* a 'phases' entry on every edge. For each
    edge in circuit_data it is interpreted as follows:
      - tuple of ints  (lines, reactors): both source and target carry exactly those phases
      - list of two tuples (non-consolidated transformer winding): source carries phases[0],
        target carries phases[1]; an empty tuple falls back to all three phases
      - absent / None / empty: raises ValueError, since the phases present at a bus cannot
        be inferred without it

    A bus carries the phases that its incoming (parent) edge delivers; the root, which has
    none, carries those of its outgoing edges. Every edge must draw a subset of the phases
    present at its source bus. The result does not depend on the order of circuit_data.

    Args:
        circuit_data: list of edge dicts produced by :meth:`DataLoader.to_circuit_data`
        node_list: ordered list of bus names (first entry is the root bus)

    Returns:
        tuple of two dicts:
          - bus_phases: maps each bus_name in node_list to a sorted list of valid phases
            (1-indexed). A bus that no edge touches is assumed to carry all three phases,
            with a warning.
          - branch_phases: maps each (source, target) pair to the sorted list of phases
            on that branch (the source-side phases for a transformer winding).

    Raises:
        ValueError: if an edge has no usable 'phases' entry, or if an edge claims phases
            that are not present at its source bus.
    """
    bus_phases = {}
    branch_phases = {}
    edge_phases = []

    for edge in circuit_data:
        src = edge['source']
        tgt = edge['target']
        phases = edge.get('phases')
        # 'element_name' is optional in circuit_data, so error messages must not assume it.
        edge_name = edge.get('element_name') or f"{src}->{tgt}"

        if (phases is None) or (not phases):
            raise ValueError(f"Edge {edge_name} is missing 'phases' information, which is required to determine bus valid phases.")
        elif isinstance(phases, list) and len(phases) == 2 and not isinstance(phases[0], int):
            # Transformer winding: phases = [src_phases_tuple, tgt_phases_tuple]
            src_ph = {p for p in phases[0] if p in (1, 2, 3)} or {1, 2, 3}
            tgt_ph = {p for p in phases[1] if p in (1, 2, 3)} or {1, 2, 3}
        else:
            # Line or reactor: flat tuple of phase ints, same for both ends
            src_ph = {p for p in phases if p in (1, 2, 3)}
            tgt_ph = src_ph

        branch_phases[(src, tgt)] = sorted(src_ph)
        edge_phases.append((edge_name, src, tgt, src_ph, tgt_ph))

    # Phases present at each bus, from the edges into it. They are all collected before any
    # check, so that an edge listed before its parent edge is checked against the right set.
    for _edge_name, _src, tgt, _src_ph, tgt_ph in edge_phases:
        bus_phases.setdefault(tgt, set()).update(tgt_ph)
    fed = set(bus_phases)
    for _edge_name, src, _tgt, src_ph, _tgt_ph in edge_phases:
        if src not in fed:
            bus_phases.setdefault(src, set()).update(src_ph)

    # Child elements should be subset of existing nodes' valid phases.
    for edge_name, src, _tgt, src_ph, _tgt_ph in edge_phases:
        if not src_ph.issubset(bus_phases[src]):
            raise ValueError(
                f"Bus '{src}': edge '{edge_name}' claims phases {sorted(src_ph)} "
                f"but bus parent element valid phases are restricted to {sorted(bus_phases[src])}"
            )

    bus_phases_clean = {}
    for bus in node_list:
        ph = bus_phases.get(bus, set())
        if ph:
            bus_phases_clean[bus] = sorted(ph)
        else:
            logger.warning(
                "Bus '%s' has no valid phases detected from circuit data. "
                "Assuming all phases are present.",
                bus,
            )
            bus_phases_clean[bus] = [1, 2, 3]

    return bus_phases_clean, branch_phases


if __name__ == "__main__":
    """Example usage: parse a shipped test feeder and summarize it."""
    data_loader = DataLoader("data/tiny_dy_dy_line_3phase")
    dss_data = data_loader.parse()
    circuit_data = data_loader.to_circuit_data(feeder_root="bus0", plot_circuit=False)
    print(f"Parsed {len(dss_data['buses'])} buses, {len(circuit_data)} branches")
    for branch in circuit_data:
        print(f"  {branch['source']:>6s} -> {branch['target']:<6s}  {branch['element_type']}")

    