"""
Data Loading and Preprocessing — Paper Section IV.A (case study) and III.B (scenarios)

Parses OpenDSS `.dss` files (via altdss-python) and produces the network data
structures and scenario samples consumed by the optimizer.

Key responsibilities:
  - `_dssParser()`: Compile the feeder's master .dss file; extract buses,
    lines, transformers, loads, and capacitors from OpenDSS via altdss.
  - `_format_circuit_data()`: Convert DSS objects to the `circuit_data` list
    expected by `three_phase_lindistflow` (handles DY/YD transformer
    winding lookups, open-delta split-phase, parallel edge consolidation).
  - `get_circuit_model()`: Compute and cache the 3-phase LinDistFlow matrices
    R̃, X̃, C̃_0, C̃_S for a given feeder root (paper Appendix A / Section II.A).
  - `get_scenario_data_by_sampling(N, seed)`: Draw N i.i.d. hourly snapshots
    from the feeder's annual load/PV/voltage data (Section III.B); each
    snapshot contains a load profile, PV capacity factor, and feeder-head
    voltage.
  - `get_scenario_data_by_range(t0, t1)`: Fetch contiguous timeseries for
    deterministic simulation (used by `run_pvhc_simulation.py`).
  - `disaggregate_loads()`: Distribute feeder-head MW/MVAR to individual buses
    proportionally using per-bus connected-load fractions.
  - `get_annual_consumption()`: Estimate annual kWh per load for per-customer
    PV capacity limits ζ̄_c (Eq. 12 constraint x ≤ ζ̄_c).

Example feeder registered in `constants.FEEDERS` (see README for how to point
this at your own feeder instead): EPRI's public "Ckt5" distribution test
circuit — see `data/example_feeder/opendss/THIRD_PARTY_LICENSE.txt` for its
origin and license.
"""

import os
import sys
from typing import Dict, Any, Optional
from collections import OrderedDict
import re
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import networkx as nx
from altdss import altdss
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import three_phase_lindistflow as ldf
np.set_printoptions(precision=3)

from constants import * # This file contains global constants


class DataLoader:
    def __init__(self, feeder_name: str):
        """
        Initialize DataLoader for a specific feeder using altdss-python.

        Args:
            feeder_name: Name of the feeder to simulate; must be registered in
                `constants.FEEDERS`.
        """
        assert feeder_name in FEEDERS, f"Invalid feeder name: {feeder_name}. Available feeders: {list(FEEDERS)}"
        feeder_config = FEEDERS[feeder_name]

        parent_dir = Path(__file__).parent.parent
        self.data_folder = parent_dir / feeder_config["data_path"]
        self.base_path = self.data_folder / "opendss"
        self.master_filename = feeder_config["master_filename"]
        self.name = feeder_name
        self.feeder_name = feeder_name
        self.dss_data = {}
        self.circuit_data = None
        self.node_data = None
        self.load_profiles = {}
        self.pv_profiles = {}

        if not os.path.exists(self.base_path):
            raise ValueError(f"Invalid input data path: {self.base_path}")

        self.network_data_dict = None

    def _dssParser(self) -> Dict[str, Any]:
        """
        Execute .dss files using Master.dss in self.base_path.
        Parse the feeder data using altdss-python and extract relevant parameters
        Returns:
            Dictionary containing parsed network data
        """
        # Check folder/file existence
        if not self.base_path.exists():
            raise FileNotFoundError(f"DSS file folder not found at {self.base_path}")
        master_file = self.base_path / self.master_filename
        if not master_file.exists():
            raise FileNotFoundError(f"{self.master_filename} not found in {self.base_path}")

        # Change to the feeder directory to handle relative paths properly
        original_dir = os.getcwd()
        os.chdir(self.base_path)

        # Clear any existing circuit and compile the master file. Bus coordinates
        # are whatever the master file itself loads (e.g. a `Buscoords` command) —
        # no separate coordinate file is required.
        altdss("Clear")
        altdss(f"Compile {master_file.name}")
        altdss("CalcVoltageBases")
        altdss("calcv")
        altdss('calcincmatrix_o')
        # Branches isolated from feeder_root (e.g. normally-open tie switches to other
        # sub-feeders sharing the same substation) are expected and are dropped later
        # when the graph is truncated to the subtree rooted at feeder_root.
        isolated = altdss.Topology.AllIsolatedBranches
        if isolated != ['NONE']:
            print(f"[Info] {len(isolated)} branch(es) isolated from the main circuit graph "
                  f"(expected for normally-open ties): {isolated}")

        # Check if compilation was successful
        if altdss.Error.Number != 0:
            raise RuntimeError(f"AltDSS compilation error: {altdss.Error.Description}")
        # Restore original directory
        os.chdir(original_dir)

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
            "voltage_source": altdss.Vsource.BasekV.to_array()[0],
        }

        # Store the parsed data
        self.dss_data = dss_data
        return dss_data

    def _extract_buses(self) -> Dict[str, Any]:
        """Extract bus information using AltDSS interface"""
        buses = {}

        # Get all bus names
        print(f"Loading {len(altdss.Bus)} buses")
        for bus in altdss.Bus:
            buses[bus.Name] = {
                "name": bus.Name,
                "base_kv": bus.kVBase,
                "nodes": bus.Nodes.tolist(),
                "coordinate": (bus.X, bus.Y),
            }
        return buses

    def _split_bus_and_phases(self, bus_str):
        """
        Split an OpenDSS bus name like 'BusName.1.2.3' into:
        (base_name, [phase_numbers]), e.g. ('BusName', [1, 2, 3])
        """
        accidental_suffix_match = re.search(r"\.\d+$", bus_str)
        if accidental_suffix_match and not re.match(r"^.+\.(\d(\.\d)*)$", bus_str):
            # Remove accidental numeric suffix
            cleaned_bus_str = re.sub(r"\.\d+$", "", bus_str)
            print(f"[Warning] Bus name '{bus_str}' had trailing numeric suffix removed -> '{cleaned_bus_str}'")
            bus_str = cleaned_bus_str
        
        match = re.match(r'^(?P<name>[^\s\.]+?)(?:\.(?P<phases>(?:\d(?:\.\d)*)))?$', bus_str)
        if match:
            base_name = match.group("name")
            phases_str = match.group("phases")
            if phases_str:
                phases = tuple(map(int, phases_str.split(".")))
                for p in phases:
                    assert p in (0, 1, 2, 3), phases
            else:
                phases = tuple()
            if '.' in base_name:
                raise ValueError(f"Invalid bus string format: {bus_str}")
            return base_name, phases
        else:
            raise ValueError(f"Invalid bus string format: {bus_str}")

    def _extract_lines(self) -> Dict[str, Any]:
        """Extract line information using AltDSS interface.

        Reads impedance (RMatrix/XMatrix/CMatrix) and current rating (NormAmps)
        directly off the Line object rather than through its LineCode, since
        OpenDSS resolves these the same way regardless of whether the line was
        defined via `linecode=` or `geometry=` — this keeps line extraction
        correct for either convention without needing a separate lookup.
        """
        lines = {}

        print(f"Loading {len(altdss.Line)} lines")
        n_disabled = 0
        for line in altdss.Line:
            if not line.Enabled:
                # A disabled line (e.g. a normally-open tie switch) carries no
                # current and must not be modeled as a closed circuit edge.
                n_disabled += 1
                continue
            lines[line.Name] = {
                "name": line.Name,
                "length": line.Length,
                "units": line.Units,
                "num_phases": line.Phases,
                "line_type": line.LineType_str,
                "RMatrix": line.RMatrix,
                "XMatrix": line.XMatrix,
                "CMatrix": line.CMatrix,
                "norm_amps": line.NormAmps,
            }
            bus1, phases1 = self._split_bus_and_phases(line.Bus1)
            bus2, phases2 = self._split_bus_and_phases(line.Bus2)
            # A bus reference with no explicit ".1.2.3"-style phase suffix implicitly
            # means phases 1..num_phases, per OpenDSS convention.
            if not phases1:
                phases1 = tuple(range(1, line.Phases + 1))
            if not phases2:
                phases2 = tuple(range(1, line.Phases + 1))
            assert (len(phases1) == lines[line.Name]["num_phases"]) and (phases1 == phases2), \
                (phases1, phases2, lines[line.Name]["num_phases"])
            lines[line.Name]["bus1"] = bus1
            lines[line.Name]["bus2"] = bus2
            lines[line.Name]["phases"] = phases1
        if n_disabled:
            print(f"[Info] Skipped {n_disabled} disabled line(s) (open switches).")
        return lines

    def _extract_transformers(self) -> Dict[str, Any]:
        """Extract transformer information using AltDSS interface"""
        transformers = {}
        print(f"Loading {len(altdss.Transformer)} transformers")
        for xfmr in altdss.Transformer:  # iterable, like `for bus in altdss.Bus`
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
                "buses": [],
                "phases": [],
            }
            for bus in list(xfmr.Buses):
                bus_name, phases = self._split_bus_and_phases(bus)
                if not phases:
                    phases = tuple(range(1, xfmr.Phases + 1))
                transformers[xfmr.Name]["buses"].append(bus_name)
                transformers[xfmr.Name]["phases"].append(phases)

        return transformers

    def _extract_reactors(self) -> Dict[str, Any]:
        """Extract reactor information using AltDSS interface"""
        reactors = {}
        print(f"Loading {len(altdss.Reactor)} reactors")
        for reactor in altdss.Reactor:
            reactors[reactor.Name] = {
                "name": reactor.Name,
                "num_phases": reactor.Phases,
                "R": reactor.R,
                "X": reactor.X,
                "enabled": reactor.Enabled,
            }
            bus1, phases1 = self._split_bus_and_phases(reactor.Bus1)
            bus2, phases2 = self._split_bus_and_phases(reactor.Bus2)
            if not phases1:
                phases1 = tuple(range(1, reactor.Phases + 1))
            if not phases2:
                phases2 = tuple(range(1, reactor.Phases + 1))
            assert (len(phases1) == reactors[reactor.Name]["num_phases"]) and (phases1 == phases2), \
                (phases1, phases2, reactors[reactor.Name]["num_phases"])
            reactors[reactor.Name]["bus1"] = bus1
            reactors[reactor.Name]["bus2"] = bus2
            reactors[reactor.Name]["phases"] = phases1
        return reactors

    def _extract_swtcontrol(self) -> Dict[str, Any]:
        """Extract device information using AltDSS interface"""
        swt_ctrls = {}
        print(f"Loading {len(altdss.SwtControl)} switch controllers")
        for swt in altdss.SwtControl:
            swt_ctrls[swt.Name] = {
                "name": swt.Name,
                "swt_obj": swt.SwitchedObj,
                "normal": swt.Normal
            }
        return swt_ctrls

    def _extract_loads(self) -> Dict[str, Any]:
        """Extract load information using AltDSS interface"""
        loads = {}

        print(f"Loading {len(altdss.Load)} loads")
        for load in altdss.Load:
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
                "yearly_shape": load.Yearly.Name if load.Yearly is not None else None,
            }
            bus_name, phases = self._split_bus_and_phases(load.Bus1)
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

        print(f"Loading {len(altdss.Capacitor)} capacitors")
        for cap in altdss.Capacitor:
            capacitors[cap.Name] = {
                "name": cap.Name,
                "num_phases": cap.Phases,
                "kvar": cap.kvar,
                "kv": cap.kV,
                "enabled": cap.Enabled,
                "connection": cap.Conn.name,
            }
            bus_name, phases = self._split_bus_and_phases(cap.Bus1)
            capacitors[cap.Name]["bus"] = bus_name
            capacitors[cap.Name]["phases"] = phases
            
        return capacitors

    def _extract_line_geometry(self) -> Dict[str, Any]:
        """Extract line geometry information"""
        geometries = {}
        print(f"Loading {len(altdss.LineCode.Name)} line geometries")
        for data in altdss.LineCode:
            geometries[data.Name] = {
                "XMatrix": data.RMatrix,
                "RMatrix": data.XMatrix,
                "NPhases": data.NPhases
            }
        return geometries

    def get_load_info(self,):
        """
        Extract load information from the DSS data, restricted to buses kept in
        the feeder_root subtree (self.node_data) — the DSS file may contain
        loads on other, unrelated sub-feeders sharing the same substation.
        """
        if not self.dss_data:
            self.dss_data = self._dssParser()
        assert self.node_data, "circuit_data is not loaded. Call get_circuit_model() first."
        load_info = []
        for load in self.dss_data['loads'].values():
            if load['bus'].lower() not in self.node_data:
                continue
            if load['num_phases'] == 1:
                phases = [p for p in load['phases'] if p != 0]
                if len(phases) == 1:
                    assert load['connection'] == 'wye', load
                    load_info.append({'bus': load['bus'], 'phases': (phases[0], 0)})
                elif len(phases) == 2:
                    assert load['connection'] == 'delta', load
                    load_info.append({'bus': load['bus'], 'phases': tuple(phases)})
                else:
                    raise ValueError(f"Unsupported phase configuration: {load['phases']} for {load}")
            elif load['num_phases'] == 3:
                # Assumption: 3-phase loads are balanced (dss files do not specify)
                assert (len([p for p in load['phases'] if p != 0]) == 3), load['phases']
                if load['connection'] == 'wye':
                    for p in (1, 2, 3):
                        load_info.append({'bus': load['bus'], 'phases': (p, 0)})
                elif load['connection'] == 'delta':
                    for phases in ((1, 2), (2, 3), (3, 1)):
                        load_info.append({'bus': load['bus'], 'phases': phases})
                else:
                    raise ValueError(f"Unsupported connection type: {load['connection']} for {load}")
            else:
                raise ValueError(f"Unsupported number of phases: {load['num_phases']} for {load}")
        return load_info

    def _format_circuit_data(self, feeder_root, plot_circuit=False) -> list:
        """
        Map dss data into 3-phase LindistFlow format
        returns:
            circuit_data: list of dicts, each dict has keys:
                - source: str, source bus name
                - target: str, target bus name
                - phases: tuple of int, e.g. (1, 2) for AB, (1, 0) for A only
                - R: np.ndarray, resistance matrix for the line
                - X: np.ndarray, reactance matrix for the line
                - C: np.ndarray, capacitance matrix for the line
                - S_rating: float, apparent power rating of the line (if available)
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

        # Begin main function content
        assert feeder_root, "feeder_root must be specified to format circuit data."
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
            # OpenDSS gives RMatrix/XMatrix in ohms per unit of the line's own length unit, the
            # unit of line['length'], so their product is the series impedance in ohms.
            R_line = np.asarray(line['RMatrix'], dtype=float) * line['length']
            X_line = np.asarray(line['XMatrix'], dtype=float) * line['length']
            if len(R_line) == 9:
                edge_data['series_impedance'] = R_line.reshape(3,3) + 1j * X_line.reshape(3,3)
            elif len(R_line) == 4:
                assert len(line['phases']) == 2, (line['phases'], line['name'])
                # For non-existent lines, make the self-impedance infinity (LARGE).
                edge_data['series_impedance'] = np.ones((3, 3), dtype=complex) * (1 + 1j) * LARGE
                R, X = R_line.reshape(2,2), X_line.reshape(2,2)
                p1, p2 = line['phases'][0] - 1, line['phases'][1] - 1
                edge_data['series_impedance'][p1, p1] = R[0, 0] + 1j * X[0, 0]
                edge_data['series_impedance'][p2, p2] = R[1, 1] + 1j * X[1, 1]
                edge_data['series_impedance'][p1, p2] = R[0, 1] + 1j * X[0, 1]
                edge_data['series_impedance'][p2, p1] = R[1, 0] + 1j * X[1, 0]
            elif len(R_line) == 1:
                # Extract scalar to avoid shape-(1,) array assignment to scalar index
                Z = complex(R_line[0] + 1j * X_line[0])
                # For non-existent lines, make the self-impedance infinity (LARGE).
                edge_data['series_impedance'] = np.ones((3, 3), dtype=complex) * (1 + 1j) * LARGE
                for p in line['phases']:
                    if p == 0: continue
                    edge_data['series_impedance'][p-1, p-1] = Z
            else:
                raise ValueError(f"Unexpected RMatrix length: {len(R_line)}")
            if (edge_data['series_impedance'] == 0).all():
                print(f"[Warning] Zero impedance detected in branch: {line['name']}.")
                
            # Line power rating, from the current rating OpenDSS itself resolved
            # for this line (via its LineCode or Geometry, whichever it uses).
            # bus1 may be undefined if this line only ties to a sub-feeder outside
            # feeder_root's subtree (e.g. a normally-open connector) — such lines are
            # dropped later when the graph is pruned to feeder_root, so an unrated
            # (unconstrained) placeholder here is harmless.
            bus1_info = self.dss_data['buses'].get(line['bus1'])
            edge_data['S_rating'] = line['norm_amps'] * bus1_info['base_kv'] if bus1_info else None
            circuit_data.append(edge_data)
            
        # Winding pairings
        # Textbook: AB – AN, BC – BN, CA – CN
        # Caltech (alt): AC – AN, BA – BN, CB – CN
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
        for xfmr in self.dss_data['transformers'].values():
            # Make sure the transformer windings buses are in the form of src, tgt, tgt, ...
            bus_set = set(xfmr['buses'])
            assert len(bus_set) == 2, (xfmr['buses'], xfmr['name'])
            src, tgt = xfmr['buses'][0], (bus_set - {xfmr['buses'][0]}).pop()
            if (src in remove_nodes)  or (tgt in remove_nodes):
                continue
            assert all([b != src for b in xfmr['buses'][1:]]), (xfmr['buses'], src, xfmr['name'])
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
                }
                # Type of transformer, i.e. 'element_type'
                if tuple(edge_data["winding_connections"]) in winding_connection_mapping:
                    edge_data['element_type'] = winding_connection_mapping[tuple(edge_data["winding_connections"])]
                    if edge_data['element_type'] in ('DY', 'YD'):
                        winding_pair = (tuple(sorted(set(xfmr['phases'][0]) - {0})), tuple(sorted(set(xfmr['phases'][t]) - {0})))
                        # For 3-phase transformers, default to (assume) the alternative winding pairing.
                        if len(winding_pair[0]) == len(winding_pair[1]) == 3:
                            edge_data['element_type'] = edge_data['element_type'] + '_alt'
                        # Otherwise, for a single pair of windings, use the lookup table.
                        elif winding_pair in xfmr_lookup[edge_data['element_type']]:
                            edge_data['element_type'] = xfmr_lookup[edge_data['element_type']][winding_pair]
                        else:
                            raise ValueError(f"Unknown transformer winding connection {winding_pair} for {xfmr}")  
                else:
                    raise ValueError(f"Unknown transformer winding connection: {xfmr['winding_connections']}")  
                # Calculate series impedance
                z_pu = (xfmr['pcrRs'][0] + xfmr['pcrRs'][t] + 1j * xfmr['xhl']) * 1e-2
                if xfmr['kvas'][0] == 0:
                    z_base = 0
                    print(f"[Warning] Transformer {xfmr['name']} has zero kVA rating. Assuming series impedance is zero.")
                else:
                    z_base = xfmr['kvs'][0] ** 2 / xfmr['kvas'][0] * 1e3     # Normalize to primary winding voltage
                # For single-phase transformers, we still use the diagonal impedance matrix.
                # This is not physically correct, but does not affect the solution, since the fictitious
                # (non-existent) winding ports will always have zero current injection (constrained by the load injections).
                edge_data['series_impedance'] = np.eye(3) * z_pu * z_base
                # Turns ratio. Note that DSS kV ratings are always line-to-line.
                # For 3-phase transformers, the turns ratio needs to be adjusted based on the winding connection.
                # https://dss-extensions.org/dss-format/Transformer.html
                if (edge_data['element_type'] in ('DY', 'DY_alt')) and (xfmr['num_phases'] == 3):
                    edge_data['turns_ratio'] *= np.sqrt(3)
                elif (edge_data['element_type'] in ('YD', 'YD_alt')) and (xfmr['num_phases'] == 3):
                    edge_data['turns_ratio'] /= np.sqrt(3)
                circuit_data.append(edge_data)

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
            # Impedance only on the switch's own phases, so that it matches 'phases'
            # (three_phase_lindistflow rejects a line whose impedance has phases it does not declare).
            edge_data['series_impedance'] = np.zeros((3, 3), dtype=complex)
            for p in swt_obj['phases']:
                if p == 0: continue
                edge_data['series_impedance'][p-1, p-1] = swt_obj['R'] + 1j * swt_obj['X']
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
                    print(f"[Warning]: All {k} should be the same for {[ele['element_name'] for ele in l]}. Picking the first arbitrarily.")
            if l[0]['element_type'] in ('line',):
                # Multiple single-phase lines sharing the same named bus pair, each
                # carrying a distinct phase (e.g. secondary drops modeled as separate
                # per-phase branches rather than one multi-phase line) — merge into a
                # single edge with each phase's own impedance on the diagonal.
                all_phases = [p for e in l for p in e['phases'] if p != 0]
                assert len(all_phases) == len(set(all_phases)), \
                    f"Overlapping phases in duplicate line group for {[e['element_name'] for e in l]}"
                combined_impedance = np.ones((3, 3), dtype=complex) * (1 + 1j) * LARGE
                for e in l:
                    for p in e['phases']:
                        if p == 0: continue
                        combined_impedance[p-1, p-1] = e['series_impedance'][p-1, p-1]
                circuit_data.append({
                    'element_name': "|".join(e['element_name'] for e in l),
                    'source': l[0]['source'],
                    'target': l[0]['target'],
                    'turns_ratio': 1.0,
                    'element_type': 'line',
                    'phases': tuple(sorted(all_phases)),
                    'series_impedance': combined_impedance,
                    'S_rating': l[0]['S_rating'],
                })
                continue
            # Combine parallel edges into a single element
            consolidated_edge_data = {
                "element_name": "|".join([edge_data['element_name'] for edge_data in l]),
                "source": l[0]['source'],
                "target": l[0]['target'],
                'S_rating': l[0]['S_rating'],
            }
            # Here we only handle open type transformers. Other types are not implemented yet and will throw an error.
            # Single-phase and standard three-phase transformers are handled above.
            # Primary conection type (delta, wye)
            assert len(set([edge_data['winding_connections'][0] for edge_data in l])) == 1, \
                f"All primary winding connections should be the same for {[ele['element_name'] for ele in l]}"
            conn_pri = l[0]['winding_connections'][0]
            # Store winding pairings in a dictionary
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
            # Detect standard 3-phase transformer
            if len(winding_pairing) == 3:
                assert len({edge_data['element_type'] for edge_data in l}) == 1, \
                    f"All element types should be the same for {[ele['element_name'] for ele in l]}"
                consolidated_edge_data['element_type'] = l[0]['element_type']
                if consolidated_edge_data['element_type'] in ('YY', 'YD', 'YD_alt'):
                    series_impedance_list = [series_impedance[p][0][0,0] for p in ((1,), (2,), (3,))]
                elif consolidated_edge_data['element_type'] in ('DD', 'DY', 'DY_alt'):
                    series_impedance_list = [series_impedance[p][0][0,0] for p in ((1,2), (2,3), (1,3))]
                consolidated_edge_data["series_impedance"] = np.diag(series_impedance_list)
                assert len({edge_data['turns_ratio'] for edge_data in l}) == 1, \
                    f"Non-equal turns_ratio is not implemented yet."
                consolidated_edge_data['turns_ratio'] = l[0]['turns_ratio']
                circuit_data.append(consolidated_edge_data)
                continue
            # Detect split phase. Treat all three-winding transformers as split-phase
            split_phase = None
            turns_ratios_list, series_impedance_list = [], []
            for pri, sec_list in winding_pairing.items():
                if len(sec_list) == 1:
                    turns_ratios_list.append(turns_ratios[pri][0])
                    series_impedance_list.append(series_impedance[pri][0][0, 0])    # Assumes: edge_data['series_impedance'] = np.eye(3) * z_pu * z_base
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
            series_impedance_list += [0]
            consolidated_edge_data["series_impedance"] = np.diag(series_impedance_list)
            circuit_data.append(consolidated_edge_data)

        # Network sanity checks: connectivity and duplicate-edge detection
        edge2name = ldf.to_edge_list(circuit_data, include_name=True)
        edge_list = list(edge2name.keys())
        edge2index = {e: i for i, e in enumerate(edge_list)}
        graph = nx.DiGraph(edge_list)
        isolated_nodes = list(nx.isolates(graph))
        unknown_buses = [n for n in graph.nodes if n not in self.dss_data['buses']]
        if len(circuit_data) != len(graph.edges):
            raise RuntimeError(f"Found {len(circuit_data) - len(graph.edges)} duplicate (parallel) edges.")
        print(f"Number of edges, nodes: {len(graph.edges)}, {len(graph.nodes)}")
        if unknown_buses:
            print(f"[Warning] Unknown buses detected: {unknown_buses}.")
        print(f"[Info] Number of isolated nodes: {len(isolated_nodes)}")
        graph.remove_nodes_from(isolated_nodes)
        components = list(nx.connected_components(graph.to_undirected()))
        print(f"[Info] Number of connected components: {len(components)}")
        print(f"[Info] Component sizes: {[len(c) for c in components]}")
        cycles = nx.cycle_basis(graph.to_undirected())
        print(f"[Info] Number of cycles: {len(cycles)}.")
        if len(cycles):
            os.makedirs(f"temp/data_loader/{self.name}", exist_ok=True)
            with open(f"temp/data_loader/{self.name}/cycle.txt", "w") as f:
                for c, cycle in enumerate(cycles):
                    if c > 0: 
                        f.write('\n')
                    f.write(f"Cycle {c}:\n")
                    f.write(",".join(cycle))
            
        # Take the subgraph at feeder_root
        try:
            nodes_to_keep = list(nx.bfs_tree(graph, source=feeder_root.lower()).nodes)
        except Exception as e:
            print(f"[Error] Failed to find feeder root: {e}")
            raise e
        # Build the pruned graph in the parent graph's (OpenDSS) order. Iterating a subgraph view
        # goes through a set of bus names when it keeps fewer than half of the nodes (Ckt5 keeps
        # 1196 of 2997), and that order changes from one Python process to the next.
        keep = set(nodes_to_keep)
        graph_pruned = nx.DiGraph()
        graph_pruned.add_nodes_from((n, graph.nodes[n]) for n in graph.nodes if n in keep)
        graph_pruned.add_edges_from((u, v, graph.edges[u, v]) for u, v in graph.edges if u in keep and v in keep)
        circuit_data = [circuit_data[edge2index[e]] for e in graph_pruned.edges]

        # Plot circuit (restricted to nodes with known coordinates — some nodes in
        # the subtree, e.g. internal terminals of normally-open switches, may not
        # be real DSS buses with a position).
        if plot_circuit:
            pos = {
                n: v['coordinate'] for n, v in self.dss_data['buses'].items()
                if n in nodes_to_keep
            }
            plottable = graph_pruned.subgraph(pos.keys())
            if len(pos) < len(nodes_to_keep):
                print(f"[Info] {len(nodes_to_keep) - len(pos)} node(s) in the feeder subtree "
                      f"have no coordinate and are excluded from the network plot.")
            ldf.plot_nx_graph(
                plottable,
                outpath=f"temp/data_loader/{self.name}/network.png",
                figsize=(24, 24),
                pos=pos,
                highlight_cycles=True,
                plot_cycles=True,
                show=False,
            )
            # Plot the smaller graph of the feeder head
            N_head = 15
            nodes_to_plot = [n for n in nodes_to_keep if n in pos][:N_head]
            subgraph = graph_pruned.subgraph(nodes_to_plot)
            ldf.plot_nx_graph(
                subgraph,
                outpath=f"temp/data_loader/{self.name}/network_feeder_head.png",
                figsize=(12, 12),
                highlight_cycles=True,
                edge_info=edge2name,
                node_color=['blue' if n in nodes_to_keep else 'gray' for n in subgraph.nodes],
                show=False,
            )

        # Find the closest connected transformer upstream
        circuit_dict, node_data = ldf.build_node_data(
            feeder_root.lower(), graph=graph, circuit_data=circuit_data, node_list=nodes_to_keep
        )
        # Save results
        self.circuit_data = circuit_data
        self.circuit_dict = circuit_dict
        self.node_data = node_data
        self.graph = graph_pruned

    def get_circuit_model(self, feeder_root=None, recompute=False, save=True, plot_circuit=False):
        """
        Return the 3-phase LinDistFlow matrices for this feeder (paper Section
        II.A / Appendix A). R, X matrices are in Ohms.

        :param feeder_root: str, root node of the circuit. The root node in
            the DSS file is not always the bus at which the feeder-head
            voltage data applies, so the graph is truncated to the subtree
            rooted at feeder_root.
        :param plot_circuit: if True, also render a diagnostic network layout
            via matplotlib (requires a working GUI/interactive backend).
            Off by default so this data-loading step has no plotting side
            effects and never touches a display backend.
        Returns:
            network_data = {
                "node_list": node_list,
                "edge_list": edge_list,
                "R": R,
                "X": X,
                "C_tilde_0": C_tilde_0,
                "C_tilde_S": C_tilde_S,
                "n_buses": N,
                "edge2idx": dict,
                "node2idx": dict,
                "branch_pwr_ratings": float or None, if None then the edge has infinite power rating
            }
        """
        # Return cached network data if available in memory (unless recompute is True)
        if self.network_data_dict:
            return self.network_data_dict

        # Always run these to ensure some intermediate data is available
        if not self.dss_data:
            self.dss_data = self._dssParser()

        if not self.circuit_data:
            self._format_circuit_data(feeder_root, plot_circuit=plot_circuit)

        # Check if temp/cache/circuit_model/{feeder_name} has the precomputed model
        cache_dir = Path(__file__).parent.parent / "temp" / "cache" / "circuit_model" / self.feeder_name
        cache_file = cache_dir / "network_data.npz"

        if not recompute and cache_file.exists():
            try:
                print(f"Loading cached circuit model from {cache_file}")
                loaded_data = np.load(cache_file, allow_pickle=True)
                network_data = {
                    "node_list": loaded_data['node_list'].tolist(),
                    "edge_list": [tuple(edge) for edge in loaded_data['edge_list']],
                    "R": loaded_data['R'],
                    "X": loaded_data['X'],
                    "C_tilde_0": loaded_data['C_tilde_0'],
                    "C_tilde_S": loaded_data['C_tilde_S'],
                    "n_buses": int(loaded_data['n_buses']),
                    "edge2idx": loaded_data['edge2idx'].item(),
                    "node2idx": loaded_data['node2idx'].item(),
                    "branch_pwr_ratings": loaded_data['branch_pwr_ratings'].tolist(),
                    "load_info": loaded_data['load_info'].tolist()
                }
                self.N = network_data['n_buses']
                self.network_data_dict = network_data
                print(f"Successfully loaded cached model with {network_data['n_buses']} buses")
                return network_data
            except Exception as e:
                print(f"Warning: Failed to load cached model: {e}")
                print("Recomputing circuit model...")

        if recompute:
            print("Recomputing circuit model (recompute=True)...")

        C_tilde_0, R, X, C_tilde_S, node_list, edge_list = \
            ldf.three_phase_lindistflow_real_wrapper(self.circuit_data, self.node_data)
        N = len(node_list) - 1
        self.N = N
        network_data = {
            "node_list": node_list,
            "edge_list": edge_list,
            "R": R,
            "X": X,
            "C_tilde_0": C_tilde_0,
            "C_tilde_S": C_tilde_S,
            "n_buses": N,
        }

        network_data["edge2idx"] = {edge: i for i, edge in enumerate(edge_list)}
        network_data["node2idx"] = {node: i for i, node in enumerate(node_list)}

        # Build branch maximum power list in the order of edge_list
        network_data["branch_pwr_ratings"] = [None] * len(edge_list)
        for edge_data in self.circuit_data:
            i = network_data["edge2idx"][(edge_data['source'], edge_data['target'])]
            network_data["branch_pwr_ratings"][i] = edge_data['S_rating']

        # Add load_info
        network_data["load_info"] = self.get_load_info()

        # Cache result in memory
        self.network_data_dict = network_data

        # Store the network data in temp/cache/circuit_model/{feeder_name}
        cache_dir.mkdir(parents=True, exist_ok=True)
        if save:
            print(f"Saving circuit model to {cache_file}")
            np.savez_compressed(
                cache_file,
                node_list=np.array(network_data['node_list'], dtype=object),
                edge_list=np.array(network_data['edge_list'], dtype=object),
                R=network_data['R'],
                X=network_data['X'],
                C_tilde_0=network_data['C_tilde_0'],
                C_tilde_S=network_data['C_tilde_S'],
                n_buses=network_data['n_buses'],
                edge2idx=network_data['edge2idx'],
                node2idx=network_data['node2idx'],
                branch_pwr_ratings=np.array(network_data['branch_pwr_ratings'], dtype=object),
                load_info=np.array(network_data['load_info'], dtype=object),
            )

        return network_data

    def get_branch_sensitivity_matrices(self):
        """
        Return (C_inv, C_coef_inv, D_r, D_x, c_0) for the Appendix B dual-variable
        sensitivity analysis (`scripts/dual_analysis.py`), in the same node/edge
        ordering as `get_circuit_model()`: the per-branch block-diagonal
        resistance/reactance matrices (D_r, D_x) and the incidence-based
        matrices (C_inv from nodal injections to branch flows; C_coef_inv, c_0
        from branch voltage drops to nodal voltages) of the wye-wye reference
        model `three_phase_lindistflow.gan_hxu_3pldf`.

        That model uses a scalar per-branch turns ratio, which is exact for line
        segments and for 3-phase transformers with a uniform per-phase ratio
        (YY, DY, YD), but is not a valid representation for split-phase or
        open-delta transformers with per-phase-differentiated ratios. Check that
        your feeder has no such transformers before relying on this for a new
        feeder (see README).
        """
        assert self.network_data_dict, "Call get_circuit_model() first."
        _, _, _, C_inv, _, edge_list, C_coef_inv, D_r, D_x, c_0 = ldf.gan_hxu_3pldf(
            self.circuit_data, node_list=self.network_data_dict["node_list"], return_additional=True
        )
        if [tuple(e) for e in edge_list] != [tuple(e) for e in self.network_data_dict["edge_list"]]:
            raise RuntimeError(
                "The cached circuit model's edge order differs from this parse; delete "
                "temp/cache/circuit_model/ (and temp/cache/limits/) and rebuild."
            )
        return C_inv, C_coef_inv, D_r, D_x, c_0
    
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

    def get_annual_consumption(self, load_info):
        """
        Estimate annual kWh per load from its nameplate rating and its assigned
        OpenDSS `yearly` load shape (if any), rather than from a separate
        billing-data file: annual_kWh = kW_rated * mean(yearly load-shape
        multiplier) * 8760h. This makes the estimate self-contained (works for
        any feeder with load shapes, real or synthetic) at the cost of not
        reflecting real per-customer billing variation.

        Returns:
            load_info: updated load_info dict with annual consumption added
        """
        assert self.dss_data, "DSS data is not loaded. Call get_circuit_model() first."
        shape_mean_cache: Dict[str, float] = {}
        loads_by_name = self.dss_data['loads']
        for load in load_info:
            dss_load = loads_by_name[load['name']]
            shape = dss_load.get('yearly_shape')
            if shape:
                if shape not in shape_mean_cache:
                    shape_mean_cache[shape] = float(np.mean(altdss.LoadShape[shape].PMult))
                load_factor = shape_mean_cache[shape]
            else:
                load_factor = DEFAULT_ANNUAL_LOAD_FACTOR
            load['annual_consumption'] = dss_load['kw'] * load_factor * 8760.0
        return load_info

    """Time-series data processing below"""
    def _get_pv_generation(self, file_path: str, t0, t1, t_column='t', utc2LA=False):
        """
        Parse and store PV generation profile data

        Args:
            dir_name: Directory containing PV data
            file_name: Name of the PV data file
            format: Format of the returned data (e.g., "df" for DataFrame, "array" for NumPy array)
        """
        # Assume CSV format with timestamps and bus columns
        df = pd.read_csv(file_path, parse_dates=[t_column])
        # Assumes UTC time
        if utc2LA:
            df["t"] = df["t"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)

        # Convert t0, t1 to datetime
        t0_dt = pd.to_datetime(t0)
        t1_dt = pd.to_datetime(t1)
        # Filter by time range
        mask = (df[t_column] >= t0_dt) & (df[t_column] <= t1_dt)
        df = df.loc[mask].reset_index(drop=True)
        # Rename columns to standardized names
        df.rename(columns={t_column: 't'}, inplace=True)
        return df

    def get_feeder_head_data(self, t0, t1, v_root_nom, voltage_noise_pu=0.005, regulator_band_pu=0.01, seed=0):
        """
        Build a feeder-head MW/MVAR/voltage timeseries from the circuit's own
        annual load shapes, in place of real feeder-head SCADA/AMI telemetry.

        Each load's nameplate (kW, kvar) is scaled by the per-unit multiplier
        of its assigned `yearly` load shape (falling back to
        DEFAULT_ANNUAL_LOAD_FACTOR's shape-less loads, held flat), then summed
        to give an aggregate active/reactive power timeseries with the same
        real diurnal/seasonal structure the load shapes encode. There is no
        real historical feeder-head voltage measurement to substitute for, so
        the source voltage is held at a regulated setpoint (FEEDERS'
        `v_root_setpoint_pu`, default 1.0) with small (seeded, clearly synthetic)
        noise kept within the regulator's band, to avoid a perfectly flat
        voltage series.

        Args:
            t0, t1: Start/end of the period to return, as parseable date strings.
            v_root_nom: Nominal line-to-neutral voltage at the feeder root (kV).
            voltage_noise_pu: Standard deviation of the voltage noise (per unit).
            regulator_band_pu: The noise is clipped to +/- this (per unit).
        Returns:
            DataFrame with columns t, MW, MVAR, VA, VB, VC — the same schema
            consumed by get_scenario_data_by_range().
        """
        assert self.node_data, "circuit_data is not loaded. Call get_circuit_model() first."

        # Aggregate P/Q per unique load shape, weighted by each load's nameplate rating.
        # Restricted to loads within the feeder_root subtree (the DSS file may contain
        # other, unrelated sub-feeders sharing the same substation).
        p_by_shape: Dict[Optional[str], float] = {}
        q_by_shape: Dict[Optional[str], float] = {}
        for load in self.dss_data['loads'].values():
            if load['bus'].lower() not in self.node_data:
                continue
            shape = load['yearly_shape']
            p_by_shape[shape] = p_by_shape.get(shape, 0.0) + load['kw']
            q_by_shape[shape] = q_by_shape.get(shape, 0.0) + load['kvar']

        # All load shapes in this circuit share the same interval/length; use whichever
        # named shape appears first to set the time base (shape-less loads are flat).
        named_shapes = [s for s in p_by_shape if s]
        assert named_shapes, "No loads have an assigned yearly load shape."
        reference_shape = altdss.LoadShape[named_shapes[0]]
        n_pts, interval_hr = reference_shape.NPts, reference_shape.Interval
        t = pd.date_range("2023-01-01", periods=n_pts, freq=pd.Timedelta(hours=interval_hr))

        p_total = np.zeros(n_pts)
        q_total = np.zeros(n_pts)
        for shape, p_kw in p_by_shape.items():
            mult = np.array(altdss.LoadShape[shape].PMult) if shape else np.ones(n_pts)
            p_total += p_kw * mult
            q_total += q_by_shape[shape] * mult

        rng = np.random.default_rng(seed)
        setpoint = FEEDERS[self.feeder_name].get("v_root_setpoint_pu", 1.0)
        noise = 1.0 + np.clip(rng.normal(0, voltage_noise_pu, size=n_pts), -regulator_band_pu, regulator_band_pu)
        v_head = v_root_nom * setpoint * noise
        df = pd.DataFrame({
            't': t,
            'MW': p_total * 1e-3,
            'MVAR': q_total * 1e-3,
            'VA': v_head,
            'VB': v_head,
            'VC': v_head,
        })
        t0_dt, t1_dt = pd.to_datetime(t0), pd.to_datetime(t1)
        mask = (df['t'] >= t0_dt) & (df['t'] <= t1_dt)
        return df.loc[mask].reset_index(drop=True)
    
    def align_data(self, data, dt, unit='s', max_interval=4):
        """
        Align multiple datasets by a common time base using linear interpolation.
        For downsampling, this does not take averages.
        
        :param data: dict of named dicts, each with keys 't' and one other data array
        :param resample_interval: float, time step in seconds for interpolation
        :return: (time_array, aligned_data)
            time_array: np.array of datetime64, common time base
            aligned_data: same as input data, but with data arrays resampled
        """
        aligned_data = {}
        # Determine intersection of time ranges
        starts = []
        ends = []
        for entry in data.values():
            starts.append(entry['t'][0])
            ends.append(entry['t'][len(entry['t']) - 1])
        t_start, t_end = max(starts), min(ends)
        # Create new common time vector
        time_array = np.arange(t_start, t_end, dt * np.timedelta64('1', 's')).astype(f'datetime64[{unit}]')
        new_t_float = time_array.astype(float)
        # Interpolate each series onto new time base
        for name, entry in data.items():
            aligned_data[name] = {}
            # Identify data field (exclude 't')
            for k, v in entry.items():
                if k == 't': continue
                # Linear interpolation
                aligned_data[name][k] = np.interp(new_t_float, entry['t'].to_numpy().astype(f'datetime64[{unit}]').astype(float), v)
            aligned_data[name] = pd.DataFrame(aligned_data[name])

        if max_interval:
            err = np.zeros_like(time_array, dtype=bool)
            for df in data.values():
                intervals = df['t'][1:] - df['t'][:-1]
                interval_size = intervals.median()
                period_mask = intervals > (max_interval * interval_size)
                err_intervals = period_mask.to_numpy().nonzero()[0]
                if len(err_intervals) == len(intervals):
                    print(f"utils.py:resample All data intervals are too large.\n{max_interval}, {interval_size}\n{intervals}")
                for i in err_intervals:
                    sample_mask = (time_array > df["t"][i]) & (time_array < df["t"][i + 1])
                    err |= sample_mask
            valid_mask = ~err
            for d in aligned_data.values():
                for k, v in d.items():
                    d[k] = v[valid_mask]
            time_array = time_array[valid_mask]

        return time_array, aligned_data

    def disaggregate_loads(self, p_inj, q_inj, p_profile_head, q_profile_head):
        """
        p_inj, q_inj are the average loads of the period.
        Normalize the p_inj, q_inj respectively.
        Then disaggregate the loads such that the sum of all loads equal to p_profile_head, q_profile_head.
        :param p_inj: np.array, shape=(3*N)
        :param q_inj: np.array, shape=(3*N)
        :param p_profile_head: np.array, shape=(T)
        :param q_profile_head: np.array, shape=(T)

        :return: (p_inj_nodal_timeseries, q_inj_nodal_timeseries) each shape is (T, 3*N)
        """
        # Normalize the injected powers
        p_inj_sum = np.sum(p_inj)
        q_inj_sum = np.sum(q_inj)
        if p_inj_sum == 0:
            raise ValueError("Sum of all nodal p_inj is zero")
        if q_inj_sum == 0:
            raise ValueError("Sum of all nodal q_inj is zero")
        # Normalize to fraction
        p_inj = p_inj / p_inj_sum
        q_inj = q_inj / q_inj_sum
        # Reshape
        p_profile_head = np.array(p_profile_head)[None, :]
        q_profile_head = np.array(q_profile_head)[None, :]
        # positive is defined as into the node
        p_inj_nodal_timeseries = -p_inj[:, None] * p_profile_head
        q_inj_nodal_timeseries = -q_inj[:, None] * q_profile_head

        return p_inj_nodal_timeseries.T, q_inj_nodal_timeseries.T

    def get_scenario_data_by_range(
        self,
        pv_profiles_filename,
        t0,
        t1,
        v_root_nom,
        v_root_multiplier,
        dt=60*15
    ):
        """
        Get scenario data by time range.
        :param pv_profiles_filename: Path to the PV capacity-factor timeseries file
        :param t0: str, Start time, in utc time
        :param t1: str, End time, in utc time
        :param v_root_nom: Nominal line-to-neutral voltage at the feeder root (kV)
        :param dt: Time step (in seconds) for resampling
        :return: (time_array, pv_profiles, load_profiles, V0_profiles)
            pv_profiles: capacity factor, a numpy array with shape (T,)
            load_profiles: a tuple of (active_power, reactive_power) profiles, each with shape (T, 3*N), unit in kW and kVAR
            header_voltage_profiles is a numpy array with shape (T, 3) representing the voltage at the header bus, unit in kV
            load_info is a list of dicts, each corresponding to a single-phase load e.g.
                {'bus': <bus_name>, 'phases': (1, 2)} for a phase-to-phase load
                {'bus': <bus_name>, 'phases': (1, 0)} for a phase-to-neutral load
        """
        # PV and feeder head timeseries
        pv_cap_timeseries = self._get_pv_generation(pv_profiles_filename, t0, t1)
        feeder_head_data = self.get_feeder_head_data(t0, t1, v_root_nom)

        # Align timeseries
        timeseries_data = {
            'pv': pv_cap_timeseries,
            'feeder_head': feeder_head_data
        }
        time_array, timeseries_data_aligned = self.align_data(timeseries_data, dt)
        pv_cap_factor = np.array(timeseries_data_aligned['pv']['capacity_factor'])
        
        # Compute root node voltage phasor in kV
        V0_mag = np.stack([
            timeseries_data_aligned['feeder_head'][k] for k in ['VA', 'VB', 'VC']
        ], axis=1)      # (T, 3)
        alpha = np.exp(-1j * 2/3 * np.pi)
        # Assume the phase angles are exactly 0, -120, +120
        V0 = V0_mag * np.array([1, alpha, alpha**2]).reshape(1, 3) * v_root_multiplier
        
        # Format list of single-phase loads, restricted to the feeder_root subtree
        # (the DSS file may contain other, unrelated sub-feeders on the same substation).
        assert self.node_data, "circuit_data is not loaded. Call get_circuit_model() first."
        p_inj, q_inj, load_info = [], [], []
        for load in self.dss_data['loads'].values():
            if load['bus'].lower() not in self.node_data:
                continue
            ct = self.node_data[load['bus'].lower()]['connected_transformer']
            ln = load['name']
            # Determine single-phase vs. three-phase load
            if load['num_phases'] == 1:
                p_inj.append(load['kw'])
                q_inj.append(load['kvar'])
                phases = [p for p in load['phases'] if p != 0]
                if len(phases) == 1:
                    assert load['connection'] == 'wye', load
                    load_info.append({'bus': load['bus'], 'phases': (phases[0], 0), 'connected_transformer': ct, 'name': ln})
                elif len(phases) == 2:
                    assert load['connection'] == 'delta', load
                    load_info.append({'bus': load['bus'], 'phases': tuple(phases), 'connected_transformer': ct, 'name': ln})
                else:
                    raise ValueError(f"Unsupported phase configuration: {load['phases']} for {load}")
            elif load['num_phases'] == 3:
                # Assumption: 3-phase loads are balanced (dss files do not specify)
                p_inj += [load['kw'] / 3] * 3
                q_inj += [load['kvar'] / 3] * 3
                assert (len([p for p in load['phases'] if p != 0]) == 3), load['phases']
                if load['connection'] == 'wye':
                    for p in (1, 2, 3):
                        load_info.append({'bus': load['bus'], 'phases': (p, 0), 'connected_transformer': ct, 'name': ln})
                elif load['connection'] == 'delta':
                    for phases in ((1, 2), (2, 3), (3, 1)):
                        load_info.append({'bus': load['bus'], 'phases': phases, 'connected_transformer': ct, 'name': ln})
                else:
                    raise ValueError(f"Unsupported connection type: {load['connection']} for {load}")
            else:
                raise ValueError(f"Unsupported number of phases: {load['num_phases']} for {load}")

        p_profile_head = timeseries_data_aligned['feeder_head']['MW'] * 1e3
        q_profile_head = timeseries_data_aligned['feeder_head']['MVAR'] * 1e3
        load_profiles = self.disaggregate_loads(p_inj, q_inj, p_profile_head, q_profile_head)

        return time_array, pv_cap_factor, load_profiles, load_info, V0

    def get_scenario_data_by_sampling(
            self,
            pv_profiles_filename,
            N,
            v_root_nom,
            v_root_multiplier,
            seed=42,
            dt=60*30,  # 30 minutes
            t0_str='2023-01-01 00:00:00',
            t1_str='2023-12-31 23:59:59',
        ):
        """
        Get N random single-timestep scenario data samples from [t0_str, t1_str).
        Optimized version: load all data once, then sample from memory.
        Filters out samples where V0 contains zero values.
        """
        np.random.seed(seed)

        print(f"Loading all data from {t0_str} to {t1_str}...")

        time_array, pv_profiles, load_profiles, load_info, V0_profiles = \
            self.get_scenario_data_by_range(
                pv_profiles_filename,
                t0_str,
                t1_str,
                v_root_nom,
                v_root_multiplier,
                dt=dt
            )

        total_samples = len(time_array)
        print(f"Loaded {total_samples} time points from 2023-2024")
        
        # Filter out samples where V0 contains zero or NaN
        # V0_profiles shape: (T, 3) - complex voltages for 3 phases
        valid_mask = np.all(V0_profiles != 0, axis=1) & ~np.any(np.isnan(V0_profiles), axis=1)
        valid_indices = np.where(valid_mask)[0]
        
        num_valid = len(valid_indices)
        print(f"Found {num_valid} valid samples (non-zero V0)")
        
        if num_valid < N:
            print(f"Warning: Only {num_valid} valid samples available, but {N} requested.")
            print(f"Returning all {num_valid} valid samples.")
            N = num_valid
            selected_indices = valid_indices
        else:
            # Randomly sample N from valid indices
            selected_indices = np.random.choice(valid_indices, size=N, replace=False)
        
        # Extract samples using selected indices
        sampled_timestamps = [time_array[i] for i in selected_indices]
        pv_samples = pv_profiles[selected_indices]
        load_P_samples = load_profiles[0][selected_indices]
        load_Q_samples = load_profiles[1][selected_indices]
        V0_samples = V0_profiles[selected_indices]
        
        # Final validation
        print(f"Successfully sampled {len(sampled_timestamps)} time points")
        print(f"V0_samples shape: {V0_samples.shape}")
        print(f"V0 contains zero: {np.any(V0_samples == 0)}")
        print(f"V0 contains NaN: {np.any(np.isnan(V0_samples))}")
        
        # Additional checks for other data
        if np.any(np.isnan(pv_samples)):
            print(f"WARNING: pv_samples contains {np.sum(np.isnan(pv_samples))} NaN values")
            pv_samples = np.nan_to_num(pv_samples, nan=0.0)
        
        if np.any(np.isnan(load_P_samples)) or np.any(np.isnan(load_Q_samples)):
            print(f"WARNING: load samples contain NaN, cleaning...")
            load_P_samples = np.nan_to_num(load_P_samples, nan=0.0)
            load_Q_samples = np.nan_to_num(load_Q_samples, nan=0.0)
        
        return sampled_timestamps, pv_samples, (load_P_samples, load_Q_samples), load_info, V0_samples