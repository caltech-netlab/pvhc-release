"""
Utility Functions — Algorithm 1 (Campi-Garatti), I/O helpers, and config dataclasses

Paper references:
  Algorithm 1 / Theorem III.2  — `update_num_scenarios`: evaluates the
    Campi-Garatti "wait-and-judge" φ_{k,n,β}(1-ε) criterion to determine
    whether the current number of scenarios n is sufficient for the ε-β
    reliability guarantee (Section III.C, ε=0.01, β=0.05 in paper).

  Section IV.C configuration   — `PVHCTestConfig` / `PVHCBatteryTestConfig`:
    dataclasses used by `run_pvhc_estimation.py` and `run_battery_enhancement.py`
    to specify feeder, solver, and ablation parameters (utility_pv_ratio,
    with_fairness, etc.).

  Results I/O                  — `store_optimization_results` / `load_optimization_results`:
    Save/load LP solutions (capacities, dual_var, pv_profiles, load_profiles,
    scenario_indices) to temp/optimization_results/<feeder>/<subfolder>/.
    The `dual_var` array (shape T×10×3N) is loaded by `dual_analysis.py`
    for the Appendix B sensitivity computation.

  Capacity limits              — `save_capacity_limits` / `load_capacity_limits`:
    Per-customer PV capacity limits ζ̄_c derived from annual kWh consumption
    (used as the x ≤ ζ̄_c box constraint in Eq. 12).
"""
import math
import pickle
import cvxpy as cp
import numpy as np
from decimal import Decimal, localcontext
import pathlib, sys
from datetime import datetime
from dateutil import parser
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pytz

file = pathlib.Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from src.constants import *

#==================== Useful Dataclasses =====================#

# Configuration for PVHC test runs
@dataclass
class PVHCTestConfig:
    feeder_name: str
    feeder_root: str
    v_root_nom: float
    v_root_multiplier: float
    seed: int = 42  # Default random seed
    num_scenarios: int = 1000  # Default number of scenarios
    solver: Any = cp.HIGHS  # Default solver: HiGHS is open-source, needs no license, and is
                             # fast for the plain (many-constraint) LP. It cannot handle the
                             # log-fairness objective (Eq. 16, an exponential-cone problem) --
                             # callers must pass a conic solver whenever with_fairness=True,
                             # e.g. src.solver_utils.FAIRNESS_SOLVERS. Can also be a list of
                             # solvers or (solver, options) pairs, tried in order on each
                             # subproblem with fallback on any failure (solver_utils.solve_with_fallback).
    initial_sample_size: int = 50  # Default initial sample size for iterative scenario addition
    maximum_subproblem_size: int = 10  # Default maximum subproblem size for iterative scenario addition
    clarkson_max_iterations: int = 30  # Clarkson iterations per outer iteration; an unconverged run raises
    sub_folder: str = None  # Optional subfolder for storing results
    with_fairness: bool = False  # Whether to include fairness in the optimization
    batt_size_kwh: float = 0.0  # Battery size in kWh, default 0.0 (no battery)
    batt_size_kw: float = 0.0  # Battery size in kW, default 0.0 (no battery)
    utility_pv_ratio: float = 0.0  # Ratio of PV installations that are utility-scale [0,1], 0=all customer, 1=all utility
    prioritize_low_capacity_buses: bool = False  # If True, prioritize buses with low baseline PV capacity for utility sites
    reference_subfolder: str = None  # Reference subfolder to load baseline PV capacities for prioritization
    load_cached_limits: bool = False  # If True, load previously saved network limits instead of recalculating
    use_capacity_limits: bool = False  # If True, use calculated capacity limits instead of uniform 10kW

    def resolved_sub_folder(self) -> str:
        """Return a stable subfolder name for this run.

        Existing scripts may set config.sub_folder explicitly; otherwise we derive
        a simple default that keeps runs organized and consistent across scripts.
        """
        if self.sub_folder is not None:
            return self.sub_folder
        else:
            return "with_fairness" if self.with_fairness else "without_fairness"


@dataclass
class PVHCBatteryTestConfig(PVHCTestConfig):
    """Configuration for PVHC test runs that include open-loop batteries."""

    days_sampled: int = 0
    sampling_mode: str = "whole_year"
    equal_samples_per_month: bool = False

    num_battery_candidates: int = 0
    battery_power_kw: float = 0.0
    battery_energy_kwh: float = 0.0
    battery_cost: float = 0.0
    zeta_b_binary: bool = False

    def resolved_sub_folder(self) -> str:
        if self.sub_folder is not None:
            return self.sub_folder
        else:
            base = super().resolved_sub_folder()
            return f"with_battery/{base}"

# Configuration for time-series simulation runs
@dataclass
class SimulationConfig:
    mode: str  # "whole_year", "spring", "summer", "fall", "winter", "custom"
                # "whole_year": sample from entire year
                # "spring", "summer", "fall", "winter": sample from respective seasons
                # "custom": use custom date range, see below
    days_sampled: int = 5  # Number of days sampled for seasonal modes
    # Note that these go above 365 for 2-year data
    custom_start_day: int = None  # Start day for custom mode
                                    # This is the day of the year (0-based), e.g., Jan 1 = 0, Dec 31 = 364
    custom_end_day: int = None    # End day for custom mode
                                    # This is the day of the year (0-based), e.g., Jan 1 = 0, Dec 31 = 364
    equal_samples_per_month: bool = False  # If True, sample equal number of days from each month for whole_year mode

#==================== Helper Functions for Bus Selection =====================#

def get_low_capacity_bus_indices(feeder_name, reference_subfolder="without_fairness"):
    """
    Load baseline optimization results and return bus indices sorted by PV capacity (low to high).
    
    This helper function is useful for prioritizing buses with low baseline customer PV capacity
    when selecting utility-scale PV sites.
    
    Parameters:
    - feeder_name: Name of the feeder (str)
    - reference_subfolder: Subfolder containing baseline results (default: "without_fairness")
    
    Returns:
    - sorted_indices: np.ndarray of bus indices sorted by capacity (ascending)
    - sorted_capacities: np.ndarray of corresponding capacities
    
    Example:
        indices, capacities = get_low_capacity_bus_indices("ckt5")
        # indices[0] is the bus with lowest capacity
        # indices[-1] is the bus with highest capacity
    """
    try:
        # Load baseline results
        baseline_results = load_optimization_results(feeder_name, sub_folder=reference_subfolder)
        
        # Get customer PV capacities (baseline should be all customer-based)
        capacities = baseline_results["capacities_isa_customer"]
        
        if len(capacities) == 0:
            raise ValueError(f"No customer PV data found in {reference_subfolder} for {feeder_name}")
        
        # Sort indices by capacity (ascending)
        sorted_indices = np.argsort(capacities)
        sorted_capacities = capacities[sorted_indices]
        
        return sorted_indices, sorted_capacities
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to load baseline results from {feeder_name}/{reference_subfolder}. "
            f"Make sure baseline optimization has been run. Error: {e}"
        )


#==================== Store and Load Optimization Results =====================#

def store_optimization_results(
    feeder_name, utility_pv_info, customer_pv_info, 
    capacities_isa_utility, capacities_isa_customer, config,
    node_list, edge_list,
    *,
    utility_battery_info: Optional[List[dict]] = None,
    zeta_b: Optional[np.ndarray] = None,
    p_b: Optional[np.ndarray] = None,
    q_b: Optional[np.ndarray] = None,
    soc: Optional[np.ndarray] = None,
    dual_var: Optional[np.ndarray] = None,
    pv_profiles: Optional[np.ndarray] = None,
    load_profiles: Optional[np.ndarray] = None,
    load_info: Optional[List[dict]] = None,
    results_summary: Optional[dict] = None,
    scenario_indices: Optional[List[int]] = None,
):
    """
    Store optimization results including capacities and PV info.
    
    Parameters:
    - feeder_name: Name of the feeder (str)
    - utility_pv_info: List of dictionaries with utility PV node info
    - customer_pv_info: List of dictionaries with customer PV node info
    - capacities_isa_utility: np.ndarray of utility PV hosting capacities
    - capacities_isa_customer: np.ndarray of customer PV hosting capacities
    - config: PVHCTestConfig object containing configuration details
    """
    root_path = pathlib.Path(__file__).parent.parent 
    results_dir = root_path / "temp" / "optimization_results" / feeder_name
    sub_folder = config.resolved_sub_folder() if hasattr(config, "resolved_sub_folder") else getattr(config, "sub_folder", None)
    if sub_folder is not None:
        results_dir = results_dir / sub_folder
    results_dir.mkdir(parents=True, exist_ok=True)

    # Store everything in one compressed file (battery keys are optional)
    npz_payload = {
        "utility": np.asarray(capacities_isa_utility),
        "customer": np.asarray(capacities_isa_customer),
    }
    if zeta_b is not None:
        npz_payload["zeta_b"] = np.asarray(zeta_b)
    if p_b is not None:
        npz_payload["p_b"] = np.asarray(p_b)
    if q_b is not None:
        npz_payload["q_b"] = np.asarray(q_b)
    if soc is not None:
        npz_payload["soc"] = np.asarray(soc)
    if dual_var is not None:
        npz_payload["dual_var"] = np.asarray(dual_var)
    if pv_profiles is not None:
        npz_payload["pv_profiles"] = np.asarray(pv_profiles)
    if load_profiles is not None:
        npz_payload["load_profiles"] = np.asarray(load_profiles)

    np.savez_compressed(results_dir / "optimization_results.npz", **npz_payload)
    
    # Store dictionaries with pickle (handles numpy arrays automatically)
    with (results_dir / "pv_info.pkl").open("wb") as f:
        payload = {
            "utility_pv_info": utility_pv_info,
            "customer_pv_info": customer_pv_info,
            "node_list": node_list,
            "edge_list": edge_list,
        }
        if utility_battery_info is not None:
            payload["utility_battery_info"] = utility_battery_info
        if results_summary is not None:
            payload["results_summary"] = results_summary
        if load_info is not None:
            payload["load_info"] = load_info
        if scenario_indices is not None:
            payload["scenario_indices"] = scenario_indices
        pickle.dump(payload, f)
    
    # Generate README
    readme_content = f"""# PV Hosting Capacity Optimization Results
## Feeder Information
- **Feeder Name**: {config.feeder_name}
- **Feeder Root**: {config.feeder_root}

## Voltage Parameters
- **Nominal Root Voltage**: {config.v_root_nom} kV
- **Root Voltage Multiplier**: {config.v_root_multiplier}

## Optimization Configuration
- **Solver**: {config.solver}
- **Number of Scenarios**: {config.num_scenarios}
- **Initial Sample Size**: {config.initial_sample_size}
- **Random Seed**: {config.seed}

## Battery Outputs (optional)
- If present in `optimization_results.npz`: `zeta_b`, `p_b`, `q_b`, `soc`, `dual_var`

## Output Files
- `optimization_results.npz`: Compressed numpy arrays containing hosting capacities
- `utility`: Utility-scale PV hosting capacities (iterative scenario addition)
- `customer`: Customer-side PV hosting capacities (iterative scenario addition)
- `pv_info.pkl`: Pickle file containing PV node information dictionaries
- `utility_pv_info`: Information about utility-scale PV nodes
- `customer_pv_info`: Information about customer-side PV nodes

## Loading Results
```python
import numpy as np
import pickle

# Load optimization results
data = np.load("optimization_results.npz")
capacities_utility = data["utility"]
capacities_customer = data["customer"]

# Load PV info
with open("pv_info.pkl", "rb") as f:
    pv_info = pickle.load(f)
    utility_pv_info = pv_info["utility_pv_info"]
    customer_pv_info = pv_info["customer_pv_info"]
```
"""
    
    with (results_dir / "README.md").open("w") as f:
        f.write(readme_content)


def load_optimization_results(feeder_name, sub_folder=None):
    """
    Returns: dictionary containing
       {
           "utility_pv_info": List of dictionaries with utility PV node info, e.g.
            [
                {
                    "bus": <bus_name>,
                    "phases": tuple of phases, e.g. (2, 3) or (1, 0),
                    "capacity": 
                },
                ...
            ],
           "customer_pv_info": List of dictionaries with customer PV node info,
           "capacities_isa_utility": np.ndarray of utility PV hosting capacities,
           "capacities_isa_customer": np.ndarray of customer PV hosting capacities
       }
       The length and order of utility_pv_info/customer_pv_info matches their 
         respective capacities arrays.
       For pv_info, see README.md
    """
    root_path = pathlib.Path(__file__).parent.parent 
    results_dir = root_path / "temp" / "optimization_results" / feeder_name
    if sub_folder is not None:
        results_dir = results_dir / sub_folder
    
    npz_data = np.load(results_dir / "optimization_results.npz")
    
    with (results_dir / "pv_info.pkl").open("rb") as f:
        pv_data = pickle.load(f)
    
    out = {
        **pv_data,
        "capacities_isa_utility": npz_data["utility"],
        "capacities_isa_customer": npz_data["customer"]
    }

    # Optional battery artifacts (present only for battery runs)
    for k in ("zeta_b", "p_b", "q_b", "soc", "dual_var", "pv_profiles", "load_profiles"):
        if k in npz_data:
            out[k] = npz_data[k]
    return out


def load_optimization_results_from_config(config: PVHCTestConfig):
    """Convenience wrapper that loads results using config's derived subfolder."""
    return load_optimization_results(config.feeder_name, sub_folder=config.resolved_sub_folder())


#==================== Simulation and Storage =====================#
def save_simulation_results(
    feeder_name, sub_folder, voltage_profiles, active_flows, reactive_flows, 
    p_inj, q_inj, time_array, network_data, utility_pv_info, customer_pv_info,
    capacities_isa_utility, capacities_isa_customer, injection_map, 
    config, time_periods,
):
    """
    Save simulation results to disk with sub_folder.
    
    Parameters:
    - feeder_name: Name of the feeder (str)
    - voltage_profiles: np.ndarray of voltage profiles
    - active_flows: np.ndarray of active power flows
    - reactive_flows: np.ndarray of reactive power flows
    - p_inj: np.ndarray of active power injections
    - q_inj: np.ndarray of reactive power injections
    - time_array: np.ndarray of time stamps
    - network_data: Dictionary containing network information
    - utility_pv_info: List of dictionaries with utility PV node info
    - customer_pv_info: List of dictionaries with customer PV node info
    - capacities_isa_utility: np.ndarray of utility PV hosting capacities
    - capacities_isa_customer: np.ndarray of customer PV hosting capacities
    - injection_map: dictionary containing load_info and pv_info mapping
    - config: SimulationConfig object
    - time_periods: List of tuples with (start_time, end_time) strings
    
    Returns:
    - output_dir: Path object pointing to the saved results directory
    """
    from datetime import datetime
    import json
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if sub_folder is None:
        output_dir = pathlib.Path(__file__).resolve().parents[1] / "temp" / "simulation_results" / feeder_name / timestamp
    else:
        output_dir = pathlib.Path(__file__).resolve().parents[1] / "temp" / "simulation_results" / feeder_name / sub_folder 
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration README
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(f"# Simulation Configuration\n\n")
        f.write(f"**Timestamp**: {timestamp}\n\n")
        f.write(f"**Feeder**: {feeder_name}\n\n")
        f.write(f"**Configuration**:\n")
        f.write(f"- Mode: {config.mode}\n")
        if config.days_sampled:
            f.write(f"- Days Sampled: {config.days_sampled}\n")
        if config.custom_start_day is not None:
            f.write(f"- Custom Start Day: {config.custom_start_day}\n")
        if config.custom_end_day is not None:
            f.write(f"- Custom End Day: {config.custom_end_day}\n")
        f.write(f"\n**Time Periods Simulated**:\n")
        for i, (t0, t1) in enumerate(time_periods, 1):
            f.write(f"{i}. {t0} to {t1}\n")
        f.write(f"\n**Total Timesteps**: {len(time_array)}\n")
        f.write(f"\n**PV Capacities**:\n")
        f.write(f"- Utility PV Total: {np.sum(capacities_isa_utility):.2f} kW\n")
        f.write(f"- Customer PV Total: {np.sum(capacities_isa_customer):.2f} kW\n")

    # Save Used PV Info and capacities
    np.savez_compressed(
        output_dir / "optimization_results.npz",
        utility=capacities_isa_utility,
        customer=capacities_isa_customer,
    )

    with (output_dir / "pv_info.pkl").open("wb") as f:
        pickle.dump({
            "utility_pv_info": utility_pv_info,
            "customer_pv_info": customer_pv_info,
            "injection_map": injection_map,
        }, f)
    
    # Save simulation results as compressed npz file
    np.savez_compressed(
        output_dir / "simulation_results.npz",
        voltage_profiles=voltage_profiles,
        active_flows=active_flows,
        reactive_flows=reactive_flows,
        p_inj=p_inj,
        q_inj=q_inj,
        time_array=time_array
    )
    
    # Save network data reference
    with open(output_dir / "network_info.json", "w") as f:
        json.dump({
            "n_buses": network_data["n_buses"],
            "node_list": network_data["node_list"],
            "edge_list": network_data["edge_list"],
            "feeder_name": feeder_name,
        }, f, indent=2)
    
    return output_dir

# Load simulation results 
def load_simulation_results(
    feeder_name, 
    sub_folder, 
    sim_out_basedir=pathlib.Path(__file__).resolve().parents[1] / "temp" / "simulation_results", 
    verbose=False
):
    """
    Load simulation results from the specified feeder and sub_folder.
    
    Parameters:
    - feeder_name: Name of the feeder (str)
    - sub_folder: string corresponding to the simulation run
    - verbose: If True, print the README content (bool)
    
    Returns:
    - Dictionary containing all saved data: voltage profiles, active flows, reactive flows, 
      time array, network info, optimization results (PV capacities and info)
        - "voltage_profiles": np.ndarray of voltage profiles, shape (timesteps, n_buses*3), unit in kV
        - "active_flows": np.ndarray of active power flows, shape (timesteps, n_lines*3)
        - "reactive_flows": np.ndarray of reactive power flows, shape (timesteps, n_lines*3)
        - "p_inj": np.ndarray of active power injections, shape (timesteps, n_inj), unit in kW
        - "q_inj": np.ndarray of reactive power injections, shape (timesteps, n_inj), unit in kVAR
        - "time_array": np.ndarray of datetime strings for each timestep
        - "n_buses": Number of buses in the network (int)
        - "node_list": List of node names (list of str)
        - "edge_list": List of edges (list of tuples), each tuple is (from_node, to_node)
        - "feeder_name": Name of the feeder (str)
        - "capacities_isa_utility": np.ndarray of utility PV hosting capacities
        - "capacities_isa_customer": np.ndarray of customer PV hosting capacities
        - "injection_map": dictionary containing load_info and pv_info mapping
        - "utility_pv_info": List of dictionaries, pv capacities
        - "customer_pv_info": List of dictionaries, pv capacities
    """
    results_dir = sim_out_basedir / feeder_name / sub_folder
    
    # Load simulation results
    sim_data = np.load(results_dir / "simulation_results.npz")
    
    # Load network info
    import json
    with open(results_dir / "network_info.json", "r") as f:
        network_info = json.load(f)
    
    # Load optimization results (PV capacities)
    opt_data = np.load(results_dir / "optimization_results.npz")
    
    # Load PV info
    with (results_dir / "pv_info.pkl").open("rb") as f:
        pv_info = pickle.load(f)
    
    # Print README if verbose
    if verbose:
        readme_path = results_dir / "README.md"
        if readme_path.exists():
            with open(readme_path, "r") as f:
                print(f.read())
        else:
            print(f"README.md not found in {results_dir}")
    
    return {
        # Simulation results
        "voltage_profiles": sim_data["voltage_profiles"],
        "active_flows": sim_data["active_flows"],
        "reactive_flows": sim_data["reactive_flows"],
        "p_inj": sim_data["p_inj"],
        "q_inj": sim_data["q_inj"],
        "time_array": sim_data["time_array"],
        
        # Network information
        "n_buses": network_info["n_buses"],
        "node_list": network_info["node_list"],
        "edge_list": network_info["edge_list"],
        "feeder_name": network_info["feeder_name"],

        # PV capacities
        "capacities_isa_utility": opt_data["utility"],
        "capacities_isa_customer": opt_data["customer"],
        
        # PV node information
        "utility_pv_info": pv_info["utility_pv_info"],
        "customer_pv_info": pv_info["customer_pv_info"],
        "injection_map": pv_info["injection_map"],
        
        # Metadata
        "results_dir": str(results_dir),
        "sub_folder": sub_folder
    }
    

"""
Time related
"""


def get_timezone(
    at_time=datetime.utcnow(),
    location=LOCATION,
    input_tz="utc",
    output_type="str",
):
    """
    Given location and time, get timezone of that location at that utc time.
    :param at_time: datetime, utc time. If the input is in local time, specify input_tz.
    :param location: str, e.g. "America/Los_Angeles"
    :param input_tz: str, one of 'utc', 'local'. This determines how to interpret {at_time}.
    :param output_type: str, one of {'str', 'total_seconds'}
    :return: depending on output_type,
        - 'str': str, e.g. "-07:00"
        - 'total_seconds': float, e.g. 3600.
    """
    pytz_tz = pytz.timezone(location)
    localized_time = (
        pytz.utc.localize(at_time).astimezone(pytz_tz)
        if input_tz == "utc"
        else pytz_tz.localize(at_time).astimezone(pytz_tz)
    )
    tz_offset = localized_time.utcoffset().total_seconds()
    if output_type == "total_seconds":
        return tz_offset
    elif output_type == "str":
        sign = "-" if tz_offset <= 0 else "+"
        tz_offset = abs(tz_offset)
        hours = int(tz_offset / 3600)
        minutes = int((tz_offset - hours * 3600) / 60)
        return sign + "%02d" % hours + ":" + "%02d" % minutes
    else:
        raise ValueError(f"Invalid input: output_type={output_type}")


def utc2timezone(time_column, location=LOCATION, output_type="np"):
    """
    Given a numpy datetime64 array in utc time, return the timezone as 6-character string
    at the location specified.

    :param time_column: list of datetime or np.ndarray with dtype=datetime64
    :param location: str, e.g. 'America/Los_Angeles'
    :param output_type: str, one of {'np', 'py'}, numpy string array or pythong list of str.
    :return: np.ndarray, same shape as time_column, dtype of str; or list of str.
    """
    if type(time_column) is np.ndarray:
        time_column = time_column.astype("datetime64[us]").tolist()
    timezone_column = [get_timezone(t, location=location) for t in time_column]
    return (
        np.array(timezone_column, dtype=str) if output_type == "np" else timezone_column
    )


def local2utc(time_column, location=LOCATION, output_type="np", dtype="datetime64[s]"):
    """
    Given a time column in local time (without timezone info), convert to utc.
    This requires looking up time zone offset for each of the entry in time_column
    at the specified location.
    :param time_column: list of datetime or np.ndarray with dtype=datetime64
    :param location: str, e.g. 'America/Los_Angeles'
    :param output_type: str, one of {'np', 'py'}, numpy string array or pythong list of str.
    :param dtype: str, if output_type == 'np', specifies the output array dtype
    :return: np.ndarray, or list of datetime, same shape as time_column
    """
    time_column_np = None
    if type(time_column) is np.ndarray:
        time_column_np = time_column
        time_column = time_column.astype(datetime)
    timezone_offset = [
        get_timezone(
            t, location=location, input_tz="local", output_type="total_seconds"
        )
        for t in time_column
    ]
    if output_type == "np":
        time_column_np = (
            np.array(time_column, dtype=dtype)
            if (time_column_np is None)
            else time_column_np
        )
        return time_column_np - np.array(timezone_offset).astype("timedelta64[s]")
    elif output_type == "py":
        return [
            time_column[i] - datetime.timedelta(seconds=int(timezone_offset[i]))
            for i in range(len(timezone_offset))
        ]
    else:
        raise ValueError(f"Invalid input: output_type={output_type}")


def utc2local(t, tz=None, output_type="np"):
    """
    Converts utc time to local time, dropping timezone information
    e.g. 2022-04-30T16:45:00.636 & "-07:00" --> 2022-04-30T23:45:00.636
    :param t: np.datetime64, timezone-unaware time
    :param tz: str, optional
    :returns: np.datetime64
    """
    tz = tz or utc2timezone([t.tolist()])[0]
    sign, hours, minutes = tz[0], tz[1:3], tz[4:6]
    hours, minutes = int(sign + hours), int(sign + minutes)
    timezone_timedelta = (
        np.timedelta64(1, "h") * hours + np.timedelta64(1, "m") * minutes
    )
    t_local = t + timezone_timedelta
    if output_type == "np":
        return t_local
    elif output_type == "py":
        return t_local.tolist()
    else:
        raise ValueError(f"Invalid input: output_type={output_type}")

def update_num_scenarios(epsilon, beta, num_scenarios, num_active_constraints, n_I=1):
    """
    Algorithm 1 step 3: Campi-Garatti "wait-and-judge" criterion (Theorem III.2).

    Evaluates the φ_{k,n,β}(1-ε) ≥ 0 feasibility criterion:

        φ_{k,n,β}(1-ε) = (β/(n+1)) · Σ_{m=k}^{n} C(m,k)·(1-ε)^{m-k}
                          - C(n,k)·(1-ε)^{n-k}

    where k = |support set| and n = num_scenarios.
    When φ ≥ 0 the guarantee holds: Prob[V(x*) > ε] ≤ β (paper ε=0.01, β=0.05).

    Returns -1 if the current n is sufficient (φ ≥ 0); otherwise returns the
    smallest n' > n (incremented by n_I) for which φ ≥ 0.  The outer loop in
    `run_iterative_pvhc_analysis` uses the return value to decide whether to
    stop or resample with more scenarios (see Table I for convergence history).

    Args:
        epsilon (float): Violation probability tolerance ε (paper: 0.01)
        beta (float): Confidence failure probability β (paper: 0.05)
        num_scenarios (int): Current scenario count n
        num_active_constraints (int): |support set| k from clarkson_solver
        n_I (int): Increment step when searching for sufficient n

    Returns:
        int: -1 if n is sufficient; else the required n' > n
    """
    k = num_active_constraints
    n = num_scenarios

    # phi() is the *difference* of two sums of terms C(m,k)*(1-eps)^(m-k),
    # m=k..n. At even moderately large (n,k) -- e.g. n~64000, k~200 -- these
    # individual terms, and often phi() itself, are genuinely astronomical
    # (C(63800,200) alone has 586 digits) -- there usually isn't deep
    # cancellation near the n where phi crosses zero (verified: the two
    # terms typically agree to within a handful of leading digits there, not
    # hundreds), but forming either term as a Python/numpy float64 always
    # overflows at this scale regardless (float64 tops out ~1.8e308).
    # Computing in log-space and exponentiating back doesn't help -- the
    # overflow is in the final real-space conversion of an individually-huge
    # term, not in the summation. `Decimal`, with its default exponent range
    # (|E| up to ~1e6) and precision raised well past what these terms need,
    # represents both sums exactly enough and lets the final subtraction/comparison
    # happen without ever going through float64.
    with localcontext() as ctx:
        ctx.prec = 150
        one_minus_eps = Decimal(1) - Decimal(repr(epsilon))
        beta_dec = Decimal(repr(beta))

        # Running sum of C(m,k)*(1-eps)^(m-k) for m=k..cur_m, plus `term` =
        # the m=cur_m summand itself (so it doubles as the n=cur_m "last_term"
        # below with no extra work). Both are extended incrementally as the
        # search below grows n, via term(m) = term(m-1) * m/(m-k) * (1-eps)
        # (from C(m,k)/C(m-1,k) = m/(m-k)) -- same amortized-cost intent as
        # the original implementation, just in exact Decimal arithmetic.
        running_sum = Decimal(1)  # m=k term: C(k,k)*(1-eps)^0 == 1
        term = Decimal(1)
        cur_m = k

        def extend_to(target_n: int) -> None:
            nonlocal running_sum, term, cur_m
            while cur_m < target_n:
                cur_m += 1
                term = term * cur_m * one_minus_eps / (cur_m - k)
                running_sum += term

        def phi(target_n: int) -> Decimal:
            """Phi_{k,n,beta}(1-epsilon); see docstring above for the formula."""
            extend_to(target_n)
            sum_part = beta_dec / Decimal(target_n + 1) * running_sum
            last_term = term  # == C(target_n,k)*(1-eps)^(target_n-k)
            return sum_part - last_term

        if phi(n) >= 0:
            return -1  # indicates that the number of scenarios is enough

        # Search for the smallest n' = n + i*n_I (i=1,2,...) for which phi(n') >= 0.
        while True:
            n += n_I
            if phi(n) >= 0:
                return n


# ==============================================================================
# CAPACITY LIMITS STORAGE AND LOADING
# ==============================================================================

def save_capacity_limits(
    feeder_name: str,
    load_info_with_limits: List[Dict[str, Any]],
    cache_dir: str = "temp/cache/capacity_limits"
) -> None:
    """
    Save calculated capacity limits to disk.

    Args:
        feeder_name: Name of the feeder
        load_info_with_limits: List of load dictionaries with 'capacity' field added
        cache_dir: Directory to store the cache file
    """
    import os

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Create filename from feeder name
    safe_feeder_name = feeder_name.replace('/', '_')
    filename = f"{safe_feeder_name}_capacity_limits.pkl"
    filepath = os.path.join(cache_dir, filename)

    # Save with metadata
    cache_data = {
        'load_info_with_limits': load_info_with_limits,
        'feeder_name': feeder_name,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"[Capacity Limits] Saved {len(load_info_with_limits)} load capacity limits to {filepath}")


def load_capacity_limits(
    feeder_name: str,
    cache_dir: str = "temp/cache/capacity_limits"
) -> Optional[List[Dict[str, Any]]]:
    """
    Load previously calculated capacity limits from disk.
    
    Args:
        feeder_name: Name of the feeder
        cache_dir: Directory where cache file is stored
    
    Returns:
        List of load info dictionaries with 'capacity' field, or None if not found
    """
    import os
    
    safe_feeder_name = feeder_name.replace('/', '_')
    filename = f"{safe_feeder_name}_capacity_limits.pkl"
    filepath = os.path.join(cache_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"[Capacity Limits] No cached limits found at {filepath}")
        return None
    
    try:
        with open(filepath, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"[Capacity Limits] Loaded {len(cache_data['load_info_with_limits'])} load capacity limits from {filepath}")
        return cache_data['load_info_with_limits']
    
    except Exception as e:
        print(f"[Capacity Limits] Error loading from {filepath}: {e}")
        return None
