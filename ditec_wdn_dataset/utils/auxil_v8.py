#
# Created on Tue Jan 16 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Auxiliary functions
# ------------------------------
#

import os
from dataclasses import fields
import wntr.network as wntrn
import wntr.network.elements as wntre
from wntr.sim import SimulationResults
from typing import Callable, Generator, Literal, Tuple, Union, Any, Optional, Type, Mapping
import pandas as pd
from copy import deepcopy
import numpy as np

# import ray
import shutil
from datetime import datetime
import zarr

from enum import Enum, IntEnum
import tempfile
from numcodecs import blosc  # type:ignore
from scipy.optimize import curve_fit
import math
import networkx as nx
from collections import OrderedDict
import logging
import json
from sys import stdout

from ditec_wdn_dataset.utils.configs import (
    SimConfig,
    TuneConfig,
    JunctionTuneConfig,
    TankTuneConfig,
    ReservoirTuneConfig,
    PipeTuneConfig,
    PowerPumpConfig,
    HeadPumpConfig,
    PRVTuneConfig,
    PSVTuneConfig,
    PBVTuneConfig,
    FCVTuneConfig,
    TCVTuneConfig,
    GPVTuneConfig,
    ADGV2Config,
    AbstractConfig,
)
from ditec_wdn_dataset.utils.properties import (
    JUNCTION_PROPS,
    PIPE_PROPS,
    PUMP_PROPS,
    RESERVOIR_PROPS,
    TANK_PROPS,
    VALVE_PROPS,
    JunctionProperties,
    PipeProperties,
    PumpProperties,
    ReservoirProperties,
    TankProperties,
    ValveProperties,
)

from ditec_wdn_dataset.utils.misc import get_flow_units

blosc.use_threads = False
A_HOUR_IN_SECOND = 3600
MAX_BUFFER_SIZE = 2**31 - 1


class SimpleDataLoader:
    def __init__(self, data: np.ndarray | zarr.Array, batch_size: int = 32, shuffle: bool = True):
        """
        Initialize the data loader with the given dataset.

        Parameters:
        - data: NumPy array containing the dataset.
        - batch_size: Number of samples in each batch.
        - shuffle: If True, shuffle the dataset before each epoch.
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = data.shape[0]
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.current_batch = 0
        self.indices = np.arange(self.num_samples)

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next batch of data.

        Returns:
        - batch_data: NumPy array containing the batch of data.
        """
        if self.current_batch >= self.num_batches:
            self.current_batch = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration

        start = self.current_batch * self.batch_size
        end = min((self.current_batch + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start:end]
        batch_data = self.data[batch_indices]

        self.current_batch += 1

        return batch_data


def _get_params_by_type_backup(wntr_obj: Union[wntre.Node, wntre.Link]) -> dict:
    if isinstance(wntr_obj, wntre.Junction):
        parsed_obj = JunctionProperties()  # noqa: F405, F821
    elif isinstance(wntr_obj, wntre.Reservoir):
        parsed_obj = ReservoirProperties()  # noqa: F405
    elif isinstance(wntr_obj, wntre.Tank):
        parsed_obj = TankProperties()  # noqa: F405
    elif isinstance(wntr_obj, wntre.Pipe):
        parsed_obj = PipeProperties()  # noqa: F405
    elif isinstance(wntr_obj, wntre.Pump):
        parsed_obj = PumpProperties()  # noqa: F405
    elif isinstance(wntr_obj, wntre.Valve):
        parsed_obj = ValveProperties()  # noqa: F405
    else:
        raise TypeError(f"wntr object's type is unknown! Get type: {type(wntr_obj)}")
    node_fields = fields(parsed_obj)
    return {field.name: getattr(wntr_obj, field.name) for field in node_fields}


def _get_params_by_type(wntr_obj: Union[wntre.Node, wntre.Link]) -> dict:
    if isinstance(wntr_obj, wntre.Junction):
        node_fields = JUNCTION_PROPS  # noqa: F405
    elif isinstance(wntr_obj, wntre.Reservoir):
        node_fields = RESERVOIR_PROPS  # noqa: F405
    elif isinstance(wntr_obj, wntre.Tank):
        node_fields = TANK_PROPS  # noqa: F405
    elif isinstance(wntr_obj, wntre.Pipe):
        node_fields = PIPE_PROPS  # noqa: F405
    elif isinstance(wntr_obj, wntre.Pump):
        node_fields = PUMP_PROPS  # noqa: F405
    elif isinstance(wntr_obj, wntre.Valve):
        node_fields = VALVE_PROPS  # noqa: F405
    else:
        raise NotImplementedError(f"Found an implemented instance = {type(wntr_obj)}")

    return {field: getattr(wntr_obj, field) for field in node_fields}


def _get_params_and_wrapped_fn_on_value(wntr_obj: Union[wntre.Node, wntre.Link], wrapped_value_fn: Callable) -> dict:
    param_dict = _get_params_by_type(wntr_obj)

    return {k: wrapped_value_fn(v, wntr_obj) for k, v in param_dict.items()}


def list_all_simulation_parameters(inp_paths: list[str], verbose: bool = True, wrapped_value_fn: Callable | None = None) -> tuple[dict, dict, dict]:
    if wrapped_value_fn is None:
        wrapped_value_fn = lambda x, _: x  # noqa: E731
    total_param_dict = {}
    node_dict = {}
    link_dict = {}
    for inp_path in inp_paths:
        try:
            wn = wntrn.WaterNetworkModel(inp_file_name=inp_path)
            for _, node in wn.nodes():
                param_dict = _get_params_and_wrapped_fn_on_value(node, wrapped_value_fn)
                node_dict.update(param_dict)

            for _, link in wn.links():
                param_dict = _get_params_and_wrapped_fn_on_value(link, wrapped_value_fn)
                link_dict.update(param_dict)

            total_param_dict.update(node_dict)
            total_param_dict.update(link_dict)
            if verbose:
                print(f"total_param_dict = {total_param_dict.keys()}")
                # for k,v in total_param_dict.items():
                #    print(f'k={k},v={v}')
        except Exception as e:
            print(e)
            continue

    return total_param_dict, node_dict, link_dict


def get_sim_output_df_from_result(
    result: SimulationResults,
    sim_output: Literal["pressure", "head", "demand", "flowrate", "velocity", "headloss", "friction_factor"],
) -> pd.DataFrame:
    if sim_output in ["pressure", "head", "demand"]:
        sim_value = result.node[sim_output]  # type: ignore
    else:
        sim_value = result.link[sim_output]  # type: ignore

    return sim_value


def aggregate_simulated_output(
    result: SimulationResults,
    sim_output: Literal["pressure", "head", "demand", "flowrate", "velocity", "headloss", "friction_factor"],
    aggr: Literal["mean", "max", "min", None],
    axis: int = 0,
) -> pd.Series:
    sim_value = get_sim_output_df_from_result(result, sim_output)

    del result
    if aggr is None:
        return sim_value  # type:ignore

    if aggr == "mean":
        return sim_value.mean(axis=axis)  # type:ignore
    elif aggr == "min":
        return sim_value.min(axis=axis)  # type:ignore
    else:
        return sim_value.max(axis=axis)  # type:ignore


def remove_all_demand(obj: wntre.Junction, default_pattern_name: str):
    for ts in obj.demand_timeseries_list:
        if obj._pattern_reg.get_usage(ts.pattern_name) is not None:
            obj._pattern_reg.remove_usage(ts.pattern_name, (obj.name, "Junction"))
    obj.demand_timeseries_list.clear()
    obj.add_demand(1.0, pattern_name=default_pattern_name)


def init_water_network(
    wn: wntrn.WaterNetworkModel,
    duration: int,
    time_step: int,
    remove_controls: bool = True,
    remove_patterns: bool = True,
    remove_curve: bool = True,
) -> wntrn.WaterNetworkModel:
    """sanitize unused things and correct time settings.
    Args:
        wn (wntrn.WaterNetworkModel): water network model
        duration (int): time duration (hours)
        time_step (int): time step (hours)
        remove_controls (bool, optional): flag indicates whether we remove all control rules in INP when loaded. Defaults to True.
        remove_patterns (bool, optional): flag indicates whether we remove all patterns in INP when loaded. Defaults to True.
        remove_curve (bool, optional): flag indicates whether we remove all curves in INP when loaded.  Defaults to True.

    Returns:
        wntrn.WaterNetworkModel: sanitized water network model
    """
    time_dim = 1 if duration <= 1 else duration

    duration_in_seconds = time_dim * A_HOUR_IN_SECOND
    time_step_in_seconds = time_step * A_HOUR_IN_SECOND

    wn.options.time.duration = duration_in_seconds
    wn.options.time.hydraulic_timestep = time_step_in_seconds
    wn.options.time.pattern_timestep = time_step_in_seconds
    wn.options.time.report_timestep = time_step_in_seconds
    wn.options.time.rule_timestep = time_step_in_seconds
    wn.options.time.quality_timestep = time_step_in_seconds
    # wn.options.hydraulic.demand_model ='PDA'
    if remove_controls:
        for control_name in wn.control_name_list:
            wn.remove_control(control_name)
    if remove_patterns:
        default_pattern_name = wn._pattern_reg.default_pattern.name
        for _, obj in wn.junctions():
            remove_all_demand(obj, default_pattern_name)
        for _, obj in wn.reservoirs():
            obj.head_pattern_name = None
        for _, obj in wn.pumps():
            obj.speed_pattern_name = None
            obj.energy_pattern = None
            obj._pattern_reg.remove_usage(obj._speed_timeseries.pattern_name, (obj.name, "pump"))

        for pat_name in wn.pattern_name_list:
            if pat_name != default_pattern_name:
                wn.remove_pattern(pat_name)
    if remove_curve:
        for _, obj in wn.pumps():
            if hasattr(obj, "pump_curve_name"):
                setattr(obj, "pump_curve_name", "")
        for _, obj in wn.valves():
            if hasattr(obj, "headloss_curve_name"):
                setattr(obj, "headloss_curve_name", "")
        for curve_name in wn.curve_name_list:
            wn.remove_curve(curve_name)
    wn.reset_initial_values()
    return wn


class ParamRegistry:
    node_params = []
    link_params = []

    @staticmethod
    def register(params, is_node):
        for p in params:
            if is_node:
                if p not in ParamRegistry.node_params:
                    ParamRegistry.node_params.append(p)
            else:
                if p not in ParamRegistry.link_params:
                    ParamRegistry.link_params.append(p)


def prepare_sen_records(
    param: str,
    is_node_param: bool,
    value_list: list[float],
    wn: wntrn.WaterNetworkModel,
    alter_strategy: Literal["add", "replace"] = "replace",
    default_value: float = 0,
):
    param_idx = ParamRegistry.node_params.index(param) if is_node_param else ParamRegistry.link_params.index(param)
    num_objs = len(wn.node_name_list) if is_node_param else len(wn.link_name_list)
    # num_objs, len_value_list
    values = np.tile(np.array(value_list, dtype=float), num_objs)
    values = values.reshape([-1, len(value_list)])

    if alter_strategy != "replace":
        bl_values = []
        name_list = wn.node_name_list if is_node_param else wn.link_name_list
        for obj in name_list:
            has_attr = hasattr(obj, param)
            if has_attr:
                if param == "base_demand":
                    raw_value = obj.demand_timeseries_list[0].base_value
                else:
                    raw_value = getattr(obj, param)
            else:
                raw_value = default_value
            bl_values.append(raw_value)
        bl_values = np.array(bl_values)
        bl_values = np.reshape(bl_values, [-1, 1])
        values = values + bl_values

    # num_objs, 1, len_value_list
    node_or_link_indices = np.arange(num_objs).reshape([-1, 1]).repeat(len(value_list), axis=1)

    # num_objs, 1,  len_value_list
    param_indices = np.full_like(node_or_link_indices, fill_value=param_idx)
    is_node_params = np.full_like(node_or_link_indices, fill_value=is_node_param)

    # num_objs,  4, len_value_list
    records = np.stack(
        [param_indices, is_node_params, node_or_link_indices, values], axis=1
    )  # np.hstack([param_indices, is_node_params, node_or_link_indices, values])

    return records


def select_enum_value(cur_val: int | float | np.ndarray, my_enum: IntEnum) -> int | np.ndarray:
    unique_enum_values = set([e.value for n, e in my_enum._member_map_.items()])
    max_value = max(unique_enum_values)
    min_value = min(unique_enum_values)
    if isinstance(cur_val, (float, int)):
        int_cur_val = int(cur_val)
        return max(min(int_cur_val, max_value), min_value)
    elif isinstance(cur_val, np.ndarray):
        int_cur_val = cur_val.astype(int)
        return np.clip(int_cur_val, min_value, max_value)
    else:
        raise NotImplementedError(f"value has type = {type(cur_val)} which is no support")


def assign_value2(
    record: np.ndarray,
    inplaced_wn: wntrn.WaterNetworkModel,
    node_params: list,
    link_params: list,
    forced_value: Any | None = None,
    old_value: Any | None = None,
) -> bool:
    param_idx = int(record[0])
    is_node_param = record[1] > 0.5
    current_idx = int(record[2])
    value = record[3] if forced_value is None else forced_value

    if old_value is not None:
        if isinstance(old_value, bool):
            value = 1.0 if value >= 0.5 else 0.0
        elif isinstance(old_value, (Enum, IntEnum)):
            value = select_enum_value(cur_val=value, my_enum=type(old_value))  # type: ignore

    if is_node_param:
        obj = inplaced_wn.get_node(inplaced_wn.node_name_list[current_idx])
    else:
        obj = inplaced_wn.get_link(inplaced_wn.link_name_list[current_idx])

    param_name = node_params[param_idx] if is_node_param else link_params[param_idx]

    has_attr = hasattr(obj, param_name)
    if has_attr:
        if param_name == "base_demand":
            new_pattern_name = str(obj.name).replace(" ", "")  # type: ignore
            if new_pattern_name in inplaced_wn.pattern_name_list:
                inplaced_wn.remove_pattern(new_pattern_name)
            inplaced_wn.add_pattern(new_pattern_name, pattern=[value])
            new_pattern = inplaced_wn.get_pattern(new_pattern_name)
            obj.add_demand(1, new_pattern, None)  # type: ignore
            obj.demand_timeseries_list[0].base_value = 1  # type: ignore
            del new_pattern
        else:
            setattr(obj, param_name, value)
    del obj
    del record
    del node_params
    del link_params
    del forced_value
    del old_value
    return has_attr


def get_dtype_by_param_name(param_name: str) -> Type:
    if param_name in ["leak_status", "overflow", "cv", "bulk_coeff"]:
        return bool
    elif param_name in ["initial_status", "status"]:
        return IntEnum
    else:
        return float


def create_pattern(
    inplaced_wn: wntrn.WaterNetworkModel,
    obj: Any,
    value: np.ndarray | float | int | list,
) -> str:
    PAD_TOKEN = -100
    postfix = datetime.today().strftime("%Y%m%d_%H%M")
    pattern_name = f"P_{obj.name}" if hasattr(obj, "name") else f"P_{postfix}"  # type: ignore
    new_pattern_name = pattern_name
    i = 0
    while new_pattern_name in inplaced_wn.pattern_name_list:
        new_pattern_name = pattern_name + f"_{i}"
        i += 1
    if isinstance(value, np.ndarray):
        value = value[value != PAD_TOKEN]
        value = value.tolist()
    elif isinstance(value, (float, int)):
        value = [value if value != PAD_TOKEN else 0]
    assert isinstance(value, list), f"only support list type, but get {type(value)}"
    inplaced_wn.add_pattern(new_pattern_name, pattern=value)
    return new_pattern_name


def create_curve(
    inplaced_wn: wntrn.WaterNetworkModel,
    obj: Any,
    curve_type: Literal["HEAD", "HEADLOSS", "VOLUME", "EFFICIENCY"],
    value: np.ndarray | float | list,
    order: Literal["interleave", "concat"] = "concat",
) -> str:
    PAD_TOKEN = -100
    postfix = datetime.today().strftime("%Y%m%d_%H%M")
    curve_name = f"C_{obj.name}" if hasattr(obj, "name") else f"C_{postfix}"  # type: ignore
    new_curve_name = curve_name
    i = 0
    while new_curve_name in inplaced_wn.curve_name_list:
        new_curve_name = curve_name + f"_{i}"
        i += 1
    if isinstance(value, np.ndarray):
        # filter zero padding
        value = value[value != PAD_TOKEN]

        length = value.shape[-1] // 2
        if order == "concat":
            xs = value[:length]
            ys = value[length:]
        else:
            xs = value[::2]
            ys = value[1::2]

        assert xs.shape == ys.shape, f"create_curve: xs.shape: {xs.shape} != ys.shape: {ys.shape}"
        value = list(zip(xs, ys))
    else:
        raise NotImplementedError(f"value has type ={type(value)}")
    inplaced_wn.add_curve(new_curve_name, curve_type=curve_type, xy_tuples_list=value)
    return new_curve_name


def assign_value3(
    record: np.ndarray,
    inplaced_wn: wntrn.WaterNetworkModel,
    node_params: list,
    link_params: list,
    forced_value: Any | None = None,
    old_value: Any | None = None,
) -> bool:
    param_idx = int(record[0])
    is_node_param = record[1] > 0.5
    current_idx = int(record[2])
    value = record[3] if forced_value is None else forced_value

    if old_value is not None:
        if isinstance(old_value, bool):
            value = 1.0 if value >= 0.5 else 0.0
        elif isinstance(old_value, (Enum, IntEnum)):
            value = select_enum_value(cur_val=value, my_enum=type(old_value))  # type: ignore
            value = float(value)

    if is_node_param:
        obj: wntre.Node = inplaced_wn.get_node(inplaced_wn.node_name_list[current_idx])  # type:ignore
    else:
        obj: wntre.Link = inplaced_wn.get_link(inplaced_wn.link_name_list[current_idx])  # type:ignore

    param_name = node_params[param_idx] if is_node_param else link_params[param_idx]

    dtype = get_dtype_by_param_name(param_name)
    has_attr = hasattr(obj, param_name)
    if has_attr:
        if param_name == "base_demand":
            junc: wntre.Junction = obj
            new_pattern_name = create_pattern(
                inplaced_wn=inplaced_wn,
                obj=junc,
                value=value,
            )
            new_pattern = inplaced_wn.get_pattern(new_pattern_name)
            junc.demand_timeseries_list.clear()

            base_demand = 1
            # if flow_units is not None:
            #     from_si_ret_value = from_si(flow_units, base_demand, HydParam.Demand) #type:ignore
            #     if from_si_ret_value != base_demand:
            #         base_demand = to_si(flow_units, base_demand, HydParam.Demand) #type:ignore

            junc.add_demand(base_demand, new_pattern, None)  # type: ignore
            # junc.demand_timeseries_list[0].base_value = 1 #type: ignore
        elif param_name == "head_pattern_name":
            res: wntre.Reservoir = obj
            new_pattern_name = create_pattern(
                inplaced_wn=inplaced_wn,
                obj=res,
                value=value,
            )
            res.head_pattern_name = new_pattern_name
        elif param_name == "speed_pattern_name":
            pump: wntre.Pump = obj
            new_pattern_name = create_pattern(
                inplaced_wn=inplaced_wn,
                obj=pump,
                value=value,
            )
            pump.base_speed = 1.0
            pump.speed_pattern_name = new_pattern_name
        elif param_name == "energy_pattern":
            pump: wntre.Pump = obj
            new_pattern_name = create_pattern(
                inplaced_wn=inplaced_wn,
                obj=pump,
                value=value,
            )
            pump.energy_pattern = new_pattern_name
        elif param_name == "vol_curve_name":
            tank: wntre.Tank = obj
            new_curve_name = create_curve(
                inplaced_wn=inplaced_wn,
                obj=tank,
                curve_type="VOLUME",  # HEAD, HEADLOSS, VOLUME or EFFICIENCY
                value=value,
                order="concat",
            )
            tank.vol_curve_name = new_curve_name
        elif param_name == "pump_curve_name":
            hpump: wntre.HeadPump = obj
            new_curve_name = create_curve(
                inplaced_wn=inplaced_wn,
                obj=hpump,
                curve_type="HEAD",  # HEAD, HEADLOSS, VOLUME or EFFICIENCY
                value=value,
                order="concat",
            )
            hpump.pump_curve_name = new_curve_name
        elif param_name == "efficiency":
            pump: wntre.Pump = obj
            new_curve_name = create_curve(
                inplaced_wn=inplaced_wn,
                obj=pump,
                curve_type="EFFICIENCY",  # HEAD, HEADLOSS, VOLUME or EFFICIENCY
                value=value,
                order="concat",
            )
            pump.efficiency = new_curve_name
        elif param_name == "headloss_curve_name":
            gpv: wntre.GPValve = obj
            new_curve_name = create_curve(
                inplaced_wn=inplaced_wn,
                obj=gpv,
                curve_type="HEADLOSS",  # HEAD, HEADLOSS, VOLUME or EFFICIENCY
                value=value,
                order="concat",
            )
            gpv.headloss_curve_name = new_curve_name
        else:
            if dtype is bool:
                if isinstance(value, np.ndarray):
                    value = np.where(value >= 0.5, 1, 0).astype(bool)
                else:
                    value = True if value >= 0.5 else False
            elif dtype == IntEnum or dtype == Enum:
                if param_name == "initial_status":
                    my_enum = wntrn.LinkStatus
                else:
                    raise NotImplementedError()
                unique_enum_values = set([e.value for n, e in my_enum._member_map_.items()])  # type:ignore
                enum_length = len(unique_enum_values)
                value = value * enum_length
                value = select_enum_value(cur_val=value, my_enum=my_enum)  # type: ignore

            setattr(obj, param_name, value)

    del obj
    del record
    del node_params
    del link_params
    del forced_value
    del old_value
    return has_attr


def calculate_head_curve_coefficients(curve: wntre.Curve) -> tuple[float, float, float]:
    # ref: https://usepa.github.io/WNTR/apidoc/wntr.network.elements.HeadPump.html#wntr.network.elements.HeadPump.get_head_curve_coefficients
    Q = []
    H = []
    for pt in curve.points:
        Q.append(pt[0])
        H.append(pt[1])

    # 1-Point curve - Replicate EPANET for a one point curve
    if curve.num_points == 1:
        A = (4.0 / 3.0) * H[0]
        B = (1.0 / 3.0) * (H[0] / (Q[0] ** 2))
        C = 2
    # 2-Point curve - Replicate EPANET - generate a straight line
    elif curve.num_points == 2:
        B = -(H[1] - H[0]) / (Q[1] ** 2 - Q[0] ** 2)
        A = H[0] + B * Q[0] ** 2
        C = 1
    # 3 - Multi-point curve (3 or more points) - Replicate EPANET for
    #     3 point curves.  For multi-point curves, this is not a perfect
    #     replication of EPANET. EPANET uses a mult-linear fit
    #     between points whereas this uses a regression fit of the same
    #     H = A - B * Q **C curve used for the three point fit.
    elif curve.num_points >= 3:
        A0 = H[0]
        C0 = math.log((H[0] - H[1]) / (H[0] - H[-1])) / math.log(Q[1] / Q[-1])
        B0 = (H[0] - H[1]) / (Q[1] ** C0)

        def flow_vs_head_func(Q, a, b, c):
            return a - b * Q**c

        try:
            coeff, cov = curve_fit(flow_vs_head_func, Q, H, [A0, B0, C0])
        except RuntimeError:
            raise RuntimeError("Head pump results in a poor regression fit to H = A - B * Q^C")

        A = float(coeff[0])  # convert to native python floats
        B = float(coeff[1])
        C = float(coeff[2])
    else:
        raise RuntimeError("Head pump has an empty pump curve.")

    if A <= 0 or B < 0 or C <= 0:
        raise RuntimeError("Head pump has a negative head curve coefficient.")
    # with using scipy curve_fit, I think this is a waranted check
    elif np.isnan(A + B + C):
        raise RuntimeError("Head pump  has a coefficient which is NaN!")

    return (A, B, C)


def calculate_volume_curve(diameter: float, level: float, vol_curve: Optional[wntre.Curve] = None) -> float:
    if vol_curve is None:
        A = np.pi / 4.0 * diameter**2
        vol = A * level
    else:
        arr = np.array(vol_curve.points)
        vol = np.interp(level, arr[:, 0], arr[:, 1])
    return float(vol)


def check_valid_curve(curve: wntre.Curve | None) -> bool:
    return curve is not None and len(curve.points) > 0


def get_curve_related_parameters() -> list[str]:
    return ["vol_curve_name", "pump_curve_name", "efficiency", "headloss_curve_name"]


def list_filtered_simulation_parameters(inp_paths: list[str], verbose: bool = True, wrapped_value_fn: Callable | None = None):
    sim_output_list = ["pressure", "head", "demand", "flow", "velocity", "headloss", "friction_factor"]
    # irrelevant_params = ['valve_type','speed_pattern_name','bulk_coeff','headloss_curve','speed_timeseries','start_node_name','end_node_name','node_type','link_type','reaction_rate','vertices','demand_timeseries_list','name','tag','coordinates','initial_quality','quality','mixing_model','mixing_fraction','vol_curve', 'leak_demand' , 'leak_area', 'leak_discharge_coeff', 'leak_status','emitter_coefficient', 'level', 'setting','cv','status','head_timeseries']
    irrelevant_params = [
        "valve_type",
        "bulk_coeff",
        "headloss_curve",
        "speed_timeseries",
        "start_node_name",
        "end_node_name",
        "node_type",
        "link_type",
        "reaction_rate",
        "vertices",
        "demand_timeseries_list",
        "name",
        "tag",
        "coordinates",
        "initial_quality",
        "quality",
        "mixing_model",
        "mixing_fraction",
        "vol_curve",
        "leak_demand",
        "leak_area",
        "leak_discharge_coeff",
        "leak_status",
        "emitter_coefficient",
        "level",
        "setting",
        "cv",
        "status",
        "head_timeseries",
    ]

    param_type_dict, node_dict, link_dict = list_all_simulation_parameters(
        inp_paths, verbose=verbose, wrapped_value_fn=wrapped_value_fn
    )  # float, bool, Enum

    selected_params = [k for k, v in param_type_dict.items() if k not in sim_output_list]

    selected_params = list(set(selected_params).difference(irrelevant_params))
    node_keys = sorted(list(set(selected_params).intersection(node_dict.keys())))
    link_keys = sorted(list(set(selected_params).intersection(link_dict.keys())))
    if verbose:
        print(f"selected params: {selected_params}")
    return node_keys, link_keys


def fs2zip(zarr_paths: list, new_save_path: str = r"gigantic-dataset\outputs") -> list[str]:
    new_zip_paths = []
    for path in zarr_paths:
        if path[-4:] != "zarr":
            continue
        base_name = os.path.basename(path)[:-5]
        new_path = os.path.join(new_save_path, f"{base_name}.zip")
        old_store = zarr.open(path, mode="r")
        new_store = zarr.ZipStore(path=new_path, mode="a")
        root = zarr.group(store=new_store)
        old_attrs = old_store.attrs.asdict()
        if len(old_attrs) > 0:
            root.attrs.update(old_attrs)
        else:
            print("WARNING! old fs systems has no attrs dict!")
        print(old_store.tree())  # type: ignore
        print("#" * 80)
        for n in old_store:
            old_arr = old_store[n]
            new_arr = root.empty_like(name=n, data=old_arr)
            if old_arr.shape:
                new_arr[:] = old_arr
            else:
                new_arr = old_arr

            print("Values of {n} are saved")
        print(root.info)
        new_store.close()
        new_zip_paths.append(new_path)
    return new_zip_paths


def setup_logger(logger_name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s")

    if log_file is not None:
        fileHandler = logging.FileHandler(log_file, mode="w")
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    return logger


def zarr2csv(zarr_path: str, dir_path: Optional[str] = None, limit: int = 1000, archive: bool = True) -> list[str]:
    """
    Args:
        zarr_path (str): zarr file. support .zip and .zarr
        dir_path (Optional[str], optional): the csv dir path. If none, the created dir is at cwd. Defaults to None.
        limit (int, optional): take (limit) records. Defaults to 1000.
        archive (bool, optional): if truem, convert to zip file. Defaults to True.
    """
    assert zarr_path[-4:] == ".zip" or zarr_path[-5:] == ".zarr"

    g = zarr.open_group(zarr_path, mode="r")
    postfix = datetime.today().strftime("%Y%m%d_%H%M")

    if dir_path is None:
        dir_path = os.path.basename(zarr_path).split(".")[0] + "_csvdir" + f"_{postfix}"
        assert not os.path.exists(dir_path)
        os.mkdir(dir_path)

    # save attrs
    attrs_path = os.path.join(dir_path, "attrs.json")
    with open(attrs_path, mode="w") as f:
        json.dump(g.attrs.asdict(), f, indent=4, sort_keys=False)

    print(g.attrs.asdict())
    onames: Optional[OrderedDict] = g.attrs["onames"] if "onames" in g.attrs else None
    sim_outputs: list[str] = get_all_simulation_output_parameters()
    # save arrays
    for k in list(g.array_keys()):
        if k == "headpump_base_speed":
            raise NotImplementedError()
        print(f"Converting {k}...")
        v = g[k]
        assert isinstance(v, zarr.Array)

        if v.shape[0] > limit:  # type:ignore
            print(f"WARNING! {k} has array length exceeding limit {limit}! It will be cut off")
            v = v[: limit + 1]
        else:
            v = v[:]
        if len(v.shape) > 2:
            print(f"WARNING! {k} has array whose size >= 2! It will be reshaped to [-1, array.shape[-1]]")
            v = v.reshape([-1, v.shape[-1]])

        csv_path = os.path.join(dir_path, f"{k}.csv")
        if onames is None:
            # convert to numpy
            np.savetxt(csv_path, v, delimiter=",")
        else:
            assert isinstance(k, str)
            ks = k.split("_")
            if k in sim_outputs:
                component = "node" if k in ["pressure", "head", "demand"] else "link"
            else:
                component = ks[0]

            columns: list[str] = onames[component]

            num_components = len(columns)
            # [#scenes, #curve_points or duration, #components]
            v = v.reshape([v.shape[0], -1, num_components])  # type:ignore
            # [#scenes * #curve_points or duration, #components]
            v = v.reshape([-1, v.shape[-1]])

            df = pd.DataFrame(v, columns=columns)
            print(f"k = {k}, df.head = {df.head()}")
            df.to_csv(csv_path)
        print(f"Converting {k}...saved to {csv_path}")

    if archive:
        dir_path = shutil.make_archive(dir_path, "zip", dir_path)

    return [dir_path]


def csv2zarr(csvdir_path: str, zarr_path: Optional[str] = None, archive: bool = True) -> list[str]:
    """

    Args:
        csvdir_path (str): csv dir path. Support .zip and directory file systems
        zarr_path (Optional[str], optional):  zarr file. support .zip and .zarr. If none, the created zarr is at cwd. Defaults to None.
        archive (bool, optional): If true, perform zarr2zip . Defaults to True.

    Returns:
        list[str]: _description_
    """
    if csvdir_path[-4:] == ".zip":
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = temp_dir.name
        shutil.unpack_archive(csvdir_path, temp_path)
    else:
        temp_dir = None
        assert os.path.isdir(csvdir_path)
        temp_path = csvdir_path

    if zarr_path is None:
        postfix = datetime.today().strftime("%Y%m%d_%H%M")
        zarr_path = os.path.basename(csvdir_path).split(".")[0] + f"_{postfix}.zarr"
        assert not os.path.exists(zarr_path)

    # convert zattrs
    g = zarr.group(store=zarr_path)

    # save attrs
    attrs_path = os.path.join(temp_path, "attrs.json")
    with open(attrs_path, mode="r") as f:
        attrs = json.load(f)
        g.attrs.update(attrs)

    # save arrays
    for filename in os.listdir(temp_path):
        f = os.path.join(temp_path, filename)
        if f[-4:] != ".csv":
            continue

        print(f"Converting {filename}...")

        arr = np.genfromtxt(f, delimiter=",")
        # TODO: chunk is auto picked ! We should optimize chunk size!
        zarr_arr = g.create(name=filename[:-4], shape=arr.shape, dtype=arr.dtype, chunks=True)
        zarr_arr[:] = arr
        print(f"Converting {filename}...saved")

    output_paths = [zarr_path]
    if archive:
        output_paths = fs2zip(output_paths, new_save_path=os.path.dirname(zarr_path))

    if temp_dir is not None:
        temp_dir.cleanup()
    return output_paths


def recover_zarr(zarr_path: str, new_zarr_path: str, verbose: bool = True):
    """Remove incomplete records in input arrays! Cleanup zarr!"""
    assert zarr_path[-5:] == ".zarr" and new_zarr_path[-5:] == ".zarr", "only support .zarr"
    # copy to new zarr! Essential for backup
    with zarr.group(store=zarr_path, overwrite=False) as old_g:
        new_store = zarr.ZipStore(path=new_zarr_path, mode="a") if new_zarr_path[-4:] == ".zip" else zarr.DirectoryStore(path=new_zarr_path)
        g = zarr.group(store=new_store, overwrite=True)
        zarr.copy_all(old_g, g, log=stdout if verbose else None)

    attrs: dict = g.attrs.asdict()  # type:ignore

    assert set(["index_tracers", "okeys", "odims", "batch_size", "sim_outputs"]).issubset(attrs.keys()), (
        "Missing one of required keys! The dataset is unable to recover :("
    )
    index_tracers = attrs["index_tracers"]
    okeys = attrs["okeys"]
    odims = attrs["odims"]
    batch_size = attrs["batch_size"]
    sim_outputs = attrs["sim_outputs"]
    arr_list = list(g.array_keys())

    if verbose:
        print(f"index_tracers= {index_tracers}")
        print(f"list of arrs  = {arr_list}")

        for arr_name in arr_list:
            print(f"arr: {arr_name} | shape = {g[arr_name].shape}")

    len_index_tracers = len(index_tracers)

    for arr_name in arr_list:
        if arr_name not in sim_outputs and "temp" not in arr_name:
            del g[arr_name]

    # update list
    arr_list = list(g.array_keys())
    for arr_name in arr_list:
        if "temp" in arr_name:
            tmp_arr = g[arr_name]
            tmp_stored_name: str = arr_name  # type:ignore
            component = tmp_stored_name.split("_temp_")[-1]
            assert component in okeys and component in odims
            params = okeys[component]
            param_dims = odims[component]

            current_index = 0
            for i in range(len(params)):
                param_name = params[i]
                param_dim = param_dims[i]
                param_values = tmp_arr[:, current_index : current_index + param_dim]
                current_index += param_dim
                compo_param_name = component + "_" + param_name
                assert compo_param_name not in arr_list
                in_arr: zarr.Array = g.empty(
                    name=compo_param_name,
                    shape=[0, param_dim],
                    chunks=[batch_size, param_dim],
                    dtype=tmp_arr.dtype,
                    write_empty_chunks=False,
                )
                for success_id in index_tracers:
                    tmp_row = param_values[success_id]
                    assert not np.any(np.isnan(tmp_row))  # type:ignore
                    tmp_row = np.reshape(tmp_row, [1, -1])  # type:ignore
                    if in_arr.shape[0] < len_index_tracers:
                        in_arr.append(tmp_row)
            del g[arr_name]
    if verbose:
        print("*" * 80)
        print("After elimation...")
        for arr_name in list(g.array_keys()):
            print(f"arr: {arr_name} | shape = {g[arr_name].shape}")


def concatenate_zarr(zarr_paths: list, new_concat_path: str, verbose: bool = True) -> str:
    assert len(zarr_paths) > 1
    assert all([p[-4:] in ["zarr", ".zip"] for p in zarr_paths]), "we support only .zarr or .zip files"
    assert new_concat_path[-5:] == ".zarr", "new_concat_path has to be in .zarr"
    assert not os.path.exists(new_concat_path), "concat_path should be unique and unexisted before"
    main_store = zarr.DirectoryStore(path=new_concat_path)
    main_g = zarr.open_group(main_store, mode="a")
    if "index_tracers" not in list(main_g.attrs.keys()):
        main_g.attrs["index_tracers"] = []
    main_arr_list = []
    for i, sub_zarr_path in enumerate(zarr_paths):
        with zarr.open_group(sub_zarr_path, mode="r") as sub_g:
            sub_arr_list = list(sub_g.array_keys())
            if i == 0:
                if verbose:
                    print("cope the first path")
                zarr.copy_all(sub_g, main_g, log=stdout if verbose else None)
                main_arr_list = sub_arr_list
            else:
                for arr_name in main_arr_list:
                    if arr_name in sub_arr_list:
                        print(f"cating parameter {arr_name}..")
                        main_arr = main_g[arr_name]
                        sub_arr = sub_g[arr_name]
                        concatenated_shape = (main_arr.shape[0] + sub_arr.shape[0], *main_arr.shape[1:])  # type:ignore
                        concatenated_array = zarr.zeros(concatenated_shape, dtype=main_arr.dtype)
                        concatenated_array[: main_arr.shape[0]] = main_arr[:]
                        concatenated_array[main_arr.shape[0] :] = sub_arr[:]
                        main_g[arr_name] = concatenated_array

                if "index_tracers" in list(sub_g.attrs.keys()):
                    main_g.attrs["index_tracers"].extend(sub_g.attrs["index_tracers"])

    if verbose:
        print("*" * 40 + f"after concatenation:{new_concat_path}" + "*" * 40)
        for arr_name in main_g.array_keys():
            print(f"{arr_name}:  {main_g[arr_name].shape}")

    return new_concat_path


def rechunk_zarr(
    zarr_paths: list[str],
    new_save_path: str,
    postfix: str = "rechunked",
    rechunk_orient: Literal["scene_time_compo", "scenetime_compo"] = "scene_time_compo",
) -> list[str]:
    DYNAMIC_PARAMS = [
        "pressure",
        "head",
        "demand",
        "flowrate",
        "velocity",
        "headloss",
        "friction_factor",
        "reservoir_head_pattern_name",
        "junction_base_demand",
        "powerpump_base_speed",
        "powerpump_energy_pattern",
        #'headpump_base_speed',
        "headpump_energy_pattern",
    ]
    assert all(p[-4:] in ["zarr", ".zip"] for p in zarr_paths), "we support only .zarr or .zip files"

    new_zip_paths = []
    for path in zarr_paths:
        print(f"Processing old path ({path})...")
        base_name = os.path.basename(path)[:-5]
        extension = path[-4:].replace(".", "")
        new_path = os.path.join(new_save_path, f"{base_name}_{postfix}.{extension}")
        old_store = zarr.open(path, mode="r")
        new_store = zarr.ZipStore(path=new_path, mode="a") if extension == "zip" else zarr.DirectoryStore(path=new_path)

        root = zarr.group(store=new_store)
        old_attrs = old_store.attrs.asdict()
        if len(old_attrs) > 0:
            root.attrs.update(old_attrs)
        else:
            print("WARNING! old fs systems has no attrs dict!")

        print("OLD STRUCTURE:")
        print(old_store.tree())  # type: ignore
        print("#" * 80)
        time_dim = old_attrs["duration"] // old_attrs["time_step"]
        for n in old_store:
            print(f"Rechunking array has key ({n})...")
            old_arr: zarr.Array = old_store[n]  # type:ignore
            src_chunks = old_arr.chunks
            src_shape = old_arr.shape

            if rechunk_orient == "scene_time_compo":
                dst_chunks = (src_chunks[0], 1, src_chunks[1]) if n not in DYNAMIC_PARAMS else (src_chunks[0], time_dim, src_chunks[1] // time_dim)
                dst_shape = (0, 1, src_shape[1]) if n not in DYNAMIC_PARAMS else (0, time_dim, src_shape[1] // time_dim)
            else:
                dst_chunks = (src_chunks[0], src_chunks[1]) if n not in DYNAMIC_PARAMS else (src_chunks[0] * time_dim, src_chunks[1] // time_dim)
                dst_shape = (0, src_shape[1]) if n not in DYNAMIC_PARAMS else (0, src_shape[1] // time_dim)
            new_arr = root.empty(name=n, data=old_arr, chunks=dst_chunks, shape=dst_shape)
            for block in old_arr.blocks:
                reshaped_block = block.reshape(dst_chunks)

                new_arr.append(reshaped_block)
            # assert new_arr.shape[0] == src_shape[0]
            print(f"Values of key ({n}) are saved! new shape = ({new_arr.shape}), new chunks = ({new_arr.chunks})")
            print("#" * 80)
        print(root.info)
        new_store.close()
        new_zip_paths.append(new_path)
    return new_zip_paths


def get_adj_list(wn: wntrn.WaterNetworkModel, skip_node_names: list[str] = []) -> list[tuple[str, str, str]]:
    # ref :  https://github.com/DiTEC-project/gnn-pressure-estimation/blob/main/utils/DataLoader.py
    # graph : nx.Graph = nx.Graph(wn.to_graph()).to_undirected()
    graph: nx.Graph = wn.to_graph().to_undirected()
    if len(skip_node_names) > 0:
        graph.remove_nodes_from(skip_node_names)

    return list(graph.edges(keys=True))


def get_object_name_list_by_component(component: str, wn: wntrn.WaterNetworkModel) -> list[str]:
    component = component.lower()
    if component == "junction":
        return wn.junction_name_list
    elif component == "tank":
        return wn.tank_name_list
    elif component == "reservoir":
        return wn.reservoir_name_list
    elif component == "pipe":
        return wn.pipe_name_list
    elif component == "powerpump":
        return wn.power_pump_name_list
    elif component == "headpump":
        return wn.head_pump_name_list
    elif component == "prv":
        return wn.prv_name_list
    elif component == "psv":
        return wn.psv_name_list
    elif component == "pbv":
        return wn.pbv_name_list
    elif component == "fcv":
        return wn.fcv_name_list
    elif component == "tcv":
        return wn.tcv_name_list
    elif component == "gpv":
        return wn.gpv_name_list
    elif component == "node":
        return wn.node_name_list
    elif component == "link":
        return wn.link_name_list
    else:
        raise NotImplementedError()


def get_curve_parameters() -> list[str]:
    return ["vol_curve_name", "pump_curve_name", "efficiency", "headloss_curve_name"]


def get_pattern_parameters() -> list[str]:
    # return ['head_pattern_name','base_demand','base_speed','energy_pattern'] #base_speed is actually speed_pattern_name
    return ["head_pattern_name", "base_demand", "speed_pattern_name", "energy_pattern"]


def get_all_simulation_output_parameters() -> list[str]:
    return ["pressure", "head", "demand", "flowrate", "velocity", "headloss", "friction_factor"]


def is_node_simulation_output(output: str) -> bool:
    return output in ["pressure", "head", "demand"]


def is_node(component: str) -> bool:
    return component.lower() in ["junction", "tank", "reservoir", "node"]


def internal_upper_bound_IQR(q1: float, q3: float) -> float:
    iqr = q3 - q1
    ret = q3 + 1.5 * iqr
    return ret


def internal_lower_bound_IQR(q1: float, q3: float) -> float:
    iqr = q3 - q1
    ret = q1 - 1.5 * iqr
    return ret


def upper_bound_IQR(x: np.ndarray) -> float:
    flatten_x = x.flatten()
    q3 = np.quantile(flatten_x, 0.75).astype(float)
    q1 = np.quantile(flatten_x, 0.25).astype(float)

    return internal_upper_bound_IQR(q1, q3)


def lower_bound_IQR(x: np.ndarray) -> float:
    flatten_x = x.flatten()
    q3 = np.quantile(flatten_x, 0.75).astype(float)
    q1 = np.quantile(flatten_x, 0.25).astype(float)
    return internal_lower_bound_IQR(q1, q3)


#########################common methods moved from simgene


def get_value_at_all(
    obj: wntre.TimeSeries | wntre.Demands | wntre.Pattern,
    duration: int,
    fill_strategy: Optional[str] = "repeat",
) -> np.ndarray | list | None:
    if obj is None:
        return None

    if isinstance(obj, wntre.Demands):
        if obj._list:
            ts: wntre.TimeSeries = obj[0]
            p: wntre.Pattern | None = ts.pattern  # type:ignore
            if p is not None:
                ret_value = ts.base_value * p.multipliers  # type:ignore
            else:
                ret_value = ts.base_value  # type:ignore

        else:
            return None
    elif isinstance(obj, wntre.TimeSeries):
        p: wntre.Pattern = obj.pattern  # type:ignore
        if p is not None:
            ret_value = p.multipliers  # type:ignore
        else:
            return None
    else:
        ret_value = obj.multipliers

    values = convert_to_float_array(ret_value)
    if fill_strategy is not None:
        residual = duration - values.shape[-1]
        if residual > 0:
            values = np.pad(values, (0, residual), "wrap")  # np.repeat(values,residual+ 1, axis=-1)
        else:
            values = values[:duration]
        assert values.shape[-1] == duration

    return values


def get_value_at_time(
    obj: wntre.TimeSeries | wntre.Demands | wntre.Pattern | None,
    duration: int,
    timestep: int,
    fill_strategy: Optional[str] = "repeat",
) -> np.ndarray | None:
    if obj is None:
        return None
    else:
        if isinstance(obj, wntre.Demands):
            if timestep < 0:
                if obj._list:
                    value = obj[0].base_value
                else:
                    value = None
            else:
                value = obj.at(timestep, category=None)
        else:
            if timestep < 0:
                value = obj.base_value if hasattr(obj, "base_value") else 0.0  # type:ignore
            else:
                value = obj.at(timestep)  # type:ignore

        if value is not None:
            values = convert_to_float_array(value)
            if fill_strategy is not None:
                residual = duration - values.shape[-1]
                if residual > 0:
                    values = np.repeat(values, residual + 1, axis=-1)
                else:
                    values = values[:duration]
                assert values.shape[-1] == duration
        else:
            values = value
        return values


def get_curve_points(
    obj: wntre.Tank | wntre.Pump | wntre.GPValve,
    curve: Optional[wntre.Curve] = None,
    curve_name: Optional[str] = None,
) -> list[Tuple] | None:
    if curve_name is None and curve is None:
        return None
    else:
        if check_valid_curve(curve):
            my_curve: wntre.Curve = curve  # type:ignore
        else:
            curve_registry: wntrn.model.CurveRegistry = obj._curve_reg
            my_curve: wntre.Curve = curve_registry._data[curve_name]

        return my_curve.points


def convert_to_float_array(value: Any) -> np.ndarray:
    assert isinstance(value, (list, float, int, bool, np.ndarray, IntEnum)), f"value has type {type(value)}]"
    # TODO: check Enum and bool
    if not isinstance(value, np.ndarray):
        new_array = np.asarray(value, dtype=float)
    else:
        new_array = value
    return new_array.flatten()


def get_default_value_from_global(param_name: str, wn: wntrn.WaterNetworkModel) -> Optional[float]:
    if hasattr(wn.options.hydraulic, param_name):
        return getattr(wn.options.hydraulic, param_name)
    else:
        # raise NotImplementedError(f'The default value should not be None, consider changing generation strategy!')
        return None


def get_total_dimensions(config: SimConfig, wn: wntrn.WaterNetworkModel) -> tuple:
    def get_dim(tune_dataclass: Any, num_components: int):
        return sum([("strategy" in k and v is None or v != "keep") for k, v in vars(tune_dataclass).items()]) * num_components

    junc_dim = get_dim(config.junction_tune, len(wn.junction_name_list))
    res_dim = get_dim(config.reservoir_tune, len(wn.reservoir_name_list))
    tank_dim = get_dim(config.tank_tune, len(wn.tank_name_list))
    pipe_dim = get_dim(config.pipe_tune, len(wn.pipe_name_list))
    power_pump_dim = get_dim(config.power_pump_tune, len(wn.power_pump_name_list))
    head_pump_dim = get_dim(config.head_pump_tune, len(wn.head_pump_name_list))

    prv_dim = get_dim(config.prv_tune, len(wn.prv_name_list))
    psv_dim = get_dim(config.psv_tune, len(wn.psv_name_list))
    pbv_dim = get_dim(config.pbv_tune, len(wn.pbv_name_list))
    fcv_dim = get_dim(config.fcv_tune, len(wn.fcv_name_list))
    tcv_dim = get_dim(config.tcv_tune, len(wn.tcv_name_list))
    gpv_dim = get_dim(config.gpv_tune, len(wn.gpv_name_list))

    total_dim = junc_dim + res_dim + tank_dim + pipe_dim + power_pump_dim + head_pump_dim + prv_dim + psv_dim + pbv_dim + fcv_dim + tcv_dim + gpv_dim

    return (
        total_dim,
        junc_dim,
        res_dim,
        tank_dim,
        pipe_dim,
        power_pump_dim,
        head_pump_dim,
        prv_dim,
        psv_dim,
        pbv_dim,
        fcv_dim,
        tcv_dim,
        gpv_dim,
    )


def get_object_dict_by_config(
    tune_config: TuneConfig,
    wn: wntrn.WaterNetworkModel,
) -> Generator[tuple, Any, None]:
    if isinstance(tune_config, JunctionTuneConfig):
        return wn.junctions()
    elif isinstance(tune_config, TankTuneConfig):
        return wn.tanks()
    elif isinstance(tune_config, ReservoirTuneConfig):
        return wn.reservoirs()
    elif isinstance(tune_config, PipeTuneConfig):
        return wn.pipes()
    elif isinstance(tune_config, PowerPumpConfig):
        return wn.power_pumps()
    elif isinstance(tune_config, HeadPumpConfig):
        return wn.head_pumps()
    elif isinstance(tune_config, PRVTuneConfig):
        return wn.prvs()
    elif isinstance(tune_config, PSVTuneConfig):
        return wn.psvs()
    elif isinstance(tune_config, PBVTuneConfig):
        return wn.pbvs()
    elif isinstance(tune_config, FCVTuneConfig):
        return wn.fcvs()
    elif isinstance(tune_config, TCVTuneConfig):
        return wn.tcvs()
    elif isinstance(tune_config, GPVTuneConfig):
        return wn.gpvs()
    else:
        raise NotImplementedError()


def get_object_name_list_by_config(
    tune_config: TuneConfig,
    wn: wntrn.WaterNetworkModel,
) -> list[str]:
    if isinstance(tune_config, JunctionTuneConfig):
        return wn.junction_name_list
    elif isinstance(tune_config, TankTuneConfig):
        return wn.tank_name_list
    elif isinstance(tune_config, ReservoirTuneConfig):
        return wn.reservoir_name_list
    elif isinstance(tune_config, PipeTuneConfig):
        return wn.pipe_name_list
    elif isinstance(tune_config, PowerPumpConfig):
        return wn.power_pump_name_list
    elif isinstance(tune_config, HeadPumpConfig):
        return wn.head_pump_name_list
    elif isinstance(tune_config, PRVTuneConfig):
        return wn.prv_name_list
    elif isinstance(tune_config, PSVTuneConfig):
        return wn.psv_name_list
    elif isinstance(tune_config, PBVTuneConfig):
        return wn.pbv_name_list
    elif isinstance(tune_config, FCVTuneConfig):
        return wn.fcv_name_list
    elif isinstance(tune_config, TCVTuneConfig):
        return wn.tcv_name_list
    elif isinstance(tune_config, GPVTuneConfig):
        return wn.gpv_name_list
    else:
        raise NotImplementedError()


def save_inp_file(save_inp_path: str, wn: wntrn.WaterNetworkModel, save_name: str) -> None:
    assert save_name[-4:] == ".inp"
    network_name = os.path.basename(wn.name).split(".")[0] if wn.name is not None else "test"
    # postfix = datetime.today().strftime('%Y%m%d_%H%M')
    filename = rf"{save_inp_path}/{network_name}_{save_name}"
    try:
        wntrn.write_inpfile(wn, filename=filename, units=get_flow_units(wn))

    except OSError as e:
        print(f"WARNING! Error in saving succ inp, filename = {filename}. Error: {e}")


def list_all_simulated_parameters(config: SimConfig) -> dict[str, list[str]]:
    selected_params_dict: dict[str, list[str]] = {}
    for k, v in vars(config).items():
        if isinstance(
            v,
            (
                JunctionTuneConfig,
                TankTuneConfig,
                ReservoirTuneConfig,
                PipeTuneConfig,
                PowerPumpConfig,
                HeadPumpConfig,
                PRVTuneConfig,
                PSVTuneConfig,
                PBVTuneConfig,
                FCVTuneConfig,
                TCVTuneConfig,
                GPVTuneConfig,
            ),
        ):
            selected_params_dict[k] = []
            for field in fields(v):
                name = field.name
                tmps = name.split("_")
                param_name = "_".join(tmps[:-1])
                if "strategy" in tmps[1]:
                    selected_params_dict[k].append(param_name)
            if len(selected_params_dict[k]) > 0:
                print(f"{k} : {selected_params_dict[k]}")
    return selected_params_dict


def get_edge_index(wn: wntrn.WaterNetworkModel, skip_node_names: list[str] = []) -> np.ndarray:
    # Adapted from :
    # (1) https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/convert.html
    # (2) https://github.com/DiTEC-project/gnn-pressure-estimation/blob/main/utils/DataLoader.py
    graph: nx.Graph = nx.Graph(wn.to_graph()).to_undirected()
    if len(skip_node_names) > 0:
        graph.remove_nodes_from(skip_node_names)
    mapping = dict(zip(graph.nodes(), range(graph.number_of_nodes())))
    edge_index = np.zeros(shape=(2, graph.number_of_edges()), dtype=np.int_)
    for i, (src, dst) in enumerate(graph.edges()):
        edge_index[0, i] = mapping[src]
        edge_index[1, i] = mapping[dst]
    return edge_index


def get_onames(actual_okeys: OrderedDict, wn: wntrn.WaterNetworkModel) -> OrderedDict:
    onames = OrderedDict()
    node_names, link_names = [], []
    for component in actual_okeys:
        onames[component] = get_object_name_list_by_component(component, wn)
        if component in ["junction", "tank", "reservoir"]:
            node_names.extend(onames[component])
        else:
            link_names.extend(onames[component])

    onames["node"] = node_names
    onames["link"] = link_names
    return onames


def auto_select_chunk_depth(big_arr_in_bytes: int, big_arr_depth: int, time_dim: int, actual_ordered_dims: list[int]) -> int:
    if big_arr_in_bytes < MAX_BUFFER_SIZE:
        return big_arr_depth
    else:
        occurance_times_dict: dict[int, int] = {}
        for d in actual_ordered_dims:
            if d not in occurance_times_dict:
                occurance_times_dict[d] = 1
            else:
                occurance_times_dict[d] += 1

        occurance_times = list(occurance_times_dict.values())
        # retrieve the max occurance times' index
        max_index: int = max(enumerate(occurance_times), key=lambda x: x[1])[0]
        dim_dict = list(occurance_times_dict.keys())
        most_repeated_dim = dim_dict[max_index]
        occurance_times_dim = int(occurance_times[max_index] * most_repeated_dim)

        chunk_depth = max([occurance_times_dim, most_repeated_dim, time_dim])

        return chunk_depth


def check_demand_q3(df_dict: dict[str, pd.DataFrame], baseline_dmd_q3: float, skip_names: list[str] = []) -> bool:
    if "demand" in df_dict:
        df = df_dict["demand"]

        if skip_names is not None and len(skip_names) > 0:
            df = df.drop(columns=skip_names)
        arr = df.to_numpy()
        q3s = np.quantile(arr, 0.75, axis=1)
        return np.any(q3s >= baseline_dmd_q3)  # type:ignore
    else:
        return True


def check_MRI_index(
    df_dict: dict[str, pd.DataFrame],
    todini_index_min: float = 0,
    todini_index_max: float = 1.0,
    skip_names: list[str] = [],
) -> bool:
    if "mri" in df_dict:
        mri_arr = df_dict["mri"].to_numpy()
        print(f"min mri = {np.min(mri_arr)} | max mri = {np.max(mri_arr)}")
        return not np.any(mri_arr < todini_index_min) and not np.any(mri_arr > todini_index_max)
    else:
        return True


def root2config(root_attrs: dict | Mapping) -> SimConfig:
    """Convert Gida.root.attrs to SimConfig"""
    config = SimConfig()
    config._parsed = True

    def parse_attr_to_config(obj, k):
        if k in root_attrs:
            setattr(obj, k, root_attrs[k])

    kv_dict = config._to_dict()
    for k in kv_dict:
        parse_attr_to_config(config, k)

    return config


def pretty_print(my_dict: dict[str, Any], indent: int = 4) -> None:
    def custom_serializer(obj):
        """Handle unserializable objects like datetime, set, etc."""
        if isinstance(obj, set):
            return list(obj)  # convert sets to lists
        if isinstance(obj, AbstractConfig):
            return obj.as_dict()
        return str(obj)  # fallback: Convert unknown objects to strings

    print(json.dumps(my_dict, indent=indent, sort_keys=True, default=custom_serializer))
    # for k, v in my_dict.items():
    #     if isinstance(v, dict):
    #         pretty_print(my_dict=v, indent=indent*2)
    #     else:
    #         print(f"{k:<{indent}}:\t{str(v):}\n")


def shuffle_list(my_list: list) -> tuple[list, list]:
    """convert to numpy array, shuffle it and return itself with random ids

    Args:
        my_list (list): a simple list

    Returns:
        tuple[list, list]: tuple of shuffled list and random indices
    """

    if len(my_list) <= 0:
        return my_list, []
    arr = np.asarray(my_list)

    random_ids = np.random.permutation(len(my_list))
    shuffled_arr = arr[random_ids]
    return shuffled_arr.tolist(), random_ids.tolist()


def masking_list(my_list: list, my_mask: list[int]) -> list:
    if len(my_list) <= 0 or len(my_mask) <= 0:
        return my_list
    arr = np.asarray(my_list)
    shuffled_arr = arr[my_mask]
    return shuffled_arr.tolist()
