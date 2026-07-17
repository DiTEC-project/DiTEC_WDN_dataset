#
# Created on Tue Feb 13 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Support large-scale simulation
# ------------------------------
#
import sys
import ray.actor
import zarr
import wntr
from wntr.network import elements as wntre
from enum import IntEnum, Enum
import numpy as np
import pandas as pd
import ray
import os
from time import time
from datetime import datetime  # type:ignore
from dataclasses import fields
from collections import OrderedDict
import tempfile
import shutil
import psutil
from multiprocessing import cpu_count
from typing import Callable, Generator, Literal, Type, Union

from ditec_wdn_dataset.utils.configs import (
    SimConfig,
    TuneConfig,
    Strategy,
    ADGV2Config,
)
from ditec_wdn_dataset.utils.auxil_v8 import (
    list_filtered_simulation_parameters,
    check_valid_curve,
    list_all_simulated_parameters,
    init_water_network,
    SimpleDataLoader,
    select_enum_value,
    get_adj_list,
    get_object_name_list_by_component,
    get_curve_parameters,
    get_pattern_parameters,
    get_curve_points,
    get_value_at_time,
    get_value_at_all,
    get_default_value_from_global,
    get_total_dimensions,
    get_object_dict_by_config,
    get_object_name_list_by_config,
    get_onames,
    save_inp_file,
    convert_to_float_array,
    auto_select_chunk_depth,
    is_node_simulation_output,
    pretty_print,
)
from ditec_wdn_dataset.utils.adj_builder import build_adj_from_input
from ditec_wdn_dataset.utils.profiler import WatcherManager
from ditec_wdn_dataset.utils.misc import get_flow_units
from ditec_wdn_dataset.utils.terrain import generate_elevation
from ditec_wdn_dataset.core.demand_generator import create_random_pattern
from ditec_wdn_dataset.utils.ray_utils import ray_mapping, ChunkFnProto, check_usage_if_verbose
from ditec_wdn_dataset.core.demand_generator_v2 import generate_demand
# from ditec_wdn_dataset.core.demand_generator_v2_fixed import generate_demand

from typing import Any, Protocol, Optional
from copy import deepcopy
from wntr.epanet.util import FlowUnits
from tqdm import tqdm, trange
from functools import partial
import math
import dataclasses
import traceback

# TODO: bring them to config
PAD_TOKEN = -100
STRICT_MODE: bool = False
ONE_YEAR_IN_HOURS = 8760
ONE_GB_IN_BYTES = 1024**3
FRACTION_CPU_USAGE_PER_UPSI_WORKER: float = 1
EPS = 1e-8
USED_MEMORY_THRESHOLD = 0.86

# rcr_stream = open("ray_collect_results.txt", "w+")
# russw_stream = open("ray_update_simulate_save_wn.txt", "w+")


class ChunkDemandFn(ChunkFnProto):
    def __call__(
        self,
        chunk_size: int,
        fn: Callable,
        wn: wntr.network.WaterNetworkModel,
        time_step: float,
        duration: int,
        adgv2_config: ADGV2Config,
        scale: float = 1.0,
    ) -> Any:
        # TODO: Next version, hh_junc_ids, comm_junc_ids, zero_junc_ids, extreme_junc_ids should be saved.
        chunk_scenes = []

        for _ in range(chunk_size):
            # single_scene has shape (num_object_lens, duration)
            single_scene, hh_junc_ids, comm_junc_ids, zero_junc_ids, extreme_junc_ids = fn(
                wn=wn,
                time_step=time_step,
                duration=duration,
                yearly_pattern_num_harmonics=adgv2_config.yearly_pattern_num_harmonics,
                summer_amplitude_range=adgv2_config.summer_amplitude_range,
                summer_start=adgv2_config.summer_start,
                summer_rolling_rate=adgv2_config.summer_rolling_rate,
                min_p_commercial=adgv2_config.p_commercial[0],
                max_p_commercial=adgv2_config.p_commercial[1],
                profile_household=adgv2_config.profile_household,
                profile_commercial=adgv2_config.profile_commercial,
                profile_extreme=adgv2_config.profile_extreme,
                min_noise=adgv2_config.noise_range[0],
                max_noise=adgv2_config.noise_range[1],
                zero_dem_rate=adgv2_config.zero_dem_rate,
                extreme_dem_rate=adgv2_config.extreme_dem_rate,
                max_extreme_dem_junctions=adgv2_config.max_extreme_dem_junctions,
            )
            random = np.random.rand(single_scene.shape[1])
            q1 = np.percentile(random, q=25).item()
            q3 = np.percentile(random, q=75).item()

            # single_scene has shape (1, num_object_lens, duration)
            # single_scene = np.expand_dims(single_scene, axis=0) * scale * np.random.rand()  # <-- this is a miracle.
            single_scene = np.expand_dims(single_scene, axis=0) * scale  # only scale

            hh_junctions_mask = np.zeros(single_scene.shape[1], dtype=np.bool_)
            hh_junctions_mask[hh_junc_ids] = True
            single_scene[:, hh_junctions_mask] = single_scene[:, hh_junctions_mask] * np.random.rand()

            comm_junctions_mask = np.zeros(single_scene.shape[1], dtype=np.bool_)
            comm_junctions_mask[comm_junc_ids] = True
            single_scene[:, comm_junctions_mask] = single_scene[:, comm_junctions_mask] * np.random.uniform(q1, 1)

            # we must exclude the the random value in case of nodes that have extreme demand
            if len(extreme_junc_ids) > 0:
                # extreme_junctions_mask indicates 1- extreme, 0 - normal junctions
                # extreme_junctions_mask has shape [num_object_lens]
                extreme_junctions_mask = np.zeros(single_scene.shape[1], dtype=np.bool_)
                extreme_junctions_mask[extreme_junc_ids] = True
                single_scene[:, extreme_junctions_mask] = single_scene[:, extreme_junctions_mask] * np.random.uniform(q3, 1)

            chunk_scenes.append(single_scene)

        # final chunk has shape (chunk_size, num_object_lens, duration)
        chunk = np.vstack(chunk_scenes)
        return chunk


def wrapper_create_random_pattern(
    time_step: int,
    time_dim: int,
    num_patterns: int,
    seasonal: int,
    frequency: int,
    scale: float = 1.0,
) -> list[np.ndarray]:
    ps = create_random_pattern(
        time_step=time_step,
        duration=time_dim,
        num_patterns=num_patterns,
        seasonal=1 if seasonal > 0 else 0,  # type:ignore
        freq=frequency,  # type:ignore
    )

    # return [p.reshape(1, -1) * scale for p in ps]
    return [p.reshape(1, -1) * scale * np.random.rand() for p in ps]


class CheckRecordFn(Protocol):
    # def __call__(self, single_record: np.ndarray, sim_output_key: Literal['pressure','head','demand','flowrate','velocity', 'headloss','friction_factor']) -> bool: # type:ignore
    #    pass
    def __call__(self, df_dict: dict[str, pd.DataFrame], skip_names: list[str] = []) -> bool:  # type:ignore
        pass


def get_value_internal(
    obj: Any,
    param_name: str,
    duration: int,
    timestep: int = -1,
    fill_strategy: Optional[str] = "repeat",
) -> float | list | np.ndarray | None:
    """
    This function is to handle special cases. Currently, we focus on demand patterns and curves.

    """
    has_attr = hasattr(obj, param_name)
    if has_attr:
        if param_name == "base_demand":
            junc: wntre.Junction = obj
            assert junc.demand_timeseries_list is not None
            dmd: wntre.Demands = junc.demand_timeseries_list
            # return dmd[0].base_value if timestep < 0 else dmd[0][timestep] #type: ignore
            return get_value_at_all(
                dmd, duration=duration, fill_strategy=fill_strategy
            )  # get_value_at_time(dmd, duration, timestep, fill_strategy) #

        elif param_name == "head_pattern_name":
            res: wntre.Reservoir = obj
            # return res.head_timeseries.base_value
            return get_value_at_all(
                res.head_timeseries, duration=duration, fill_strategy=fill_strategy
            )  # get_value_at_time(res.head_timeseries, duration, timestep, fill_strategy)  #
        elif param_name == "vol_curve_name":
            tank: wntre.Tank = obj
            points = get_curve_points(obj=tank, curve=tank.vol_curve, curve_name=tank.vol_curve_name)
            return points
        elif param_name == "pump_curve_name":
            hpump: wntre.HeadPump = obj
            points = get_curve_points(
                obj=hpump,
                curve=hpump.get_pump_curve(),
                curve_name=hpump.pump_curve_name,
            )
            return points
        elif param_name == "speed_pattern_name":
            # This param doesn't exist in 86 checking WDNs, but I place it here for some backup cases.
            pump: wntre.Pump = obj
            pattern_registry: wntr.network.model.PatternRegistry = pump._pattern_reg
            if pump.speed_pattern_name is not None and pump.speed_pattern_name in pattern_registry.keys():
                speed_pattern: wntre.Pattern = pattern_registry._data[pump.speed_pattern_name]
                return get_value_at_all(speed_pattern, duration=duration, fill_strategy=fill_strategy)
            else:
                return get_value_at_all(pump.speed_timeseries, duration=duration, fill_strategy=fill_strategy)
        elif param_name == "energy_pattern":
            pump: wntre.Pump = obj
            if pump.energy_pattern is not None:
                pattern_registry: wntr.network.model.PatternRegistry = pump._pattern_reg
                energy_pattern: wntre.Pattern = pattern_registry._data[pump.energy_pattern]
                return get_value_at_all(
                    energy_pattern, duration=duration, fill_strategy=fill_strategy
                )  ##get_value_at_time(energy_pattern, duration, timestep, fill_strategy) # get_value_at_all(energy_pattern)
        elif param_name == "efficiency":
            pump: wntre.Pump = obj
            points = get_curve_points(pump, curve=pump.efficiency, curve_name=None)
            return points
        elif param_name == "headloss_curve_name":
            gpv: wntre.GPValve = obj
            points = get_curve_points(gpv, curve=gpv.headloss_curve, curve_name=gpv.headloss_curve_name)
            return points
        elif "name" in param_name:
            raise NotImplementedError()
        else:
            ret = getattr(obj, param_name)
            if isinstance(ret, bool):
                ret = 1.0 if ret else 0.0
            elif isinstance(ret, (IntEnum, Enum)):
                if param_name == "initial_status":
                    my_enum = wntr.network.LinkStatus
                    unique_enum_values = set([e.value for n, e in my_enum._member_map_.items()])  # type:ignore
                    enum_length = len(unique_enum_values)
                else:
                    raise NotImplementedError()
                ret = ret.value / enum_length
            elif isinstance(ret, str):
                raise NotImplementedError()
            return ret

    return None


class ValueGenerationFn(Protocol):
    def __call__(
        self,
        obj_names: list[str],
        obj_dict: list[tuple[str, Any]],
        param_name: str,
        strategy: Strategy,
        values: Optional[Union[np.ndarray, list]],
        sim_duration: int,
        time_step: int,
        num_samples: int,
        wn: wntr.network.WaterNetworkModel,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Optional[np.ndarray]:  # type:ignore
        pass


def default_generation_func(
    obj_names: list[str],
    obj_dict: list[tuple[str, Any]],
    param_name: str,
    strategy: Strategy,
    values: Optional[Union[np.ndarray, list]],
    sim_duration: int,
    time_step: int,
    num_samples: int,
    wn: wntr.network.WaterNetworkModel,
    verbose: bool = False,
    **kwargs: Any,
) -> Optional[np.ndarray]:
    def print_warning_or_raise_exception(is_strict_mode: bool, verbose: bool = True) -> None:
        if is_strict_mode:
            raise ValueError(f"Try to access values of parameter {param_name} from an object dict, but it is unsuccesful")
        elif verbose:
            print(f"WARNING! param ({param_name}) has an empty value list, such that it is omitted in the random matrix generation")

    def print_missing_values_for_an_element(is_strict_mode: bool, verbose: bool = True) -> None:
        if is_strict_mode:
            raise ValueError(f"In accessing ({param_name}), value has been not found at some elements!")
        elif verbose:
            print(f"WARNING! param ({param_name}) value has been not found at some elements!")

    obj_names_len = len(obj_names)
    sim_duration = 1 if sim_duration <= 1 else sim_duration
    assert time_step <= sim_duration
    time_dim = sim_duration // time_step
    is_curve = param_name in get_curve_parameters()
    is_pattern = param_name in get_pattern_parameters()

    expected_shape = (num_samples, time_dim * obj_names_len) if is_pattern else (num_samples, obj_names_len)

    if strategy == "keep":
        num_points_list = []
        temp_list = []
        temp_indices = []
        for temp_id, (obj_name, obj) in enumerate(obj_dict):
            old_value = get_value_internal(obj, param_name, duration=sim_duration, timestep=-1)  # timestep= -1 will take base_value
            if old_value is None:  # get a default value from global config
                old_value = get_default_value_from_global(param_name, wn)
            if old_value is not None:
                if not isinstance(old_value, np.ndarray):
                    obj_array = convert_to_float_array(old_value)
                else:
                    obj_array = old_value

                if is_curve:
                    obj_array = np.concatenate([obj_array[::2], obj_array[1::2]], axis=-1)

                if is_curve or is_pattern:
                    num_points = obj_array.shape[-1]
                    num_points_list.append(num_points)
                temp_list.append(obj_array)
                temp_indices.append(temp_id)
        if len(temp_list) <= 0:
            print_warning_or_raise_exception(is_strict_mode=STRICT_MODE, verbose=verbose)
            new_values = None  # type:ignore
        else:
            if len(temp_list) != obj_names_len:
                print_missing_values_for_an_element(is_strict_mode=STRICT_MODE, verbose=verbose)
                # perform imputation if missing parameter in any object
                assert len(temp_indices) > 0
                # randomly choose ids from templist w.r.t. obj length
                random_ids = np.random.choice(len(temp_list), size=obj_names_len)
                random_ids[temp_indices] = list(range(len(temp_list)))
                random_selection_list = [temp_list[random_id] for random_id in random_ids]
                temp_list = random_selection_list

            if is_curve or is_pattern:
                max_length = max(num_points_list)
                for i in range(len(temp_list)):
                    c_or_p = temp_list[i]
                    # pad if length < max_length
                    if c_or_p.shape[-1] < max_length:
                        residual = max_length - c_or_p.shape[-1]
                        c_or_p = np.pad(
                            c_or_p,
                            pad_width=(0, residual),
                            mode="constant",
                            constant_values=PAD_TOKEN,
                        )
                        temp_list[i] = c_or_p

                expected_shape = (num_samples, obj_names_len * max_length)

            new_values = np.concatenate(temp_list, axis=0)
            new_values = np.repeat(np.expand_dims(new_values, 0), num_samples, axis=0)
            new_values = np.reshape(new_values, [num_samples, -1])

    elif strategy == "sampling":
        if not is_curve:
            # if not is_curve, we need a couple of values
            assert values is not None and len(values) == 2

            new_values = values[0] + (values[1] - values[0]) * np.random.random(size=expected_shape)

        else:
            # values list should have [xmin, xmax, ymin, ymax, num_curves]
            assert values is not None and len(values) == 5
            num_curve_points = int(values[4])
            single_expected_shape = (num_samples, obj_names_len * num_curve_points)
            new_values_x = values[0] + (values[1] - values[0]) * np.random.random(size=single_expected_shape)
            new_values_y = values[2] + (values[3] - values[2]) * np.random.random(size=single_expected_shape)
            # new_values has shape (num_samples, obj_names_len * 2) and values (x1,x2,...,y1,y2,...)
            new_values = np.concatenate([new_values_x, new_values_y], axis=-1)
            expected_shape = (num_samples, obj_names_len * num_curve_points * 2)

    elif strategy == "perturbation":
        num_points_list = []
        temp_list = []
        for obj_name, obj in obj_dict:
            old_value = get_value_internal(obj, param_name, duration=sim_duration, timestep=-1)  # timestep= -1 will take base_value
            if old_value is None:  # get a default value from global config
                old_value = get_default_value_from_global(param_name, wn)
            if old_value is not None:
                if not isinstance(old_value, np.ndarray):
                    obj_array = convert_to_float_array(old_value)
                else:
                    obj_array = old_value

                if is_curve:
                    obj_array = np.concatenate([obj_array[::2], obj_array[1::2]], axis=-1)

                if is_curve or is_pattern:
                    num_points = obj_array.shape[-1]
                    num_points_list.append(num_points)
                temp_list.append(obj_array)

        if len(temp_list) <= 0:
            print_warning_or_raise_exception(is_strict_mode=STRICT_MODE, verbose=verbose)
            new_values = None  # type:ignore
        else:
            if len(temp_list) != obj_names_len:
                print_missing_values_for_an_element(is_strict_mode=STRICT_MODE, verbose=verbose)

            if is_curve or is_pattern:
                max_length = max(num_points_list)
                for i in range(len(temp_list)):
                    c_or_p = temp_list[i]
                    # pad if length < max_length
                    if c_or_p.shape[-1] < max_length:
                        residual = max_length - c_or_p.shape[-1]
                        c_or_p = np.pad(
                            c_or_p,
                            pad_width=(0, residual),
                            mode="constant",
                            constant_values=PAD_TOKEN,
                        )
                        temp_list[i] = c_or_p

                expected_shape = (num_samples, obj_names_len * max_length)

            if not is_curve:
                assert values is not None and len(values) == 1

                old_values = np.concatenate(temp_list, axis=0)
                old_values = old_values.reshape([1, old_values.shape[-1]])  # np.expand_dims(old_values,0)
                # if "diameter" in param_name:
                #     new_values = (
                #         old_values
                #         * np.full(shape=expected_shape, fill_value=values[0])  # np.random.random(size=expected_shape) * values[0]
                #     )
                # else:
                new_values = old_values + np.random.uniform(low=-1, high=1, size=expected_shape) * values[0]

            else:
                assert values is not None and len(values) == 2

                old_values_x = []
                old_values_y = []
                old_lengths = [len(c) // 2 for c in temp_list]
                max_curve_length = max(old_lengths)

                # pad [PAD_TOKEN] in case that the current curve length is less than maximum_curve_length
                for i, c in enumerate(temp_list):
                    old_xs = c[: old_lengths[i]]
                    old_ys = c[old_lengths[i] :]
                    # if old_lengths[i] < max_curve_length:
                    #     residual = max_curve_length - old_lengths[i]
                    #     old_xs = np.pad(old_xs, pad_width=(0,residual),mode='constant',constant_values=PAD_TOKEN)
                    #     old_ys = np.pad(old_ys, pad_width=(0,residual),mode='constant',constant_values=PAD_TOKEN)
                    old_values_x.append(old_xs)
                    old_values_y.append(old_ys)

                old_values_x = np.concatenate(old_values_x, axis=0).reshape([1, obj_names_len, max_curve_length])
                old_values_y = np.concatenate(old_values_y, axis=0).reshape([1, obj_names_len, max_curve_length])

                single_expected_shape = (num_samples, obj_names_len, max_curve_length)
                new_values_x = (
                    old_values_x
                    + np.random.choice([-1.0, 1.0], size=single_expected_shape) * np.random.random(size=single_expected_shape) * values[0]
                )
                new_values_y = (
                    old_values_y
                    + np.random.choice([-1.0, 1.0], size=single_expected_shape) * np.random.random(size=single_expected_shape) * values[1]
                )
                # new_values has shape (num_samples, obj_names_len * max_curve_length * 2) and values (x1,x2,...,y1,y2,...)
                new_values = np.concatenate([new_values_x, new_values_y], axis=-1)
                new_values = new_values.reshape([num_samples, -1])
                expected_shape = (num_samples, obj_names_len * max_curve_length * 2)
    elif strategy == "substitute":
        if len(obj_dict) > 0:
            # random_obj: wntre.Node | wntre.Link = obj_dict[0][1]
            # assert (
            #     isinstance(random_obj, wntre.Link) and param_name == "diameter"
            # ), f"Currently support only Pipe-diameter, but get {random_obj.__class__.__name__}(compo)-{param_name}(param) "
            if verbose:
                if is_pattern or is_curve:
                    print(f"WARNING! ({strategy}) strategy applied to (pattern) or (curve)! They have not been tested yet! ")
            random_indices = np.random.randint(low=0, high=obj_names_len, size=[num_samples, 1])

            temp_list = []
            num_points_list = []
            for i in range(num_samples):
                random_index = random_indices[i][0]
                (_, obj) = obj_dict[random_index]
                old_value = get_value_internal(obj, param_name, duration=sim_duration, timestep=-1)  # timestep= -1 will take base_value
                if old_value is None:  # get a default value from global config
                    old_value = get_default_value_from_global(param_name, wn)
                if old_value is not None:
                    if not isinstance(old_value, np.ndarray):
                        obj_array = convert_to_float_array(old_value)
                    else:
                        obj_array = old_value

                    if is_curve:
                        obj_array = np.concatenate([obj_array[::2], obj_array[1::2]], axis=-1)

                    if is_curve or is_pattern:
                        num_points = obj_array.shape[-1]
                        num_points_list.append(num_points)
                    temp_list.append(obj_array)

            if is_curve or is_pattern:
                max_length = max(num_points_list)
                for i in range(len(temp_list)):
                    c_or_p = temp_list[i]
                    # pad if length < max_length
                    if c_or_p.shape[-1] < max_length:
                        residual = max_length - c_or_p.shape[-1]
                        c_or_p = np.pad(
                            c_or_p,
                            pad_width=(0, residual),
                            mode="constant",
                            constant_values=PAD_TOKEN,
                        )
                        temp_list[i] = c_or_p

                expected_shape = (num_samples, obj_names_len * max_length)

            assert values is not None and len(values) == 1
            vstack_values = np.vstack(temp_list)
            new_values = np.tile(vstack_values, (1, obj_names_len)) + np.random.uniform(low=-1, high=1, size=[num_samples, 1]) * values[0]
        else:
            print_warning_or_raise_exception(is_strict_mode=STRICT_MODE, verbose=verbose)
            new_values = None  # type:ignore
    elif strategy == "factor":
        if len(obj_dict) > 0:
            if verbose:
                random_obj: wntre.Node | wntre.Link = obj_dict[0][1]
                if isinstance(random_obj, wntre.Node) or is_pattern or is_curve:
                    print(
                        f"WARN! ({strategy}) strategy applied to (node) or (pattern) or (curve)! They have not been tested yet! Only (link) (static) parameters are well-tested"
                    )

            num_points_list = []
            temp_list = []
            for obj_name, obj in obj_dict:
                old_value = get_value_internal(obj, param_name, duration=sim_duration, timestep=-1)  # timestep= -1 will take base_value
                if old_value is None:  # get a default value from global config
                    old_value = get_default_value_from_global(param_name, wn)
                if old_value is not None:
                    if not isinstance(old_value, np.ndarray):
                        obj_array = convert_to_float_array(old_value)
                    else:
                        obj_array = old_value

                    if is_curve:
                        obj_array = np.concatenate([obj_array[::2], obj_array[1::2]], axis=-1)

                    if is_curve or is_pattern:
                        num_points = obj_array.shape[-1]
                        num_points_list.append(num_points)
                    temp_list.append(obj_array)

            if is_curve or is_pattern:
                max_length = max(num_points_list)
                for i in range(len(temp_list)):
                    c_or_p = temp_list[i]
                    # pad if length < max_length
                    if c_or_p.shape[-1] < max_length:
                        residual = max_length - c_or_p.shape[-1]
                        c_or_p = np.pad(
                            c_or_p,
                            pad_width=(0, residual),
                            mode="constant",
                            constant_values=PAD_TOKEN,
                        )
                        temp_list[i] = c_or_p
                expected_shape = (num_samples, obj_names_len * max_length)

            assert values is not None and len(values) == 3

            old_values = np.concatenate(temp_list, axis=0)
            old_values = old_values.reshape([1, old_values.shape[-1]])  # np.expand_dims(old_values,0)

            new_values = (
                old_values
                * np.full(
                    shape=expected_shape, fill_value=values[0] + np.random.rand() * (values[1] - values[0])
                )  # random scale in range [values[0], values[1]]
                + np.random.uniform(low=-1, high=1, size=[num_samples, 1])
                * values[2]  # same noise for all components per scenario to prevent pipe bottleneck
            )
        else:
            print_warning_or_raise_exception(is_strict_mode=STRICT_MODE, verbose=verbose)
            new_values = None  # type:ignore
    elif strategy == "terrain":
        if len(obj_dict) > 0:
            random_obj: wntre.Node | wntre.Link = obj_dict[0][1]
            assert param_name == "elevation" and isinstance(random_obj, wntre.Junction), (
                f"Error! We only support terrain strategy for junction_elevation! Get {random_obj.__class__.__name__} (compo)- {param_name} (param)"
            )
            assert values is not None and len(values) == 2

            num_cpus = kwargs.get("num_cpus", 1)
            new_values = ray_mapping(
                fn=generate_elevation,
                num_cpus=num_cpus,
                num_samples=num_samples,
                node_names=wn.junction_name_list,
                wn=deepcopy(wn),
                randomness=values[0],
                height_map_width=values[1],
                zrange=None,  # auto-pick w.r.t. baseline junctions elevations
                verbose=verbose,
                plot_fig=False,
            )
            # new_values has shape (num_samples,  obj_names_len* 1)
            new_values = new_values.reshape([num_samples, -1])

        else:
            print_warning_or_raise_exception(is_strict_mode=STRICT_MODE, verbose=verbose)
            new_values = None  # type:ignore
    elif strategy == "series":
        """
        TODO: series need to be checked

        """
        assert values is not None
        if isinstance(values, list):
            new_values = np.asarray(values, float)
        else:
            new_values: np.ndarray = values  # type:ignore

        if is_curve:
            num_curve_points = new_values.shape[-1]
            new_values = np.repeat(new_values.reshape([1, -1]), num_samples * obj_names_len, axis=0).reshape([num_samples, -1])
            expected_shape = (num_samples, obj_names_len * num_curve_points)
        else:
            if new_values.shape[0] > time_dim:
                if time_step <= 1:
                    new_values = new_values[:time_dim]
                else:
                    new_values = new_values[:sim_duration:time_step]

            assert new_values.shape[0] == time_dim
            new_values = np.reshape(new_values, [1, -1, 1]).repeat(num_samples * obj_names_len, axis=0).reshape([num_samples, -1])
            expected_shape = (num_samples, time_dim * obj_names_len)

    elif strategy == "adg":  # Automatic (Andres') Demand Generator
        assert param_name == "base_demand", "adg strategy only supports demand"
        assert values is not None and len(values) >= 2
        seasonal = values[0]
        frequency = values[1]
        scale = values[2] if len(values) > 2 else 1.0
        # time_step =  wn.options.time.hydraulic_timestep # in seconds, by default = 3600
        # pattern_list is time_dim x (1,time_dim)
        pattern_list = wrapper_create_random_pattern(
            time_step=1 * 60,  # trick to get arrays whose shape (1, time_dim)
            time_dim=time_dim,
            num_patterns=num_samples * obj_names_len,
            seasonal=seasonal,
            frequency=frequency,
            scale=scale,
        )
        assert pattern_list[0].shape == (1, time_dim)
        # new_values has shape (num_samples, obj_names_len, time_dim)
        new_values = np.concatenate(pattern_list).reshape(num_samples, obj_names_len, -1)
        # new_values has shape (num_samples,  obj_names_len* time_dim)
        new_values = new_values.reshape([num_samples, -1])
        expected_shape = (num_samples, obj_names_len * time_dim)

    elif strategy == "adg_v2":
        # Automatic (Andres') Demand Generator V2
        assert param_name == "base_demand", "adg_v2 strategy only supports demand"
        assert values is not None and len(values) == 1
        scale = values[0]

        num_cpus = kwargs.get("num_cpus", 1)
        adgv2_config = kwargs.get("adgv2_config", None)
        if adgv2_config is None:
            if verbose:
                print("WARNING! Unable to load ADGV2 config, we use the default settings")
            adgv2_config = ADGV2Config()

        assert isinstance(adgv2_config, ADGV2Config)

        # pattern_list is time_dim x (1,time_dim)

        new_values = ray_mapping(
            fn=generate_demand,
            num_cpus=num_cpus,
            num_samples=num_samples,
            chunk_fn=ChunkDemandFn(),
            # -- list of parameters required from chunk_fn--
            wn=deepcopy(wn),
            time_step=time_step,
            duration=sim_duration,
            adgv2_config=adgv2_config,
            scale=scale,
        )

        # new_values has shape (num_samples,  obj_names_len* time_dim)
        new_values = new_values.reshape([num_samples, -1])
        expected_shape = (num_samples, obj_names_len * time_dim)
    else:
        raise NotImplementedError()

    if new_values is not None:
        assert new_values.shape == expected_shape, f"Expected shape = {expected_shape}, but get new_values.shape={new_values.shape}"
    return new_values


def prepare_simgen_subset_records(
    tune_config: TuneConfig,
    sim_duration: int,
    time_step: int,
    num_samples: int,
    wn: wntr.network.WaterNetworkModel,
    valueGenerationFn: ValueGenerationFn,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, list[str], list[int]]:
    param_nested_dict = OrderedDict()
    for field in fields(tune_config):
        name = field.name
        tmps = name.split("_")
        param_name = "_".join(tmps[:-1])
        strategy_or_values = tmps[-1]
        if param_name not in param_nested_dict:
            param_nested_dict[param_name] = OrderedDict()
        param_nested_dict[param_name][strategy_or_values] = getattr(tune_config, name)

    obj_names = get_object_name_list_by_config(tune_config, wn)
    obj_dict = list(get_object_dict_by_config(tune_config, wn))

    stack_list = []
    actual_params = []
    actual_param_dims = []
    # actual_dtypes = []
    for param_name in param_nested_dict:
        param_dict = param_nested_dict[param_name]
        strategy = param_dict["strategy"]
        values = param_dict["values"]

        new_values = valueGenerationFn(
            obj_names=obj_names,
            obj_dict=obj_dict,
            param_name=param_name,
            strategy=strategy,
            values=values,
            sim_duration=sim_duration,
            time_step=time_step,
            num_samples=num_samples,
            wn=wn,
            verbose=verbose,
            **kwargs,
        )
        if new_values is not None:
            actual_param_dims.append(new_values.shape[-1])
            actual_params.append(param_name)
            stack_list.append(new_values)

    assert len(stack_list) > 0
    # new_rows has shape (num_samples, sum(actual_param_dims))
    new_rows = np.concatenate(stack_list, axis=-1)

    return new_rows, actual_params, actual_param_dims


def wrapper_prepare_simgen_subset_records(
    tune_config: TuneConfig,
    sim_duration: int,
    time_step: int,
    num_samples: int,
    zarr_group: zarr.Group,
    array_name: str,
    batch_size: int,
    wn: wntr.network.WaterNetworkModel,
    valueGenerationFn: ValueGenerationFn,
    verbose: bool = True,
    **kwargs: Any,
) -> tuple[zarr.Array, int, list[str], list[int]]:
    # time_dim = 1 if sim_duration <= 0 else sim_duration

    num_batches = math.ceil(num_samples / batch_size)
    current_index = 0

    for b in trange(num_batches):
        if current_index + batch_size > num_samples:
            remaining_size = num_samples - current_index
        else:
            remaining_size = batch_size
        rows, actual_ordered_keys, actual_ordered_dims = prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=sim_duration,
            time_step=time_step,
            num_samples=remaining_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
            **kwargs,
        )

        actual_samples = rows.shape[0]
        actual_dim = rows.shape[-1]

        rows_in_bytes = rows.nbytes
        chunk_depth = auto_select_chunk_depth(
            big_arr_depth=actual_dim,
            big_arr_in_bytes=rows_in_bytes,
            actual_ordered_dims=actual_ordered_dims,
            time_dim=sim_duration // time_step,
        )

        if array_name not in list(zarr_group.array_keys()):
            stored_array: zarr.Array = zarr_group.empty(
                array_name,
                shape=[num_samples, actual_dim],
                chunks=[batch_size, chunk_depth],
                dtype=float,
                write_empty_chunks=False,
            )

        else:
            stored_array: zarr.Array = zarr_group[array_name]  # type:ignore

        # stored_array[current_index : current_index + actual_samples] = rows
        if actual_dim == chunk_depth:
            stored_array[current_index : current_index + actual_samples] = rows
        else:
            start = 0
            num_depth_chunks = actual_dim // chunk_depth
            for i in range(num_depth_chunks):
                if i < num_depth_chunks - 1:
                    stored_array[current_index : current_index + actual_samples, start : start + chunk_depth] = rows[:, start : start + chunk_depth]
                    start += chunk_depth
                else:
                    stored_array[current_index : current_index + actual_samples, start:] = rows[:, start:]
                    start += rows.shape[-1] - start
            assert start == actual_dim
        current_index += actual_samples
        del rows

    if verbose:
        print(stored_array.info)  # type:ignore
    return stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims  # type:ignore


def prepare_simgen_records(
    config: SimConfig,
    prefix: str,
    zarr_group: zarr.Group,
    ran_zarr_group: zarr.Group,
    wn: wntr.network.WaterNetworkModel,
    batch_size: int,
    sim_batch_size: int,
    valueGenerationFn: ValueGenerationFn,
    verbose: bool = True,
) -> tuple[zarr.Array, list, OrderedDict, OrderedDict, list]:
    (
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
    ) = get_total_dimensions(config, wn)

    num_samples = int(config.num_samples * config.backup_times)
    # time_dim = 1 if config.duration <= 1 else config.duration

    stored_path_list = []
    stored_array_list = []
    actual_dims = []
    actual_odims = OrderedDict()
    actual_okeys = OrderedDict()

    if junc_dim > 0:
        component = "junction"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.junction_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
            adgv2_config=config.adgv2_config,  # kwargs
            num_cpus=config.num_cpus,  # kwargs
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if res_dim > 0:
        component = "reservoir"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.reservoir_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if tank_dim > 0:
        component = "tank"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.tank_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if pipe_dim > 0:
        component = "pipe"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.pipe_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if power_pump_dim > 0:
        component = "powerpump"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.power_pump_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if head_pump_dim > 0:
        component = "headpump"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.head_pump_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if prv_dim > 0:
        component = "prv"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.prv_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if psv_dim > 0:
        component = "psv"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.psv_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if pbv_dim > 0:
        component = "pbv"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.pbv_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if fcv_dim > 0:
        component = "fcv"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.fcv_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if tcv_dim > 0:
        component = "tcv"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.tcv_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    if gpv_dim > 0:
        component = "gpv"
        WatcherManager.track(f"Gen_{component}")
        tune_config = config.gpv_tune
        stored_array, actual_dim, actual_ordered_keys, actual_ordered_dims = wrapper_prepare_simgen_subset_records(
            tune_config=tune_config,
            sim_duration=config.duration,
            time_step=config.time_step,
            num_samples=num_samples,
            zarr_group=zarr_group,
            array_name=prefix + f"_{component}",
            batch_size=batch_size,
            wn=wn,
            valueGenerationFn=valueGenerationFn,
            verbose=verbose,
        )

        actual_dims.append(actual_dim)
        actual_okeys[component] = actual_ordered_keys
        actual_odims[component] = actual_ordered_dims
        stored_array_list.append(stored_array)
        stored_path_list.append(prefix + f"_{component}")
        WatcherManager.stop(f"Gen_{component}")

    # concatenate all stored arrays into ran_zarr_group #temp folder
    random_array_name = prefix + "_random"
    concated_shape = [num_samples, sum(actual_dims)]
    concated_array: zarr.Array = ran_zarr_group.empty(
        random_array_name,
        shape=concated_shape,
        chunks=[sim_batch_size, concated_shape[-1]],  # <-fixing OOM?
        dtype=stored_array_list[-1].dtype,
        write_empty_chunks=False,
    )

    start_index = 0
    for i in range(len(actual_dims)):
        actual_dim = actual_dims[i]
        stored_array = stored_array_list[i]
        concated_array[:, start_index : start_index + actual_dim] = stored_array
        start_index += actual_dim
        del stored_array

    ran_zarr_group.attrs["okeys"] = actual_okeys
    ran_zarr_group.attrs["odims"] = actual_odims
    if verbose:
        print(concated_array.info)
    return concated_array, actual_dims, actual_okeys, actual_odims, stored_path_list


def save_sim_results_to_disk(
    result: OrderedDict | dict,
    save_path: str,
    zarr_group: zarr.Group,
) -> None:
    for sim_output, v in result.items():
        sim_output_path = f"{save_path}/{sim_output}"
        arr: np.ndarray = v
        stored_arr = zarr_group.empty_like(name=sim_output_path, data=arr)
        stored_arr[:] = arr


def get_dtype_by_param_name(param_name: str) -> Type:
    if param_name in ["leak_status", "overflow", "cv", "bulk_coeff"]:
        return bool
    elif param_name in ["initial_status", "status"]:
        return IntEnum
    else:
        return float


def create_pattern(
    inplaced_wn: wntr.network.WaterNetworkModel,
    obj: Any,
    value: np.ndarray | float | list,
) -> str:
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
    elif isinstance(value, float):
        value = [value if value != PAD_TOKEN else 0]
    inplaced_wn.add_pattern(new_pattern_name, pattern=value)
    return new_pattern_name


def create_curve(
    inplaced_wn: wntr.network.WaterNetworkModel,
    obj: Any,
    curve_type: Literal["HEAD", "HEADLOSS", "VOLUME", "EFFICIENCY"],
    value: np.ndarray | float | list,
    order: Literal["interleave", "concat"] = "concat",
    do_sort: bool = True,
) -> str:
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
        # if do_sort
        #   xs and ys will be sorted ascending and descending respectively.
        #   The created curve will have this shape:
        #   head
        #   |_____
        #   |      \_____
        #   |             \
        #   |______________\____flow
        # else completely random at anchor points
        value = list(zip(xs, ys)) if not do_sort else list(zip(np.sort(xs), np.sort(ys)[::-1]))
    else:
        raise NotImplementedError(f"value has type ={type(value)}")
    inplaced_wn.add_curve(new_curve_name, curve_type=curve_type, xy_tuples_list=value)
    return new_curve_name


def assign_value_internal(
    obj: Any,
    inplaced_wn: wntr.network.WaterNetworkModel,
    param_name: str,
    value: np.ndarray | float,
    dtype: Type,
    is_curve: bool,
    is_pattern: bool,
    time_dim: int,
    timestep: int = -1,
    flow_units: Optional[FlowUnits] = None,
    config: SimConfig | None = None,
) -> bool:
    has_attr = hasattr(obj, param_name)
    if has_attr:
        if param_name == "base_demand":
            # if flow_units is not None:
            #     #from_si_ret_value = from_si(flow_units, value, HydParam.Demand) #type:ignore
            #     #if from_si_ret_value != value:
            #         value = to_si(flow_units, value, HydParam.Demand) #type:ignore

            junc: wntre.Junction = obj

            ################TESTING####################################
            if junc.base_demand >= 0:
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
            #############################################################
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

            pump.efficiency = inplaced_wn.get_curve(new_curve_name)  # new_curve_name
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
        elif param_name == "init_level":
            tank: wntre.Tank = obj
            assert config is not None
            init_level_range: tuple | None = config.tank_tune.init_level_values
            if init_level_range is not None:
                tank.max_level = max(init_level_range[1], init_level_range[0])
                tank.min_level = min(init_level_range[1], init_level_range[0])
            tank.init_level = value
        else:
            if dtype is bool:
                if isinstance(value, np.ndarray):
                    value = np.where(value >= 0.5, 1, 0).astype(bool)
                else:
                    value = True if value >= 0.5 else False
            elif dtype == IntEnum or dtype == Enum:
                if param_name == "initial_status":
                    my_enum = wntr.network.LinkStatus
                else:
                    raise NotImplementedError()
                unique_enum_values = set([e.value for n, e in my_enum._member_map_.items()])  # type:ignore
                enum_length = len(unique_enum_values)
                value = value * enum_length
                value = select_enum_value(cur_val=value, my_enum=my_enum)  # type: ignore
                ############################ In testing###################
                value = [name for name, member in my_enum.__members__.items() if member.value == value][0]  # type: ignore
                ############################ In testing###################
            setattr(obj, param_name, value)

    return has_attr


def update_single_wn(
    record: np.ndarray,
    wn: wntr.network.WaterNetworkModel,
    odims: OrderedDict,
    okeys: OrderedDict,
    config: SimConfig,
) -> bool:
    sim_duration: int = config.duration
    time_step: int = config.time_step

    time_dim = (1 if sim_duration <= 1 else sim_duration) // time_step
    # update wn

    current_index = 0
    for component, params in okeys.items():
        is_node = component in ["junction", "tank", "reservoir", "node"]
        obj_names = get_object_name_list_by_component(component, wn)
        param_dims = odims[component]
        for i in range(len(params)):
            param_name = params[i]
            param_dim = param_dims[i]
            param_dtype = get_dtype_by_param_name(param_name)

            is_curve = param_name in get_curve_parameters()
            is_pattern = param_name in get_pattern_parameters()

            param_values = record[current_index : current_index + param_dim]
            # if verbose:
            #    print(f'current_index = {current_index}| param_dim = {param_dim} | param_name = {param_name} | param_values shape = {param_values.shape}')
            if is_pattern:
                param_values = np.reshape(param_values, [-1, time_dim])
            elif is_curve:
                param_values = np.reshape(param_values, [len(obj_names), -1])

            flow_units = get_flow_units(wn=wn)
            for j, obj_name in enumerate(obj_names):
                param_value = param_values[j]
                obj = wn.get_node(obj_name) if is_node else wn.get_link(obj_name)
                ret = assign_value_internal(
                    obj,
                    inplaced_wn=wn,
                    param_name=param_name,
                    value=param_value,
                    dtype=param_dtype,
                    is_curve=is_curve,
                    is_pattern=is_pattern,
                    time_dim=time_dim,
                    timestep=-1,
                    flow_units=flow_units,
                    config=config,
                )
                if not ret:
                    return False
            current_index += param_dim

    return True


def gather_flatten_sim_outputs(
    result: wntr.sim.SimulationResults,
    wn: wntr.network.WaterNetworkModel,
    sim_outputs: list[
        Literal[
            "pressure",
            "head",
            "demand",
            "flowrate",
            "velocity",
            "headloss",
            "friction_factor",
        ]
    ],
    time_dim: int,
    time_consistency: bool,
    onames: OrderedDict,
    record_check_fn_list: list[CheckRecordFn] = [],
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    result_dict = {}

    df_dict = {}

    for sim_output_key in sim_outputs:
        if sim_output_key in result.node:  # type:ignore
            node_or_link = result.node
            ordered_names = onames["node"]  # wn.node_name_list
        elif sim_output_key in result.link:  # type:ignore
            node_or_link = result.link
            ordered_names = onames["link"]  # wn.link_name_list
        else:
            raise KeyError(f"sim output {sim_output_key} is not a valid key")
        df: pd.DataFrame = node_or_link[sim_output_key]  # type:ignore

        try:
            df = df[ordered_names]  # type:ignore
        except KeyError:
            diff_names = list(set(ordered_names).symmetric_difference(df.columns.names))
            raise KeyError(f"!df has no col names: {diff_names}")
        # first_null_index = df.isnull().any(axis=1).idxmax()
        # if verbose:
        #     print(f'detect first null index = {first_null_index} | df shape = {df.shape}')
        # array = df[:first_null_index].to_numpy()

        isnan = df.isnull().values.any()
        if isnan:
            if not time_consistency:
                df = df.dropna(how="all")
            else:
                if verbose:
                    print(f"REJECTED: {sim_output_key} array exists nan value!")
                return {}

        df_dict[sim_output_key] = df

    # extra values for resilience metrics
    # mri
    # junc_pressure_df = result.node['pressure'][wn.junction_name_list].dropna() #type:ignore
    # junc_elevation_df = wn.query_node_attribute('elevation')[wn.junction_name_list].dropna() #type:ignore
    # df_dict['mri'] =  wntr.metrics.modified_resilience_index(pressure=junc_pressure_df, elevation=junc_elevation_df, Pstar= 1e-6)

    # check whether the record is valid
    for i, record_check_fn in enumerate(record_check_fn_list):
        if not record_check_fn(df_dict):
            if verbose:
                print(f"REJECTED: Output array disastifies record checking function {i}")
            return {}

    for k, df in df_dict.items():
        if k in sim_outputs:
            array = df.to_numpy()
            if array is None and np.any(np.isnan(array)):
                if verbose:
                    print(f"REJECTED: {k} array exists nan value!")
                return {}

            if array.shape[0] > time_dim:
                array = array[:time_dim]

            if (array.shape[0] == time_dim and time_consistency) or not time_consistency:
                result_dict[k] = array.flatten()
            else:
                if verbose:
                    print("REJECTED: Number of output snapshots is not equal to predefined duration!")
                return {}
    return result_dict


def update_and_simulate_wn(
    batch_indices: list[int],
    batch_records: np.ndarray,
    wn: wntr.network.WaterNetworkModel,
    odims: OrderedDict,
    okeys: OrderedDict,
    onames: OrderedDict,
    config: SimConfig,
    record_check_fn_list: list[CheckRecordFn] = [],
    save_success_inp_path: str = "ditec_wdn_dataset/debug",
) -> tuple[dict[str, np.ndarray], list[int]]:
    assert len(batch_indices) == batch_records.shape[0]

    sim_duration: int = config.duration
    time_step: int = config.time_step
    sim_outputs = config.sim_outputs
    time_consistency: bool = config.time_consistency
    verbose: bool = config.verbose
    save_success_inp: bool = config.save_success_inp
    tmp_dir: str = config.temp_path

    time_dim = (1 if sim_duration <= 1 else sim_duration) // time_step
    batch_size = batch_records.shape[0]
    total_dict = {k: [] for k in sim_outputs}
    success_indices = []
    # with tempfile.TemporaryDirectory() as tmp_dir:
    sim = wntr.sim.EpanetSimulator(wn)
    saved_name = os.path.join(tmp_dir, f"sim_{batch_indices[0]}")
    for b in range(batch_size):
        wn.reset_initial_values()
        if update_single_wn(record=batch_records[b], wn=wn, odims=odims, okeys=okeys, config=config):
            try:
                res = sim.run_sim(saved_name, version=2.2)
                # result_dict {output_key : array has shape [time_dim, #nodes or #links] }
                result_dict = gather_flatten_sim_outputs(
                    res,
                    wn=wn,
                    sim_outputs=sim_outputs,
                    time_dim=time_dim,
                    time_consistency=time_consistency,
                    onames=onames,
                    record_check_fn_list=record_check_fn_list,
                    verbose=verbose,
                )
                assert not result_dict or (result_dict and set(sim_outputs) == set(result_dict.keys()))
                # update total_dict
                if len(result_dict) > 0:
                    if save_success_inp:
                        os.makedirs(save_success_inp_path, exist_ok=True)
                        save_inp_file(
                            save_inp_path=save_success_inp_path,
                            wn=wn,
                            save_name=f"{b}.inp",
                        )

                    for key, ndarray in result_dict.items():
                        total_dict[key].append(ndarray)

                    success_indices.append(batch_indices[b])
            except wntr.epanet.toolkit.EpanetException as e:
                if verbose:
                    print(f"Sim failed! Error = {e}")

    # return {k: np.stack(v,axis=0)  for k,v in total_dict.items() if v},success_indices

    ret_dict = {}
    for k, v in total_dict.items():
        if len(v) > 0:
            ret_dict[k] = np.stack(v, axis=0)
        else:
            return {}, []  # fail due to empty list

    return ret_dict, success_indices


def update_and_simulate_wn_small_chunk(
    batch_indices: list[int],
    batch_records: np.ndarray,
    wn: wntr.network.WaterNetworkModel,
    odims: OrderedDict,
    okeys: OrderedDict,
    onames: OrderedDict,
    config: SimConfig,
    record_check_fn_list: list[CheckRecordFn] = [],
    save_success_inp_path: str = "ditec_wdn_dataset/debug",
) -> Generator[tuple[dict[str, np.ndarray], list[int]], None, None]:
    assert len(batch_indices) == batch_records.shape[0]

    sim_duration: int = config.duration
    # time_step: int = config.time_step
    sim_outputs = config.sim_outputs
    time_consistency: bool = config.time_consistency
    verbose: bool = config.verbose
    save_success_inp: bool = config.save_success_inp
    tmp_dir: str = config.temp_path

    time_dim = 1 if sim_duration <= 1 else sim_duration
    batch_size = batch_records.shape[0]
    # total_dict = {k: [] for k in sim_outputs}
    # success_indices = []
    # with tempfile.TemporaryDirectory() as tmp_dir:
    sim = wntr.sim.EpanetSimulator(wn)
    saved_name = os.path.join(tmp_dir, f"sim_{batch_indices[0]}")
    for b in range(batch_size):
        wn.reset_initial_values()
        if update_single_wn(
            record=batch_records[b],
            wn=wn,
            odims=odims,
            okeys=okeys,
            config=config,
        ):
            try:
                res = sim.run_sim(saved_name, version=2.2)
                # result_dict {output_key : array has shape [time_dim, #nodes or #links] }
                result_dict = gather_flatten_sim_outputs(
                    res,
                    wn=wn,
                    sim_outputs=sim_outputs,
                    time_dim=time_dim,
                    time_consistency=time_consistency,
                    onames=onames,
                    record_check_fn_list=record_check_fn_list,
                    verbose=verbose,
                )
                assert not result_dict or (result_dict and set(sim_outputs) == set(result_dict.keys()))
                # update total_dict
                if len(result_dict) > 0:
                    if save_success_inp:
                        os.makedirs(save_success_inp_path, exist_ok=True)
                        save_inp_file(
                            save_inp_path=save_success_inp_path,
                            wn=wn,
                            save_name=f"{b}.inp",
                        )

                    tf_result_dict = {k: np.expand_dims(v, axis=0) for k, v in result_dict.items()}
                    yield (tf_result_dict, [batch_indices[b]])
            except wntr.epanet.toolkit.EpanetException as e:
                if verbose:
                    print(f"Sim failed! Error = {e}")


# @ray.remote(num_cpus=1,max_restarts=1)
@ray.remote(num_cpus=FRACTION_CPU_USAGE_PER_UPSI_WORKER, max_restarts=1)
class UpSiWorker:
    def __init__(
        self,
        wn: wntr.network.WaterNetworkModel,
        odims: OrderedDict,
        okeys: OrderedDict,
        onames: OrderedDict,
        config: SimConfig,
        record_check_fn_list: list[CheckRecordFn] = [],
        save_success_inp_path: str = "ditec_wdn_dataset/debug",
    ) -> None:
        self.wn = wn
        self.odims = odims
        self.okeys = okeys
        self.onames = onames
        self.config = config
        self.save_success_inp_path = save_success_inp_path
        self.record_check_fn_list = record_check_fn_list

    def update_and_simulate(self, batch_indices: list[int], batch_records: np.ndarray):
        return update_and_simulate_wn(
            batch_indices,
            batch_records,
            self.wn,
            self.odims,
            self.okeys,
            self.onames,
            self.config,
            self.record_check_fn_list,
            self.save_success_inp_path,
        )

    @ray.method(num_returns="dynamic")
    def update_and_simulate_small_chunk(
        self, batch_indices: list[int], batch_records: np.ndarray
    ) -> Generator[tuple[dict[str, np.ndarray], list[int]], None, None]:
        return update_and_simulate_wn_small_chunk(
            batch_indices,
            batch_records,
            self.wn,
            self.odims,
            self.okeys,
            self.onames,
            self.config,
            self.record_check_fn_list,
            self.save_success_inp_path,
        )

    def exit(self):
        ray.actor.exit_actor()


@ray.remote(num_cpus=FRACTION_CPU_USAGE_PER_UPSI_WORKER)
def ray_update_and_simulate_wn(
    batch_indices: list[int],
    batch_records: np.ndarray,
    wn: wntr.network.WaterNetworkModel,
    odims: OrderedDict,
    okeys: OrderedDict,
    onames: OrderedDict,
    config: SimConfig,
    record_check_fn_list: list[CheckRecordFn] = [],
) -> tuple[dict[str, np.ndarray], list[int]]:
    return update_and_simulate_wn(
        batch_indices,
        batch_records,
        wn,
        odims,
        okeys,
        onames,
        config,
        record_check_fn_list,
    )


def copy_inputs_and_save_outputs(
    zarr_group: zarr.Group,
    result: tuple[dict[str, np.ndarray], list[int]],
    tmp_stored_name_list: list[str],
    root_path: str,
    base_name: str,
    config: SimConfig,
    odims: OrderedDict,
    okeys: OrderedDict,
) -> dict[str, int]:
    """From outputs, we retrieve indices and gather input records at exact positions, then save them into ZARR
    Also, we track these indices for restore purpose. The tracking is not an optimal solution.
    Returns:
        dict[str,int]: dict of compo_param and its current number of actual records.
    """
    stored_name_samples = {}
    # base_name + '_' + config.sim_outputs[0]
    expected_samples = config.num_samples
    batch_size = config.batch_size
    result_dict, success_ids = result[0], result[1]
    # result_dict_size = sys.getsizeof(result_dict)
    # success_ids_size = sys.getsizeof(success_ids)
    # print(f"size of result_dict = {result_dict_size}")
    # print(f"size of success_ids = {success_ids_size}")
    # print(f"size of total = {success_ids_size +result_dict_size}")
    index_tracers = []
    for tmp_stored_name in tmp_stored_name_list:
        component = tmp_stored_name.replace("_temp", "").replace(f"{base_name}_", "")
        assert component in okeys and component in odims
        tmp_stored_path = root_path + "/" + tmp_stored_name
        tmp_arr = zarr.open(store=tmp_stored_path, mode="r")
        params = okeys[component]
        param_dims = odims[component]

        # num_batches = math.ceil(tmp_arr.shape[0] / batch_size)
        current_index = 0

        for i in range(len(params)):
            param_name = params[i]
            param_dim = param_dims[i]

            compo_param_name = component + "_" + param_name
            if compo_param_name not in list(zarr_group.array_keys()):
                in_arr: zarr.Array = zarr_group.empty(
                    name=compo_param_name,
                    shape=[0, param_dim],
                    chunks=[batch_size, param_dim],
                    dtype=tmp_arr.dtype,
                    write_empty_chunks=False,
                )  # zarr_group.empty_like(name=inp_stored_name,data=tmp_arr, write_empty_chunks=False) #type:ignore
            else:
                in_arr: zarr.Array = zarr_group[compo_param_name]  # type:ignore #zarr.open_array(inp_stored_path,mode='w', write_empty_chunks=False) #

            if compo_param_name not in stored_name_samples:
                stored_name_samples[compo_param_name] = 0

            current_runs = 0
            for success_id in success_ids:
                tmp_row = tmp_arr[success_id, current_index : current_index + param_dim]
                assert not np.any(np.isnan(tmp_row))  # type:ignore
                tmp_row = np.reshape(tmp_row, [1, -1])  # type:ignore
                if in_arr.shape[0] < expected_samples:
                    in_arr.append(tmp_row)
                    current_runs += 1
                    if success_id not in index_tracers:
                        index_tracers.append(success_id)

            stored_name_samples[compo_param_name] += current_runs
            current_index += param_dim
    if "index_tracers" not in zarr_group.attrs:
        zarr_group.attrs["index_tracers"] = index_tracers
    else:
        stored_index_tracers: list = zarr_group.attrs["index_tracers"]
        stored_index_tracers.extend(index_tracers)
        zarr_group.attrs["index_tracers"] = stored_index_tracers
    ##########################################

    # save successful outputs
    for sim_output_key, sim_output_array in result_dict.items():
        out_stored_name = sim_output_key  # base_name + '_' + sim_output_key
        if out_stored_name not in list(zarr_group.array_keys()):
            output_length = sim_output_array.shape[-1]
            out_arr: zarr.Array = zarr_group.empty(
                name=out_stored_name,
                shape=[0, output_length],
                chunks=[batch_size, output_length],
                dtype=sim_output_array.dtype,
                write_empty_chunks=False,
            )
            stored_name_samples[out_stored_name] = 0
        else:
            out_arr: zarr.Array = zarr_group[out_stored_name]  # zarr.open_array(out_stored_path, mode='w', write_empty_chunks=False) #type:ignore

        if out_stored_name not in stored_name_samples:
            stored_name_samples[out_stored_name] = 0

        assert sim_output_array.shape[0] == len(success_ids), f"sim_output_array.shape[0] = {sim_output_array.shape[0]} | success_ids = {success_ids}"

        out_current_runs = 0
        for b in range(sim_output_array.shape[0]):
            tmp_row = sim_output_array[b, :]
            assert not np.any(np.isnan(tmp_row))  # type:ignore
            tmp_row = np.reshape(tmp_row, [1, -1])  # type:ignore
            if out_arr.shape[0] < expected_samples:
                out_arr.append(tmp_row)
                out_current_runs += 1
        stored_name_samples[out_stored_name] += out_current_runs
        # assert out_current_runs == current_runs, f'out_run = {out_current_runs} | inp_run = {current_runs}|expected_samples = {expected_samples}| in_arr.shape[0] ={in_arr.shape[0]} | out_arr.shape[0] ={out_arr.shape[0] }'

    return stored_name_samples


def update_store_name_samples(total_dict: dict[str, int], sub_dict: dict[str, int]) -> dict[str, int]:
    for k in sub_dict:
        if k not in total_dict:
            total_dict[k] = 0
        total_dict[k] += sub_dict[k]
    return total_dict


def del_temp_and_rechunk(
    tmp_stored_name_list: list[str],
    root_path: str,
    zarr_group: zarr.Group,
    stored_name_samples: dict[str, int],
    verbose: bool = True,
):
    # del all temp folders
    for tmp_stored_name in tmp_stored_name_list:
        tmp_stored_path = root_path + "/" + tmp_stored_name
        shutil.rmtree(tmp_stored_path)

    def get_optimal_divisor(n: int):
        if n <= 1:
            return 1
        divisor = n
        for i in range(n // 2, 1):
            if n % i == 0:
                divisor = i
                break
        return divisor

    arr_names = list(zarr_group.array_keys())

    for arr_name in arr_names:
        arr_name = str(arr_name)
        param_path = root_path + "/" + arr_name  # type:ignore
        arr: zarr.Array = zarr_group[arr_name]  # type:ignore
        if arr_name in stored_name_samples:
            chunk_shape = arr.chunks  # type:ignore
            need_rechunking = stored_name_samples[arr_name] < chunk_shape[0]
            if need_rechunking:
                shape0 = get_optimal_divisor(stored_name_samples[arr_name])  # type:ignore
                shape1 = arr.shape[1]
                with tempfile.TemporaryDirectory() as temp_path:
                    temp_store = zarr.DirectoryStore(path=temp_path)
                    temp_group = zarr.group(store=temp_store, overwrite=True)
                    temp_array = temp_group.array(
                        name=arr_name,
                        data=arr[:],
                        chunks=(shape0, shape1),
                        write_empty_chunks=False,
                    )

                    zarr.copy(temp_array, dest=zarr_group, name=arr_name, if_exists="replace")
                    temp_store.close()
                arr: zarr.Array = zarr.open_array(param_path)  # type:ignore
            else:
                arr.resize(stored_name_samples[arr_name], arr.shape[-1])  # type:ignore

        if verbose:
            print("#" * 40 + f"{arr_name}: {arr.shape[0]}/{stored_name_samples[arr_name]}" + "#" * 20)
            print(f"param_path = {param_path}")
            print(arr.info)  # type:ignore


def single_update_simulate_save_wn(
    zarr_group: zarr.Group,
    root_path: str,
    base_name: str,
    tmp_stored_name_list: list[str],
    sim_set: zarr.Array,
    wn: wntr.network.WaterNetworkModel,
    config: SimConfig,
    odims: OrderedDict,
    okeys: OrderedDict,
    onames: OrderedDict,
    record_check_fn_list: list[CheckRecordFn] = [],
    verbose: bool = False,
    fractional_cpu_usage: float = 1.0,
    inp_id: int = 0,
):
    stored_name_samples: dict[str, int] = {}
    batch_size = config.batch_size
    loader = SimpleDataLoader(sim_set, batch_size=batch_size, shuffle=False)

    expected_runs = config.num_samples
    total_runs = int(config.num_samples * config.backup_times)
    num_success_runs = 0
    first_output_key = config.sim_outputs[0]
    with tqdm(total=len(loader), leave=not config.verbose) as pbar:
        if not config.verbose:
            pbar.close()
        for i, batch_set in enumerate(loader):
            batch_id_list: list[int] = list(range(i * batch_size, i * batch_size + batch_set.shape[0]))

            if num_success_runs >= expected_runs:
                break

            result = update_and_simulate_wn(
                batch_id_list,
                batch_set,
                deepcopy(wn),
                odims,
                okeys,
                onames,
                deepcopy(config),
                record_check_fn_list,
                "ditec_wdn_dataset/debug",
            )

            sub_stored_name_samples = copy_inputs_and_save_outputs(  # copy_inputs_and_save_outputs2(
                zarr_group=zarr_group,
                result=result,
                tmp_stored_name_list=tmp_stored_name_list,
                root_path=root_path,
                base_name=base_name,
                config=config,
                odims=odims,
                okeys=okeys,
            )
            stored_name_samples = update_store_name_samples(stored_name_samples, sub_stored_name_samples)
            num_success_runs = stored_name_samples[first_output_key] if first_output_key in stored_name_samples else 0
            if config.verbose:
                pbar.update(1)
                pbar.set_description(f"Success ({num_success_runs}) | Expected: ({expected_runs}) | Total ({total_runs})- Processing batches")

    del_temp_and_rechunk(
        tmp_stored_name_list=tmp_stored_name_list,
        root_path=root_path,
        zarr_group=zarr_group,
        stored_name_samples=stored_name_samples,
        verbose=verbose,
    )


def ray_collect_results(
    zarr_group: zarr.Group,
    tmp_stored_name_list: list[str],
    root_path: str,
    base_name: str,
    config: SimConfig,
    odims: OrderedDict,
    okeys: OrderedDict,
    stored_name_samples: dict[str, int],
    result_dict: dict,
) -> dict[str, int]:
    # update result_refs to only
    # track the remaining tasks.
    ready_refs, unready_refs = ray.wait(list(result_dict), num_returns=1)
    del unready_refs
    ready_ref = ready_refs[0]

    tuple_or_generator: Generator[tuple[dict[str, np.ndarray], list[int]], None, None] | tuple[dict[str, np.ndarray], list[int]] = ray.get(ready_ref)
    if isinstance(tuple_or_generator, tuple):  # tuple[dict[str, np.ndarray], list[int]]
        sub_stored_name_samples = copy_inputs_and_save_outputs(  # copy_inputs_and_save_outputs2(
            zarr_group=zarr_group,
            result=tuple_or_generator,
            tmp_stored_name_list=tmp_stored_name_list,
            root_path=root_path,
            base_name=base_name,
            config=config,
            odims=odims,
            okeys=okeys,
        )
        stored_name_samples = update_store_name_samples(stored_name_samples, sub_stored_name_samples)
        del sub_stored_name_samples

    else:  # generator Ray Ref
        for result_ref in tuple_or_generator:
            result = ray.get(result_ref)  # type:ignore
            # num_success_runs = copy_inputs_and_save_outputs(result)
            sub_stored_name_samples = copy_inputs_and_save_outputs(  # copy_inputs_and_save_outputs2(
                zarr_group=zarr_group,
                result=result,
                tmp_stored_name_list=tmp_stored_name_list,
                root_path=root_path,
                base_name=base_name,
                config=config,
                odims=odims,
                okeys=okeys,
            )
            stored_name_samples = update_store_name_samples(stored_name_samples, sub_stored_name_samples)
            del sub_stored_name_samples
            del result

    active_worker = result_dict.pop(ready_ref)

    del ready_ref
    del ready_refs
    ray.kill(active_worker)
    del active_worker  # leak mem if reuse
    del tuple_or_generator

    return stored_name_samples


def update_process_bar(
    inplaced_pbar: tqdm,
    num_success_runs: int,
    expected_runs: int,
    total_runs: int,
    overwatch: bool = False,
) -> None:
    inplaced_pbar.update(1)
    if overwatch:
        used_memory_percentage = psutil.virtual_memory().used / psutil.virtual_memory().total
        inplaced_pbar.set_description(
            f"Success ({num_success_runs}) | Expected ({expected_runs}) | Total ({total_runs}) | RAM Usage ({used_memory_percentage:.2f})%- Processing batches"
        )
    else:
        inplaced_pbar.set_description(f"Success ({num_success_runs}) | Expected ({expected_runs}) | Total ({total_runs})- Processing batches")


def ray_update_simulate_save_wn(
    zarr_group: zarr.Group,
    root_path: str,
    base_name: str,
    tmp_stored_name_list: list[str],
    sim_set: zarr.Array,
    wn: wntr.network.WaterNetworkModel,
    config: SimConfig,
    odims: OrderedDict,
    okeys: OrderedDict,
    onames: OrderedDict,
    record_check_fn_list: list[CheckRecordFn] = [],
    verbose: bool = False,
    inp_id: int = 0,
):
    stored_name_samples: dict[str, int] = {}
    batch_size = config.batch_size

    config_mem = config.mem_per_worker * ONE_GB_IN_BYTES
    available_mem = psutil.virtual_memory().available
    total_mem = psutil.virtual_memory().total
    expected_input_mem_per_upsi_worker = sim_set.chunks[0] * sim_set.chunks[1] * 8

    actual_num_workers = int(config.num_cpus * config.fractional_cpu_usage)
    if config.verbose:
        num_components_per_output = [
            len(wn.node_name_list) if is_node_simulation_output(sim_output_key) else len(wn.link_name_list) for sim_output_key in config.sim_outputs
        ]

        expected_output_mem_per_upsi_worker = float(sum(num_components_per_output)) * (config.duration / config.time_step) * config.batch_size

        config_mem_gb = config_mem / ONE_GB_IN_BYTES
        expected_input_mem_gb = expected_input_mem_per_upsi_worker / ONE_GB_IN_BYTES
        expected_output_mem_gb = expected_output_mem_per_upsi_worker / ONE_GB_IN_BYTES
        print(
            f"RAM for an upsi worker: {config_mem_gb:.2f} GB (Config) | {expected_input_mem_gb:.2f} GB (Expected Input) | {expected_output_mem_gb:.2f} GB (Expected Ouput)"
        )

        if config.num_cpus > 1:
            plural_config_mem_gb = actual_num_workers * config_mem_gb
            plural_expected_input_mem_gb = actual_num_workers * expected_input_mem_gb
            plural_expected_output_mem_gb = actual_num_workers * expected_output_mem_gb
            print(
                f"RAM for {actual_num_workers} upsi workers: {plural_config_mem_gb:.2f} GB (Config) | {plural_expected_input_mem_gb:.2f} GB (Expected Input) | {plural_expected_output_mem_gb:.2f} GB (Expected Ouput)"
            )

        expected_input_mem_in_main_process = sim_set.chunks[0] * sim_set.chunks[1] * 8 * config.num_cpus * config.fractional_cpu_usage
        expected_output_mem_in_main_process = expected_output_mem_per_upsi_worker * config.num_cpus * config.fractional_cpu_usage
        expected_input_mem_in_main_process = expected_input_mem_in_main_process / ONE_GB_IN_BYTES
        expected_output_mem_in_main_process = expected_output_mem_in_main_process / ONE_GB_IN_BYTES
        print(
            f"RAM for main process:  {expected_input_mem_in_main_process:.2f} GB (Expected Input) | {expected_output_mem_in_main_process:.2f} GB (Expected Ouput)"
        )

        print(f"Available RAM: {available_mem / ONE_GB_IN_BYTES} GB")
        print(f"Total RAM: {total_mem / ONE_GB_IN_BYTES} GB")

    ray_init_wrapper(config, True, object_store_memory=expected_input_mem_per_upsi_worker * actual_num_workers + ONE_GB_IN_BYTES)
    WatcherManager.track(f"Simulation-{inp_id}-ray-loader")

    loader = SimpleDataLoader(sim_set, batch_size=batch_size, shuffle=False)
    # data_ref_ids = []
    # data_batch_ids = []
    # for i, batch_set in enumerate(loader):
    #     # data_ref_ids.append(ray.put(batch_set))
    #     batch_id_list: list[int] = list(range(i * batch_size, i * batch_size + batch_set.shape[0]))
    #     data_batch_ids.append(batch_id_list)

    WatcherManager.stop(f"Simulation-{inp_id}-ray-loader")

    MAX_NUM_PENDING_TASKS = config.num_cpus // config.fractional_cpu_usage  # FRACTION_CPU_USAGE_PER_UPSI_WORKER #cpu_count()
    result_dict = {}
    expected_runs = config.num_samples
    total_runs = int(config.num_samples * config.backup_times)
    num_success_runs = 0
    first_output_key = config.sim_outputs[0]
    # with tqdm(total=len(loader), leave=not config.verbose) as pbar:
    num_batchs = len(loader)

    WatcherManager.track(f"Simulation-{inp_id}-ray-process")

    pbar = tqdm(total=num_batchs)
    if not config.verbose:
        pbar.leave = True
        pbar.close()

    i = -1
    first_gathered: bool = False
    overwatch: bool = WatcherManager.overwatch
    while num_success_runs < expected_runs:
        try:
            batch_set = next(loader)
            i += 1
        except StopIteration:
            break

        if not first_gathered and overwatch:
            WatcherManager.stop(f"Simulation-{inp_id}-ray-next-{i}")
            WatcherManager.track(f"Simulation-{inp_id}-ray-createworker-{i}")
        assert batch_set is not None
        batch_id_list: list[int] = list(range(i * batch_size, i * batch_size + batch_set.shape[0]))

        worker = UpSiWorker.options(  # type: ignore
            num_cpus=config.fractional_cpu_usage, memory=config_mem
        ).remote(  # type:ignore
            wn=deepcopy(wn),  # type:ignore
            odims=odims,
            okeys=okeys,
            onames=onames,
            config=config,
            record_check_fn_list=record_check_fn_list,
        )
        if config.yield_worker_generator:
            # return a generator that yields smaller tuple (consume less heap, but "lock the parallelism while gathering small tuples" (validating))
            result_dict[worker.update_and_simulate_small_chunk.remote(batch_id_list, batch_set)] = worker
        else:
            # return a full tuple (consume more heap memory, but "maintaining parallelism" (validating))
            result_dict[worker.update_and_simulate.remote(batch_id_list, batch_set)] = worker

        del batch_id_list
        del batch_set

        if i >= 0 and not first_gathered and overwatch:
            check_usage_if_verbose("end_createworker", config.verbose)
            WatcherManager.stop(f"Simulation-{inp_id}-ray-createworker-{i}")

        used_memory_percentage = psutil.virtual_memory().used / psutil.virtual_memory().total
        if len(result_dict) >= MAX_NUM_PENDING_TASKS or used_memory_percentage > USED_MEMORY_THRESHOLD:
            if not first_gathered and overwatch:
                WatcherManager.track(f"Simulation-{inp_id}-ray-gather-{i}")

            stored_name_samples = ray_collect_results(
                zarr_group=zarr_group,
                tmp_stored_name_list=tmp_stored_name_list,
                root_path=root_path,
                base_name=base_name,
                config=config,
                odims=odims,
                okeys=okeys,
                stored_name_samples=stored_name_samples,
                result_dict=result_dict,
            )

            num_success_runs = stored_name_samples[first_output_key] if first_output_key in stored_name_samples else 0
            if config.verbose:
                update_process_bar(
                    inplaced_pbar=pbar,
                    num_success_runs=num_success_runs,
                    expected_runs=expected_runs,
                    total_runs=total_runs,
                    overwatch=overwatch,
                )

            if not first_gathered and overwatch:
                first_gathered = True
                check_usage_if_verbose("end_gather", config.verbose)
                WatcherManager.stop(f"Simulation-{inp_id}-ray-gather-{i}")

    WatcherManager.stop(f"Simulation-{inp_id}-ray-process")
    WatcherManager.track(f"Simulation-{inp_id}-ray-post_process")
    first_gathered = False
    while len(result_dict) > 0:
        if not first_gathered and overwatch:
            WatcherManager.track(f"Simulation-{inp_id}-ray-lategather")

        if num_success_runs < expected_runs:
            stored_name_samples = ray_collect_results(
                zarr_group=zarr_group,
                tmp_stored_name_list=tmp_stored_name_list,
                root_path=root_path,
                base_name=base_name,
                config=config,
                odims=odims,
                okeys=okeys,
                stored_name_samples=stored_name_samples,
                result_dict=result_dict,
            )
            num_success_runs = stored_name_samples[first_output_key] if first_output_key in stored_name_samples else 0
            if config.verbose:
                update_process_bar(
                    inplaced_pbar=pbar,
                    num_success_runs=num_success_runs,
                    expected_runs=expected_runs,
                    total_runs=total_runs,
                    overwatch=overwatch,
                )

        else:
            task_list = list(result_dict.keys())
            for task in task_list:
                actor_ref = result_dict.pop(task)
                ray.cancel(task)
                ray.kill(actor_ref)
                del task
                del actor_ref
            del task_list

        if not first_gathered and overwatch:
            first_gathered = True
            WatcherManager.stop(f"Simulation-{inp_id}-ray-lategather")

    WatcherManager.stop(f"Simulation-{inp_id}-ray-post_process")
    del_temp_and_rechunk(
        tmp_stored_name_list=tmp_stored_name_list,
        root_path=root_path,
        zarr_group=zarr_group,
        stored_name_samples=stored_name_samples,
        verbose=verbose,
    )
    if overwatch:
        check_usage_if_verbose("end_ray", config.verbose)


def check_pressure_in_range(
    df_dict: dict[str, pd.DataFrame],
    skip_names: list[str] = [],
    pressure_range: tuple[float, float] = (-1e-4, 200),
    verbose: bool = True,
) -> bool:
    min_pressure: float = min(pressure_range[0], pressure_range[1])
    max_pressure: float = max(pressure_range[0], pressure_range[1])

    exceed_limit_amount = 10
    if "pressure" in df_dict:
        df = df_dict["pressure"]

        # print(f'before-len of columns = {len(df.columns)}')
        if skip_names is not None and len(skip_names) > 0:
            df = df.drop(columns=skip_names, axis=1, errors="ignore")
            # df = df.drop(columns=skip_names, errors="ignore")

        arr = df.to_numpy()
        is_validated = not np.any(np.logical_or(arr < min_pressure, arr > max_pressure))
        if verbose:
            columns_less_than_min = df.columns[(df < min_pressure).any()].tolist()
            # columns_greater_than_151 = df.columns[(df > 151).any()].tolist()
            columns_greater_than_max = df.columns[(df > max_pressure).any()].tolist()

            less_min_len = len(columns_less_than_min)
            greater_max_len = len(columns_greater_than_max)

            if less_min_len <= exceed_limit_amount and less_min_len > 0:
                left_text = f"Columns < ({min_pressure}) = {columns_less_than_min}"
            else:
                left_text = f"Num of columns < ({min_pressure}) = {less_min_len}"

            if greater_max_len <= exceed_limit_amount and greater_max_len > 0:
                right_text = f"Columns > ({max_pressure}) = {columns_greater_than_max}"
            else:
                right_text = f"Num of > ({max_pressure}) = {greater_max_len}"

            if left_text != "" or right_text != "":
                print(f"{left_text} | {right_text} | {is_validated}")

        return is_validated
    else:
        return True


def ray_init_wrapper(config: SimConfig, force_using_ray: bool = False, object_store_memory: Optional[int] = None) -> None:
    if config.num_cpus > 1 or force_using_ray:
        if not ray.is_initialized:
            num_cpus = min(config.num_cpus, cpu_count())
            if os.path.isabs(config.ray_temp_path):
                ray.init(num_cpus=num_cpus, _temp_dir=config.ray_temp_path, object_store_memory=object_store_memory)  # type:ignore
            else:
                abs_path = os.path.join(os.getcwd(), config.ray_temp_path)
                ray.init(num_cpus=num_cpus, _temp_dir=abs_path, object_store_memory=object_store_memory)  # type:ignore


def generate(config: SimConfig, force_using_ray: bool = False, bypass_checking_rules: bool = False, overwatch: bool = False) -> list[str]:
    """generate random matrix and feed random values into components. Then, we simulate and enscapsulate in/out data into zarr

    Args:
        config (SimConfig): main config
        force_using_ray (bool, optional): Flag indicates whether ray is utilize. Useful if you have only 1 cpu, and wanted to leverage fractional computation of it. Defaults to False.
        bypass_checking_rules (bool, optional): Flag allowing to skip validation checkers. Useful for debug. Defaults to False.
        overwatch (bool, optional): Flag turning memory profile mode. Extensively check and create memory profiler chart. Defaults to False.

    Raises:
        e: Exception! We do not handle it! Just catch for profiling and then RE-RAISE

    Returns:
        list[str]: list of zarr paths
    """
    ray_init_wrapper(config, force_using_ray)

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    saved_paths = []
    try:
        WatcherManager.overwatch = overwatch
        WatcherManager.track("Execution")
        for inp_id, inp_path in enumerate(config.inp_paths):
            wn = wntr.network.WaterNetworkModel(inp_path)

            wn.reset_initial_values()
            wn = init_water_network(
                wn,
                duration=config.duration,
                time_step=config.time_step,
                remove_controls=True,
                remove_curve=False,
                remove_patterns=False,
            )

            records = config._to_dict()

            postfix = datetime.today().strftime("%Y%m%d_%H%M")
            base_name = os.path.basename(inp_path)[:-4]
            tmp_base_name = base_name + "_temp"

            saved_file = f"simgen_{base_name}_{postfix}.zarr"
            saved_path = f"{config.output_path}/{saved_file}"

            store = zarr.DirectoryStore(path=saved_path)
            zarr_group = zarr.group(store=store, overwrite=True)

            WatcherManager.track(f"Generation-{inp_id}")

            # create a random folder to store concatenated array
            ran_dir = tempfile.TemporaryDirectory()
            ran_zarr_store = zarr.DirectoryStore(ran_dir.name)
            ran_zarr_group = zarr.group(store=ran_zarr_store)

            # gen random values
            ran_zarr, all_dims, actual_okeys, actual_odims, stored_path_list = prepare_simgen_records(
                config,
                prefix=tmp_base_name,
                zarr_group=zarr_group,  # type:ignore
                ran_zarr_group=ran_zarr_group,  # type:ignore
                wn=wn,
                batch_size=config.gen_batch_size,
                sim_batch_size=config.batch_size,
                valueGenerationFn=default_generation_func,
                verbose=config.verbose,
            )
            WatcherManager.stop(f"Generation-{inp_id}")

            if ray.is_initialized() and overwatch:
                ray.shutdown()  # <-remove undead workers

            ran_zarr.read_only = True
            sim_set = ran_zarr
            actual_onames = get_onames(actual_okeys=actual_okeys, wn=wn)

            # update attrs
            updated_record_dict = {}
            for k, v in records.items():
                if isinstance(v, tuple):
                    updated_record_dict[k] = list(v)
                elif dataclasses.is_dataclass(v):
                    for field in fields(v):
                        subv = getattr(v, field.name)
                        updated_record_dict[field.name] = subv if not isinstance(v, tuple) else list(subv)
                else:
                    updated_record_dict[k] = v

            zarr_group.attrs.update(updated_record_dict)
            # skip nodes list is empty ! Interface will handle skipping instead!
            # adj = get_adj_list(wn, skip_node_names=[])
            adj = build_adj_from_input(wn)["adj_list"]
            # adjacency list of edge tuples: [ (node_name_src1, node_name_dst1, edge_name1), (node_name_src2, node_name_dst2, edge_name2) ]
            zarr_group.attrs["adj_list"] = adj
            # ordered list of names per component (component1: [name1,name2] )
            zarr_group.attrs["onames"] = actual_onames
            # ordered list of parameter dimensions per component (component1: [dim_param1, dim_param2] )
            zarr_group.attrs["odims"] = actual_odims
            # ordered list of parameter names per components (component1: [param1, param2] )
            zarr_group.attrs["okeys"] = actual_okeys

            if config.verbose:
                print("#" * 40 + "Sim Config" + "#" * 40)
                pretty_print(config.as_dict())
                print("#" * 40 + "actual okeys" + "#" * 40)
                pretty_print(actual_okeys)
                print("#" * 40 + "actual odims" + "#" * 40)
                pretty_print(actual_odims)
                print("#" * 40 + "actual onames" + "#" * 40)
                pretty_print(actual_onames)
                print("#" * 40 + "actual auxils" + "#" * 40)
                print(f"Is Ray initialized: {ray.is_initialized()}")
                print(f"inpfile_units : {wn.options.hydraulic.inpfile_units}")
                print(f"#nodes = {len(wn.node_name_list)} | #links = {len(wn.link_name_list)}")
                print(f"Working on {saved_path}")
                print(f"sim_set shape = {sim_set.shape}")

            if config.num_cpus > 1 or force_using_ray:
                update_sim_save_fn = ray_update_simulate_save_wn
            else:
                print(f"config num_cpus = {config.num_cpus}! Ray is off!")
                update_sim_save_fn = single_update_simulate_save_wn

            record_check_fn_list: list[CheckRecordFn] = (
                [
                    partial(
                        check_pressure_in_range,
                        skip_names=config.skip_names,
                        pressure_range=(config.pressure_range[0], config.pressure_range[1]),
                        verbose=config.verbose,
                    ),
                ]
                if not bypass_checking_rules
                else []
            )
            WatcherManager.track(f"Simulation-{inp_id}")
            update_sim_save_fn(
                zarr_group=zarr_group,
                root_path=saved_path,
                base_name=base_name,
                tmp_stored_name_list=stored_path_list,
                sim_set=sim_set,
                wn=wn,
                config=config,
                odims=actual_odims,
                okeys=actual_okeys,
                onames=actual_onames,
                record_check_fn_list=record_check_fn_list,
                verbose=config.verbose,
                inp_id=inp_id,
            )
            WatcherManager.stop(f"Simulation-{inp_id}")
            # del unused stuffsbe
            ran_dir.cleanup()
            saved_paths.append(saved_path)

        WatcherManager.stop("Execution")

    except Exception as e:
        print("ERROR! An error occurred:", e)
        traceback.print_exc()
        # before system crash, we dump memory and print out
        WatcherManager.track("Crashing")
        WatcherManager.stop("Crashing", do_valid_stop=False)  # <-indicate a fail
    finally:
        WatcherManager.stop_all()
        if config.verbose:
            print(WatcherManager.report())
    return saved_paths
