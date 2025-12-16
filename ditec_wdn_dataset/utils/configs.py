#
# Created on Tue Feb 13 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Store all config template
# ------------------------------
#

from tap import Tap
import yaml
from typing import Any, Literal, Optional, Union
import os
from dataclasses import dataclass, asdict


Strategy = Literal["sampling", "series", "keep", "perturbation", "adg", "adg_v2", "factor", "substitute", "terrain"]


class AbstractConfig(Tap):
    @staticmethod
    def _int_or_none(string: str) -> int | None:
        return int(string) if string is not None else None

    @staticmethod
    def _str_or_none(string: str) -> str | None:
        return string if string is not None else None

    def _from_yaml(self, yaml_path: str, skip_unsettable: bool = False, unsafe_load: bool = False) -> None:
        assert yaml_path[-4:] == "yaml", f"path is not yaml file. yaml_path = {yaml_path}"
        assert os.path.isfile(yaml_path)

        with open(yaml_path, "r") as yaml_in:
            if unsafe_load:
                yaml_object = yaml.unsafe_load(yaml_in)
            else:
                yaml_object = yaml.safe_load(yaml_in)  # yaml_object will be a list or a dict

            self.from_dict(yaml_object, skip_unsettable=skip_unsettable)

    def _to_yaml(self, yaml_path) -> dict:
        from collections import OrderedDict

        assert yaml_path[-4:] == "yaml"

        kv_dict = self.as_dict()
        yaml_object = OrderedDict()
        for cv in self.class_variables:
            if cv[0] == "_":
                continue
            assert cv in kv_dict, f"class_variables has a key ({cv}) is not found in kv_dict"

            yaml_object[cv] = kv_dict[cv]

        # Order reservation in YAML file# ref: https://stackoverflow.com/questions/45253643/order-preservation-in-yaml-via-python-script
        def ordered_dict_representer(self, value):
            return self.represent_mapping("tag:yaml.org,2002:map", value.items())

        yaml.add_representer(OrderedDict, ordered_dict_representer)
        with open(yaml_path, "w") as yaml_out:
            yaml.dump(yaml_object, yaml_out, indent=4, sort_keys=False)

        return yaml_object

    def __repr__(self) -> str:
        return str(self.as_dict())


class SenConfig(AbstractConfig):
    inp_paths: list[str] = ([],)  # type: ignore list of water distribution network inp path
    sim_output: Literal[
        "pressure",
        "head",
        "demand",
        "flowrate",
        "velocity",
        "headloss",
        "friction_factor",
    ] = "pressure"  # selected simulation output
    ray_temp_path: str = r"D:\tmp\ray"  # where to store ray logging
    temp_path: str = r"D:\tmp\gigantic-dataset\temp"  # temporatory paths for generated inp, rpt, bin files
    output_path: str = r"D:\tmp\gigantic-dataset\outputs"  # output path for simulated output pkl files
    aggr: Literal["mean", "max", "min"] = "mean"  # aggregation operator for nodal values
    batch_size: int = 48  # batch size
    duration: int = 0  # execution time
    init_value_ratio_list: list[float] = [
        0.25,
        5.0,
    ]  # list of multiplied ratios with the baseline value of a single factor. Note that it defines positive side only, the final is both positive and negative ones. # noqa: E501
    num_cpus = 2  # cpu count can be put here
    skip_names: list[str]  # skip node/ link names in collecting metrics
    alter_strategy: Literal["add", "replace"] = "replace"  # alternating strategy
    default_altered_value: float = 0.0  # whenever getting a param is impossible, default value is assigned into the value matrix
    default_failed_value: float = -100.0  # when simulation is failed/got error, default_failed_value is used for that case

    def configure(self) -> None:
        self.add_argument("--temp_path", type=AbstractConfig._str_or_none)
        self.add_argument("--output_path", type=AbstractConfig._str_or_none)


class SenPlotConfig(AbstractConfig):
    zarr_paths: list[str] = []  # zarr path created by analyze_oat_sensitivity
    do_indinvidual_plot: bool = True  # if true, plot oat per water network, otherwise consider all
    do_filter_failed_cases: bool = True  # if True, we filter fails out (for equivalent computation across params)
    do_aggr_pos_neg_values: bool = True  # if True, we take mean of pos, neg pairs
    output_aggregation_strategy: Literal["mask_flat"] = "mask_flat"  # the way to treat outputs w.r.t.  various ratios and invalid cases
    do_shuffle: bool = False  #
    trial_aggr: Literal["mean", "max", "min", None] = None  # aggregate trials
    wdn_aggr: Literal["mean", "max", "min"] = "mean"  # aggregate wdns
    eps: float = 1e-7  # small number
    scatter_size: float = 20
    annot_x: float = 1
    annot_y: float = 1.02
    annot_font_family: str = "Arial"
    annot_font_size: float = 12
    annot_font_color: str = "black"

    def configure(self) -> None:
        self.add_argument("--trial_aggr", type=AbstractConfig._str_or_none)


class DenPlotConfig(AbstractConfig):
    zarr_paths: list[str] = []  # zarr path created
    sim_config_paths: list[str] = []  # sim gen path
    profiler_path: str = r"profiler_report.json"  # profiler path  # type: ignore
    do_indinvidual_plot: bool = False  # if true, plot oat per water network, otherwise consider all
    verbose: bool = True  # debugging
    dmd_mode: Literal["local", "global", "local_norm", "global_norm"] = "global"  # process demand before plotting
    pres_mode: Literal["local", "global", "local_norm", "global_norm"] = "global"  # process demand before plotting

    fill_contour: bool = True  # flag indicates whether contour is filled


@dataclass
class NodeTuneConfig:
    pass


@dataclass
class TankTuneConfig(NodeTuneConfig):
    elevation_strategy: Strategy = "keep"
    elevation_values: Union[tuple, None] = None

    diameter_strategy: Strategy = "keep"
    diameter_values: Union[tuple, None] = None

    # max_level_strategy: Strategy = "keep"
    # max_level_values: Union[tuple, None] = None

    init_level_strategy: Strategy = "keep"
    init_level_values: Union[tuple, None] = None

    # min_level_strategy: Strategy = "keep"
    # min_level_values: Union[tuple, None] = None

    overflow_strategy: Strategy = "keep"
    overflow_values: Union[tuple, None] = None

    vol_curve_name_strategy: Strategy = "keep"
    vol_curve_name_values: Union[tuple, None] = None

    min_vol_strategy: Strategy = "keep"
    min_vol_values: Union[tuple, None] = None


@dataclass
class ReservoirTuneConfig(NodeTuneConfig):
    base_head_strategy: Strategy = "keep"  # elevation of reservoir
    base_head_values: Union[tuple, None] = None

    head_pattern_name_strategy: Strategy = "keep"  # it affects head pattern
    head_pattern_name_values: Union[tuple, None] = None


@dataclass
class JunctionTuneConfig(NodeTuneConfig):
    # required_pressure_strategy: Strategy = "keep"  # can be set by a global setting
    # required_pressure_values: Union[tuple, None] = None

    # minimum_pressure_strategy: Strategy = "keep"  # can be set by a global setting
    # minimum_pressure_values: Union[tuple, None] = None

    # pressure_exponent_strategy: Strategy = "keep"  # can be set by a global setting
    # pressure_exponent_values: Union[tuple, None] = None

    elevation_strategy: Strategy = "keep"
    elevation_values: Union[tuple, None] = None

    base_demand_strategy: Strategy = "keep"
    base_demand_values: Union[tuple, None] = None


@dataclass
class LinkTuneConfig:
    initial_status_strategy: Strategy = "keep"
    initial_status_values: Union[tuple, None] = None


@dataclass
class PipeTuneConfig(LinkTuneConfig):
    diameter_strategy: Strategy = "keep"
    diameter_values: Union[tuple, None] = None

    minor_loss_strategy: Strategy = "keep"
    minor_loss_values: Union[tuple, None] = None

    roughness_strategy: Strategy = "keep"
    roughness_values: Union[tuple, None] = None

    wall_coeff_strategy: Strategy = "keep"
    wall_coeff_values: Union[tuple, None] = None

    length_strategy: Strategy = "keep"
    length_values: Union[tuple, None] = None


@dataclass
class PumpTuneConfig(LinkTuneConfig):
    energy_price_strategy: Strategy = "keep"
    energy_price_values: Union[tuple, None] = None

    energy_pattern_strategy: Strategy = "keep"
    energy_pattern_values: Union[tuple, None] = None

    base_speed_strategy: Strategy = "keep"  # speed_pattern_name_strategy: Literal['sampling','series','keep']='keep'
    base_speed_values: Union[tuple, None] = None

    speed_pattern_name_strategy: Strategy = "keep"
    speed_pattern_name_values: Union[tuple, None] = None

    efficiency_strategy: Strategy = "keep"
    efficiency_values: Union[tuple, None] = None


@dataclass
class PowerPumpConfig(PumpTuneConfig):
    power_strategy: Strategy = "keep"
    power_values: Union[tuple, None] = None


@dataclass
class HeadPumpConfig(PumpTuneConfig):
    pump_curve_name_strategy: Strategy = "keep"
    pump_curve_name_values: Union[tuple, None] = None


@dataclass
class ValveTuneConfig(LinkTuneConfig):
    initial_setting_strategy: Strategy = "keep"
    initial_setting_values: Union[tuple, None] = None


@dataclass
class GPVTuneConfig(ValveTuneConfig):
    headloss_curve_name_strategy: Strategy = "keep"
    headloss_curve_name_values: Union[tuple, None] = None


@dataclass
class PRVTuneConfig(ValveTuneConfig):
    pass


@dataclass
class PSVTuneConfig(ValveTuneConfig):
    pass


@dataclass
class PBVTuneConfig(ValveTuneConfig):
    pass


@dataclass
class FCVTuneConfig(ValveTuneConfig):
    pass


@dataclass
class TCVTuneConfig(ValveTuneConfig):
    pass


@dataclass
class ADGV2Config:
    """Automatic Demand Generator V2"""

    yearly_pattern_num_harmonics: int = 4  # unknown definition. TODO: ask Andres.
    summer_amplitude_range: tuple[int, int] = (
        2,
        3,
    )  # random lower and upper bounds to create a peak during summer
    summer_start: float = 0.4166666666666667  # ratio representing the beginning of the summer month / 12. By default, 0.41 is the beginning of June, its range is [0/12, 11/12] # noqa: E501
    summer_rolling_rate: float = (
        0.2  # exists a chance we shift(roll) the created timeseries. Note that it affects on ALL junction's patterns in a single scene.
    )
    p_commercial: tuple[float, float] = (
        0.25,
        0.35,
    )  # min and max of p_commercial
    profile_household: tuple[int, int, int, int] = (
        0,
        2,
        1,
        0,
    )  # the profile of household nodes. Each number represents low: 0, medium: 1 or high: 2 consumption
    profile_extreme: tuple[int, int, int, int] = (
        2,
        2,
        2,
        2,
    )  # the profile of nodes have extreme demand. Consumption from 00.00-06.00, 06.00-12.00, 12.00-18.00, 18.00-00.00.
    profile_commercial: tuple[int, int, int, int] = (
        2,
        2,
        2,
        1,
    )  # the profile of commercial, industrial nodes.
    noise_range: tuple[float, float] = (0.02, 0.2)  # min and max noise
    zero_dem_rate: float = 0.05  # Zerolize demand rate. Only work if it is greater than zero. This randomly selects (zero_dem_rate * #nodes) nodes and zerolize their demands.# noqa: E501
    extreme_dem_rate: float = 0.02  # prob of available extreme demand junctions. It should be small enough
    max_extreme_dem_junctions: int = 2  # maximum number of extreme junctions.


class SimConfig(AbstractConfig):
    """Compulsory general arguments"""

    inp_paths: list[str] = ([],)  # type: ignore list of water distribution network inp path
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
    ] = ["pressure", "demand"]  # selected simulation outputs
    ray_temp_path: str = r"tmp/ray"  # where to store ray logging
    temp_path: str = r"gigantic_dataset/temp"  # temporatory paths for generated inp, rpt, bin files
    output_path: str = r"gigantic_dataset/outputs"  # output path for simulated output pkl files
    gen_batch_size: int = 100  # batch size for random matrix generation. It should be less than num_samples * backup_times
    batch_size: int = 10  # batch size for computation
    duration: int = 24  # execution time  (hours)
    time_step: int = 1  # time step (hours)
    num_samples: int = 100  # created samples
    backup_times: float = 1  # backup samples = backup_times * num_samples
    num_cpus: int = 10  # define the cpus usage. Set num_cpus to 1 will turn off Ray
    fractional_cpu_usage: float = 1.0  # how much fractional usage per cpu. Only for Ray
    mem_per_worker: float = 5  # Memory in GB. By default, each worker requires 5GB. Only for Ray
    yield_worker_generator: bool = False  # if true, worker yields a generator of smaller result tuples. Otherwise, a worker yields a result tuple
    time_consistency: bool = True  # if true, gathering only simulated outputs whose duration == config.duration
    save_success_inp: bool = True  # if True, save successful input file. Note: wntr has bugs in exporting input file.
    verbose: bool = True  # if True, print debug info
    skip_names: list[str] = []  # nodes/ links in this list will be skipped in checking criteria
    pressure_range: list[float] = [-1e-4, 151]  # generated nodes (except from skip_names) must be in range of (min,max)
    tank_tune: TankTuneConfig = TankTuneConfig()  # tank's parameters tune settings
    reservoir_tune: ReservoirTuneConfig = ReservoirTuneConfig()  # reservoir's parameters tune settings
    junction_tune: JunctionTuneConfig = JunctionTuneConfig()  # junction's parameters tune settings
    pipe_tune: PipeTuneConfig = PipeTuneConfig()  # pipe's parameters tune settings
    power_pump_tune: PowerPumpConfig = PowerPumpConfig()  # power pump's parameters tune settings
    head_pump_tune: HeadPumpConfig = HeadPumpConfig()  # head pump's parameters tune settings
    prv_tune: PRVTuneConfig = PRVTuneConfig()  # prv's parameters tune settings
    psv_tune: PSVTuneConfig = PSVTuneConfig()  # psv's parameters tune settings
    pbv_tune: PBVTuneConfig = PBVTuneConfig()  # pbv's parameters tune settings
    fcv_tune: FCVTuneConfig = FCVTuneConfig()  # fcv's parameters tune settings
    tcv_tune: TCVTuneConfig = TCVTuneConfig()  # tcv's parameters tune settings
    gpv_tune: GPVTuneConfig = GPVTuneConfig()  # gpv's parameters tune settings
    adgv2_config: ADGV2Config = ADGV2Config()  # only for adg v2 cofig settings

    def _to_dict(self) -> dict:
        records = self.as_dict()
        parsed_records = {}
        for k, v in records.items():
            if isinstance(v, (LinkTuneConfig, NodeTuneConfig)):
                tmp_dict = asdict(v)  # type:ignore
                parsed_records[k] = tmp_dict
            else:
                parsed_records[k] = v
        return parsed_records


TuneConfig = Union[
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
]


class GidaConfig(AbstractConfig):
    zip_file_paths: list[str] = []  # .zip or .zarr to the dataset
    node_attrs: list[Union[str, tuple]]  # node attrs
    edge_attrs: list[Union[str, tuple]] = []  # edge attrs
    label_attrs: list[Union[str, tuple]] = []  # label atttrs. List can contain either node or edge but elements must be consistent
    edge_label_attrs: list[str] = []  # Only need when the list label_attrs is used (as node), edge_label_attrs is list of label at edge
    input_paths: list[str] = []  # unnecesary since we can call required info from zarr
    num_records: Optional[int] = None  # number of records. None means taking all
    verbose: bool = False  # flag allowing more debug info. Default is False
    split_type: Literal["temporal", "scene"] = "scene"  # horizontal cut for scene choice, and temporal cut for temporal.
    split_set: Literal["train", "val", "test", "all"] = "all"  # following the splitting ratio 60:20:20
    skip_nodes_list: list[list[str]] = []  # node names want to be skipped. By default, it includes the skip_nodes in config
    skip_types_list: list[list[str]] = []  # sugar-coating removal by types (node, link, junction, tank ...). Check the component in zarr.
    unstackable_pad_value: Any = 0.0  # if component has a missing value, we pad this with unstackable_pad_value
    bypass_skip_names_in_config: bool = False  # if True, we bypass even skip_nodes in config.
    do_lazy: bool = False  # if True, gida delays the dataset reading. Useful when we yield a generator to read chunks.
    subset_shuffle: bool = True  # given train/val/test ids, flag indicates whether shuffle this list once before taking the corresponding subset. For temporal split_type, it should be False.  # noqa: E501
    """########################### NEW FEATURES "###########################"""
    dataset_log_pt_path: str = ""  # if non empty, we load/save subset_shuffle and statistic into this pt file.
    batch_axis_choice: Literal["temporal", "scene", "snapshot"] = "scene"  # which axis is set as batch dimension.
    do_cache: bool = False  # Flag indicates whether we cache array after first loading. Very fast but OOM can happen.
    split_per_network: bool = True  # If True, foreach network, we split train, valid, test individually (Useful for multiple network joint-training). Otherwise, we concatenate all networks into a single to-be-splitted array # noqa: E501
    wdn_names: list[str] = []  # support hugging face. This list should be non-empty (v7) or `zip_file_paths` should be non-empty(v6).
    indexing: Literal["static", "dynamic"] = "static"  # Two ways to generate indices, dynamic is faster. Default is static.
    """"########################### DEPRECATED SOON "###########################"""
    time_sampling_rate: int = 1  # perform sampling on time dim(V5 only)
    overwatch: bool = False  # Turn on to capture memory snapshots at some defined phases. (V5 only)
    chunk_limit: str = "120 GB"  # depends on your available RAM. (V5 only)
    selected_snapshots: Optional[int] = None  # number of selected snapshots. None is taking all. (V5 only)

    @staticmethod
    def elelist2tuple(attrs: list[str | list[str]]) -> list[str | tuple]:
        new_attrs = []
        for attr in attrs:
            if isinstance(attr, str):
                new_attrs.append(attr)
            else:
                new_attrs.append(tuple(attr))
        return new_attrs

    def configure(self) -> None:
        self.add_argument("--node_attrs", type=GidaConfig.elelist2tuple)


class GidaNSFConfig(GidaConfig):
    context_length: int = 12  # non-overlapped windows


class ModelConfig(AbstractConfig):
    name: str = "gcn"  # name of the model
    num_layers: int = 2  # number of layers
    nc: int = 16  # number of hidden nc
    act: Literal["relu", "gelu"] = "relu"  # activation
    has_final_linear: bool = False  # if true, add a linear following the gnn layers
    weight_path: str = ""  # path storing the model weights.  If empty, we use a new model
    # do_load: bool = False  # load weights of the model


class TrainConfig(AbstractConfig):
    # do_load: bool = False  # flag represents temporarily global setting for all mdoels
    model_configs: list[ModelConfig] = []  # involving DL models in this train
    lr: float = 0.01  # 0.01 #Learning rate. Default is 0.0005
    weight_decay: float = 5e-4  # weight decay. Default is 0.000006
    epochs: int = 10  # 0 #number of epochs to train the model
    mask_rate: float = 0.95  # masking ratio. Default is 0.95
    criterion: Literal["mse", "mae", "sce", "ce"] = "mse"  # criterion loss. Support mse|sce|mae
    batch_size: int = 64  # batch size
    use_data_batch: bool = False  # pass pyg data batch as parameter into model. Set False to fasten training. Default is False
    device: str = "cuda"  # Training device. If gpu is unavailable, device is set to cpu.
    norm_type: Literal["znorm", "minmax", "unused"] = "unused"  # normalization type. Support znorm| minmax|unused"
    task: Literal["supervised", "semi"] = "semi"  # current supporting task
    """###########################TRACKING EXPERIMENTS SETTINGS################################"""
    log_method: str = ""  # log method! Support wandb or ''
    # log_gradient: bool = False #flag indicates keeping track of gradient flow
    project_name: str = "dev-pretext-train"  # name of tracking project"
    save_path: str = ""  # Path to store model weights. If empty, we create a unique name
    log_per_epoch: int = 1  # log every log_per_epoch
    run_prefix: str = ""  # it helps naming the run on WANDB
    """#########################################################################################"""
    num_cpus: int = 10  # for data loader
