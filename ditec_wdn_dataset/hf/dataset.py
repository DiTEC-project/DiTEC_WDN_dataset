import json
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from collections import OrderedDict, defaultdict
import copy
from itertools import compress
import tempfile
import os
from typing import Any, Literal, Optional, Sequence, Union
import torch
from torch import Tensor, from_numpy, save, load
from ditec_wdn_dataset.utils.auxil_v8 import (
    get_object_name_list_by_component,
    is_node_simulation_output,
    shuffle_list,
    masking_list,
    get_all_simulation_output_parameters,
)
from torch_geometric.data.data import BaseData
from torch_geometric.data import Dataset, Data, Batch

from wntr.network import WaterNetworkModel

import dask.array.core as dac
from dask.base import compute

from huggingface_hub import list_repo_files

import datasets as hfd

LARGE_NUMBER = 10000
HUGGING_FACE_REPO = "rugds/ditec-wdn"

AVAIL_WDN_NAMES: list[str] = [
    "Jilin_7GB_1Y",
    "new-york_6GB_1Y",
    "EXN_2GB_24H",
    "hanoi_8GB_1Y",
    "ky5_1GB_24H",
    "foss-poly-1_10GB_1Y",
    "large_3GB_24H",
    "KL_232GB_1Y",
    "19PipeSystem_4GB_1Y",
    "ky16_1GB_24H",
    "EPANET-Net-3_23GB_1Y",
    "balerma_101GB_1Y",
    "ky13_1GB_24H",
    "Anytown_7GB_1Y",
    "ky4_239GB_1Y",
    "ky7_122GB_1Y",
    "ky24_52GB_1Y",
    "RuralNetwork_93GB_1Y",
    "d-town_1GB_24H",
    "ky14_1GB_24H",
    "L-TOWN_155GB_1Y",
    "WA1_27GB_1Y",
    "modena_64GB_1Y",
    "ky1_199GB_1Y",
    "OBCL-1_60GB_1Y",
    "ky18_190GB_1Y",
    "FOWM_10GB_1Y",
    "FFCL-1_18GB_1Y",
    "ky10_1GB_24H",
    "NPCL-1_72GB_1Y",
    "ky3_69GB_1Y",
    "ZJ_31GB_1Y",
    "ky2_217GB_1Y",
    "ky8_2GB_24H",
    "CTOWN_1GB_24H",
    "ky6_103GB_1Y",
]
IndexType = Union[slice, Tensor, np.ndarray, Sequence]


def get_metadata(parquet_path: str, path: str = "rugds/ditec-wdn", key: str = "attrs", cache_dir: str = "data") -> dict:
    with tempfile.TemporaryDirectory(dir=cache_dir, ignore_cleanup_errors=True) as dirpath:
        # first load dataset
        parquet_path = hf_hub_download(repo_id=path, repo_type="dataset", filename=parquet_path, local_dir=dirpath)

        metadata = pq.read_metadata(parquet_path)
        binary_meta_dict = metadata.metadata

        bkey = key.encode()
        if bkey in binary_meta_dict:
            attrs = binary_meta_dict[bkey].decode()

            attrs = json.loads(attrs)
            return attrs
        else:
            print("Error! Key not found!")
            return {}


def parse2data(x: np.ndarray, edge_index: np.ndarray, edge_attr: Optional[np.ndarray], y: Optional[np.ndarray], edge_y: Optional[np.ndarray]) -> Data:
    data_dict: dict[str, Any] = defaultdict(list)
    data_dict["edge_index"] = torch.as_tensor(edge_index, dtype=torch.long)
    data_dict["x"] = torch.as_tensor(x, dtype=torch.float)
    if edge_attr is not None:
        data_dict["edge_attr"] = torch.as_tensor(edge_attr, dtype=torch.float)
    if y is not None:
        data_dict["y"] = torch.as_tensor(y, dtype=torch.float)
    if edge_y is not None:
        data_dict["edge_y"] = torch.as_tensor(edge_y, dtype=torch.float)

    data = Data.from_dict(data_dict)
    return data


class GidaV7(Dataset):
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
        "node_pressure",
        "node_head",
        "node_demand",
        "link_flowrate",
        "link_velocity",
        "link_headloss",
        "link_friction_factor",
    ]
    # sugar-coating popular merge-able params (order insensitive)
    Node_Elevation = ("reservoir_base_head", "junction_elevation", "tank_elevation")
    Link_Initial_Status = (
        "pipe_initial_status",
        "power_pump_initial_status",
        "headpump_initial_status",
        "prv_initial_status",
        "tcv_initial_status",
    )  # some valves are missing as we haven't had them yet
    Pump_Curve = ("headpump_pump_curve_name", "powerpump_pump_curve_name")

    class Root:
        def _get_component(self, attr: str) -> str:
            has_component = "_" in attr
            component = attr.split("_")[0] if has_component else "node" if is_node_simulation_output(attr) else "link"
            return component

        def _get_datafiles_and_attrs(self, wdn_folder: str) -> tuple[dict[str, str], dict[str, Any]]:
            lightweight_parquet_path = f"{wdn_folder}/junction_elevation-0-static_input.parquet"
            attrs = get_metadata(lightweight_parquet_path, cache_dir=self.cache_dir)
            datafiles = {}
            for ele, com_list in attrs["okeys"].items():
                for com in com_list:
                    ele_com = f"{ele}_{com}"
                    datafiles[ele_com] = f"{wdn_folder}/{ele_com}-*.parquet"
            for com in get_all_simulation_output_parameters():
                ele = self._get_component(com)
                ele_com = f"{ele}_{com}"
                datafiles[ele_com] = f"{wdn_folder}/{ele_com}-*.parquet"
            return datafiles, attrs

        def download_from_hf_backup(self):
            uni_list = set(self.sorted_node_attrs).union(self.sorted_edge_attrs).union(self.sorted_label_attrs).union(self.sorted_edge_label_attrs)

            com_attrs_dict = defaultdict(list)

            for attr in uni_list:
                if isinstance(attr, str):
                    com = self._get_component(attr)
                    if attr in self.data_files_template:
                        com_attrs_dict[com].append(attr)

                elif isinstance(attr, tuple):
                    for subattr in attr:
                        com = self._get_component(subattr)
                        if subattr in self.data_files_template:
                            com_attrs_dict[com].append(subattr)

            ds_dict = {}

            for com, attrs in com_attrs_dict.items():
                refined_data_files = {k: self.data_files_template[k] for k in attrs if k in self.data_files_template}
                ds: hfd.DatasetDict = hfd.load_dataset(path=HUGGING_FACE_REPO, data_files=refined_data_files, cache_dir=self.cache_dir)  # type: ignore

                selected_cols = [col for col in ds.column_names[attrs[0]] if col not in ["scenario_id", "time_id", "curve_id"]]
                if len(selected_cols) > 0:
                    ds = ds.select_columns(selected_cols)
                    for f in refined_data_files:
                        ds_dict[f] = ds[f]
                else:
                    del ds
            ds = hfd.DatasetDict(ds_dict)
            self.root = ds.with_format("numpy")

        def download_from_hf(self):
            uni_list = set(self.sorted_node_attrs).union(self.sorted_edge_attrs).union(self.sorted_label_attrs).union(self.sorted_edge_label_attrs)

            com_attrs_dict = defaultdict(list)

            for attr in uni_list:
                if isinstance(attr, str):
                    com = self._get_component(attr)
                    if attr in self.data_files_template:
                        com_attrs_dict[com].append(attr)

                elif isinstance(attr, tuple):
                    for subattr in attr:
                        com = self._get_component(subattr)
                        if subattr in self.data_files_template:
                            com_attrs_dict[com].append(subattr)

            ds_dict = {}

            for com, attrs in com_attrs_dict.items():
                refined_data_files = {k: self.data_files_template[k] for k in attrs if k in self.data_files_template}
                ds: hfd.DatasetDict = hfd.load_dataset(path=HUGGING_FACE_REPO, data_files=refined_data_files, cache_dir=self.cache_dir)  # type: ignore

                # selected_cols = [col for col in ds.column_names[attrs[0]] if col not in ["scenario_id", "time_id", "curve_id"]]
                # if len(selected_cols) > 0:
                #     ds = ds.select_columns(selected_cols)
                #     for f in refined_data_files:
                #         ds_dict[f] = ds[f]
                # else:
                #     del ds
                remove_cols = list(set(ds.column_names[attrs[0]]).intersection(["scenario_id", "time_id", "curve_id"]))
                if len(remove_cols) > 0:
                    ds = ds.remove_columns(remove_cols)
                for f in refined_data_files:
                    ds_dict[f] = ds[f]

            ds = hfd.DatasetDict(ds_dict)
            self.root = ds.with_format("numpy")

        def __init__(self, name: str, num_cpus: Optional[int] = None, cache_dir: str = "data") -> None:
            self.name = name

            # load HF
            # data_files_template, lightweight_parquet_path = self._get_template(wdn_folder=name)
            # attrs = get_metadata(lightweight_parquet_path)
            self.cache_dir = cache_dir
            self.data_files_template, self.attrs = self._get_datafiles_and_attrs(wdn_folder=name)

            self.array_keys: list[str] = list(self.data_files_template.keys())

            self.time_dim: int = self.attrs["duration"] // self.attrs["time_step"]
            self.node_mask: Optional[np.ndarray] = None
            self.edge_mask: Optional[np.ndarray] = None
            self.adj_mask: Optional[np.ndarray] = None
            self.edge_index: Optional[np.ndarray] = None
            self.node_names: list[str] = []
            self.edge_names: list[str] = []
            self.num_cpus: Optional[int] = num_cpus

            self.node_components: list[str] = []
            self.edge_components: list[str] = []

            self.sorted_node_attrs: list[str | tuple] = []
            self.node_has_asterisks: list[bool] = []
            self.sorted_edge_attrs: list[str | tuple] = []
            self.edge_has_asterisks: list[bool] = []
            self.sorted_label_attrs: list[str | tuple] = []
            self.label_has_asterisks: list[bool] = []
            self.sorted_edge_label_attrs: list[str | tuple] = []
            self.edge_label_has_asterisks: list[bool] = []

        def get_array_by_key(self, key: str) -> hfd.Dataset:
            if key in self.root:
                return self.root[key]  # type:ignore
            else:
                com = self._get_component(key)
                actual_key = f"{com}/{key}"
                assert actual_key in self.root
                return self.root[actual_key]

        def compute_first_size(self) -> int:
            if len(self.sorted_node_attrs) <= 0:
                return 0
            num_samples = self.root.num_rows
            if isinstance(num_samples, dict):
                total_samples = sum(num_samples.values())
                return total_samples
            else:
                return num_samples

    def __init__(
        self,
        wdn_names: list[str],
        node_attrs: list[str | tuple],
        edge_attrs: list[str | tuple] = [],
        label_attrs: list[str | tuple] = [],
        edge_label_attrs: list[str | tuple] = [],
        input_paths: list[str] = [],
        num_records: Optional[int] = None,
        selected_snapshots: Optional[int] = None,
        verbose: bool = True,
        split_type: Literal["temporal", "scene"] = "scene",
        split_set: Literal["train", "val", "test", "all"] = "all",
        split_fixed: bool = False,
        split_per_network: bool = True,
        skip_nodes_list: list[list[str]] = [],
        skip_types_list: list[list[str]] = [],
        unstackable_pad_value: Any = 0,
        bypass_skip_names_in_config: bool = False,
        chunk_limit: str = "120 GB",
        overwatch: bool = False,
        batch_axis_choice: Literal["scene", "temporal", "snapshot"] = "scene",
        num_cpus: int = 1,
        time_sampling_rate: int = 1,
        do_cache: bool = False,
        subset_shuffle: bool = False,
        dataset_log_pt_path: str = r"",
        **kwargs,
    ) -> None:
        """Correponding to configs.py/GiDaConfig.

        Args:
            wdn_names (list[str]): name of wdn folders. this should match ones on HF
            node_attrs (list[str  |  tuple]): node attributes. if str, we query directly. Otherwise, we hstack attributes. Support (*) operator.
            edge_attrs (list[str  |  tuple], optional): edge attributes. Support (*) operator. Defaults to [].
            label_attrs (list[str  |  tuple], optional): label attributes. Support (*) operator. Defaults to [].
            edge_label_attrs (list[str  |  tuple], optional): edge label (only used when label is non-empty). Support (*) operator. Defaults to [].
            input_paths (list[str], optional): To query networks' basis info. Defaults to [].
            num_records (Optional[int], optional): Set None to use all. Defaults to None.
            selected_snapshots (Optional[int], optional): Deprecated. Defaults to None.
            verbose (bool, optional): flag indicates whether logging debug info. Defaults to True.
            split_type (Literal[&quot;temporal&quot;, &quot;scene&quot;], optional): Deprecated. Defaults to "scene".
            split_set (Literal[&quot;train&quot;, &quot;val&quot;, &quot;test&quot;, &quot;all&quot;], optional): to split subsets. Defaults to "all".
            split_fixed (bool, optional): flag indicates whether the cutting positions are pre-defined (fixed) or dynamic
            split_per_network (bool, optional): If True, foreach network, we split train, valid, test individually (Useful for multiple network joint-training). Otherwise, we concatenate all networks into a single to-be-splitted array
            skip_nodes_list (list[list[str]], optional): add extra skipped node names, otherwise, load skip_nodes in zarr zip files. Defaults to [].
            skip_types_list (list[list[str]], optional): massively skip by node types (juctions, reservoir, tank). Defaults to [].
            unstackable_pad_value (Any, optional): pad array by this value, when array is unmatched/unstacked. Defaults to 0.
            bypass_skip_names_in_config (bool, optional): Whether bypass default skipped node names in zarrs. Mutual inclusive w skip_nodes_list, skip_types_list. Defaults to False.
            chunk_limit (str, optional): Deprecated. Defaults to "120 GB".
            overwatch (bool, optional): Deprecated. Defaults to False.
            batch_axis_choice (Literal[&quot;scene&quot;, &quot;temporal&quot;, &quot;snapshot&quot;], optional): split subset by axis. Defaults to "scene".
            num_cpus (int, optional): Support multi-cpu data loading w the number of cpus (nightly). Defaults to 1.
            time_sampling_rate (int, optional): Deprecated. Defaults to 1.
            do_cache (bool, optional): flag indicates dump everything in mem. Very fast but OOM can happen. Defaults to False.
            subset_shuffle (bool, optional): flag indicates whether subset is shuffled. After splitting WHOLE TRAIN/VAL/TEST sets, we perform subset shuffling per set. Defaults to True.
            dataset_log_pt_path (str, optional): .pt path where storing subset indices and statistic (when you call `gather_statistic`). If `dataset_log_pt_path=""`, we WON'T LOAD SHUFFLE IDs.  Defaults to ''.

        """  # noqa: E501
        self.input_paths = input_paths
        self.wdn_names = wdn_names
        self._is_init_root = False
        self.node_attrs = [att if isinstance(att, str) else tuple(att) for att in node_attrs] if node_attrs else []
        self.edge_attrs = [att if isinstance(att, str) else tuple(att) for att in edge_attrs] if edge_attrs else []
        self.label_attrs = [att if isinstance(att, str) else tuple(att) for att in label_attrs] if label_attrs else []
        self.edge_label_attrs = [att if isinstance(att, str) else tuple(att) for att in edge_label_attrs] if edge_label_attrs else []

        if "headpump_base_speed" in self.edge_attrs or "headpump_base_speed" in self.label_attrs or "headpump_base_speed" in self.edge_label_attrs:
            print("WARN! headpump_base_speed attribute is under validation phase!")

        self.verbose = verbose
        self.num_records = num_records
        self.selected_snapshots = selected_snapshots
        self.split_set = split_set
        self.skip_nodes_list = skip_nodes_list
        self.skip_types_list = skip_types_list
        self.split_type: Literal["temporal", "scene"] = split_type
        self.split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)
        if split_fixed:
            print("WARN! GidaV7 requires split_fixed set to False!")
        self.split_fixed = False
        self.split_per_network = split_per_network
        # self._arrays: list[tuple[zarr.Array, zarr.Array, zarr.Array | None, zarr.Array | None, zarr.Array | None]] = []
        self._index_map: dict[int, tuple[int | None, int | None]] = {}
        self._network_map: dict[int, int] = {}
        self._roots: list[GidaV7.Root] = []
        self.bypass_skip_names_in_config = bypass_skip_names_in_config
        self.unstackable_pad_value = unstackable_pad_value
        self.chunk_limit = chunk_limit
        self.batch_axis_choice: Literal["scene", "temporal", "snapshot"] = batch_axis_choice
        self.overwatch = overwatch
        self.num_cpus = num_cpus
        self.time_sampling_rate = time_sampling_rate
        self.do_cache = do_cache
        self.subset_shuffle = subset_shuffle
        self.train_shuffle_ids, self.val_shuffle_ids, self.test_shuffle_ids = [], [], []
        self.dataset_log_pt_path = dataset_log_pt_path

        self.cache_dir = "data"
        os.makedirs("data", exist_ok=True)
        # allow user customize function here
        self.custom_process()

        transform = kwargs.get("transform", None)
        super().__init__(None, transform=transform)

        # subset shuffle load/write depends on `subset_shuffle` and `dataset_log_pt_path`
        # self.process_subset_shuffle()
        if self._indices is None:
            self.update_indices()

    def update_indices(self) -> None:
        # update _indices
        attempted_ids = self.train_ids + self.val_ids + self.test_ids
        if len(attempted_ids) > 0:
            self._indices = attempted_ids
        else:
            assert self._index_map is not None and len(self._index_map) > 0
            self._indices = list(self._index_map.keys())

    def _get_num_samples(self) -> int:
        if self.split_fixed:
            return self.num_records if self.num_records is not None else self.length
        else:
            return self.length

    def custom_process(self) -> None:
        # load arrays from zip file (and input_paths)
        self.length, self._index_map, self._network_map, self._num_samples_per_network_list = self.compute_indices(wdn_names=self.wdn_names)

        self.train_ids, self.val_ids, self.test_ids = self.compute_subset_ids_by_ratio(self.split_ratios, num_samples=self._get_num_samples())

    def process_subset_shuffle(self, custom_subset_shuffle_pt_path: str = "", create_and_save_to_dataset_log_if_nonexist: bool = True):
        if self.subset_shuffle:
            if custom_subset_shuffle_pt_path != "":
                self._load_shuffle_indices_from_disk_internal(path=custom_subset_shuffle_pt_path, sanity_check=True)
            else:
                # we first check whether the pt file exists. If yes, we load train/val/test ids
                if not self.load_shuffle_indices_from_disk(sanity_check=True):
                    # otherwise, we perform shuffle and store ids in a saving folder
                    new_train_ids, self.train_shuffle_ids = shuffle_list(self.train_ids)
                    new_val_ids, self.val_shuffle_ids = shuffle_list(self.val_ids)
                    new_test_ids, self.test_shuffle_ids = shuffle_list(self.test_ids)
                    if self.dataset_log_pt_path != "" and create_and_save_to_dataset_log_if_nonexist:
                        self.save_shuffle_indices_to_disk()
                    else:
                        print(
                            "WARN! Subset shuffle indices cannot be saved as `dataset_log_pt_path` is empty or `do_save_to_dataset_log` is set to False in Gida Interface! You cannot re-load these ids in the inference or next train!",  # noqa: E501
                            flush=True,
                        )

                    self.train_ids = new_train_ids
                    self.val_ids = new_val_ids
                    self.test_ids = new_test_ids
            self.update_indices()

    def save_shuffle_indices_to_disk(self) -> None:
        assert self.dataset_log_pt_path != "", "ERROR! dataset_log_pt_path should not be empty!"
        assert self.dataset_log_pt_path[-4:] == ".pth" or self.dataset_log_pt_path[-3:] == ".pt", (
            f"ERROR! dataset_log_pt_path is invalid. Get self.dataset_log_pt_path = ({self.dataset_log_pt_path})"
        )
        my_dict = {
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
            "train_shuffle_ids": self.train_shuffle_ids,
            "val_shuffle_ids": self.val_shuffle_ids,
            "test_shuffle_ids": self.test_shuffle_ids,
        }

        if os.path.exists(self.dataset_log_pt_path):
            # prevent overriden
            cp_dict: dict = load(self.dataset_log_pt_path)
            cp_dict.update(my_dict)
            save(cp_dict, self.dataset_log_pt_path)
        else:
            save(my_dict, self.dataset_log_pt_path)

    def _load_shuffle_indices_from_disk_internal(self, path: str, sanity_check: bool = True) -> bool:
        if os.path.isfile(path) and (path[-4:] == ".pth" or path[-3:] == ".pt"):
            my_dict = load(path)

            if sanity_check:
                try:
                    assert set(my_dict["train_ids"]) == set(self.train_ids)
                    assert set(my_dict["val_ids"]) == set(self.val_ids)
                    assert set(my_dict["test_ids"]) == set(self.test_ids)
                except Exception:  # Include assertionerror and keyerror :)
                    return False
            if "train_shuffle_ids" in my_dict:
                self.train_ids = masking_list(self.train_ids, my_dict["train_shuffle_ids"])
            if "val_shuffle_ids" in my_dict:
                self.val_ids = masking_list(self.val_ids, my_dict["val_shuffle_ids"])
            if "test_shuffle_ids" in my_dict:
                self.test_ids = masking_list(self.test_ids, my_dict["test_shuffle_ids"])
            return True
        return False

    def load_shuffle_indices_from_disk(self, sanity_check: bool = True) -> bool:
        return self._load_shuffle_indices_from_disk_internal(path=self.dataset_log_pt_path, sanity_check=sanity_check)

    def compute_subset_ids_by_ratio(self, split_ratios: tuple[float, float, float], num_samples: int) -> tuple[list[int], list[int], list[int]]:
        train_ids, val_ids, test_ids = [], [], []
        len_of_list = len(self._num_samples_per_network_list)
        # if not split per network or existing a single network only, we split based on flatten ids
        if not self.split_per_network or len_of_list == 1:
            left = int(self.length * split_ratios[0])
            right = int(left + self.length * split_ratios[1])

            flatten_ids = np.asarray(list(self._index_map.keys()))

            flatten_ids = flatten_ids.tolist()
            train_ids = flatten_ids[:left]
            val_ids = flatten_ids[left:right]
            test_ids = flatten_ids[right:]
            if self.split_fixed:
                selected_trains = int(num_samples * split_ratios[0])
                selected_vals = int(num_samples * split_ratios[1])
                selected_test = num_samples - selected_trains - selected_vals

                train_ids = train_ids[:selected_trains]
                val_ids = val_ids[:selected_vals]
                test_ids = test_ids[:selected_test]
        else:
            # to split per network, we compute train/val/test individually
            # degree of freedom will be (len_of_list - 1)
            expected_train_samples = int(self.length * split_ratios[0])
            expected_valid_samples = int(self.length * split_ratios[1])
            expected_test_samples = self.length - expected_train_samples - expected_valid_samples
            flatten_ids = np.asarray(list(self._index_map.keys()))

            num_samples_per_network = num_samples // len(self._num_samples_per_network_list)
            current_nid = 0
            for i, network_num_samples in enumerate(self._num_samples_per_network_list):
                network_flatten_ids = flatten_ids[current_nid : current_nid + network_num_samples]

                if self.batch_axis_choice == "snapshot":
                    # with snapshots, we still split by scence to ensure the scenario independence
                    # f_0-> (n_0, t_0), f_1 -> (n_0, t_1), ..., f_T -> (n_0, t_T), f_T+1 -> (n_1, t_0), ...
                    time_dim = self._roots[i].time_dim
                    num_scenes = len(network_flatten_ids) // time_dim
                    left = int(num_scenes * split_ratios[0])
                    right = int(left + num_scenes * split_ratios[1])

                    left = left * time_dim
                    right = right * time_dim

                    network_train_ids = network_flatten_ids[:left]
                    network_val_ids = network_flatten_ids[left:right]
                    network_test_ids = network_flatten_ids[right:]

                else:
                    left = int(network_num_samples * split_ratios[0])
                    right = int(left + network_num_samples * split_ratios[1])
                    network_train_ids = network_flatten_ids[:left]
                    network_val_ids = network_flatten_ids[left:right]
                    network_test_ids = network_flatten_ids[right:]

                if i == len_of_list - 1:
                    network_train_ids = network_train_ids[: expected_train_samples - len(train_ids)]
                    network_val_ids = network_val_ids[: expected_valid_samples - len(val_ids)]
                    network_test_ids = network_test_ids[: expected_test_samples - len(test_ids)]

                if self.split_fixed:
                    selected_trains = int(num_samples_per_network * split_ratios[0])
                    selected_vals = int(num_samples_per_network * split_ratios[1])
                    selected_test = num_samples_per_network - selected_trains - selected_vals
                    network_train_ids = network_train_ids[:selected_trains]
                    network_val_ids = network_val_ids[:selected_vals]
                    network_test_ids = network_test_ids[:selected_test]

                train_ids.extend(network_train_ids.tolist())
                val_ids.extend(network_val_ids.tolist())
                test_ids.extend(network_test_ids.tolist())
                current_nid += network_num_samples

        return train_ids, val_ids, test_ids

    def compute_indices(self, wdn_names: list[str]) -> tuple[int, dict[int, tuple[int | None, int | None]], dict[int, int], list[int]]:
        # this is must-have since the size of networks is different.
        index_map: dict[int, tuple[int | None, int | None]] = {}
        network_map: dict[int, int] = {}
        num_samples_per_network_list: list[int] = []
        flatten_index = 0
        self.load_roots(wdn_names)

        root_sizes: list[int] = [r.compute_first_size() for r in self._roots]
        if self.batch_axis_choice == "scene":
            sum_of_root_sizes = sum(root_sizes)
        elif self.batch_axis_choice == "temporal":
            sum_of_root_sizes = sum([r.time_dim for r in self._roots])
        else:  # snapshot
            sum_of_root_sizes = sum([r.time_dim * root_sizes[i] for i, r in enumerate(self._roots)])

        total_num_samples: int = sum_of_root_sizes  # if self.num_records is None else min(sum_of_root_sizes, self.num_records)
        num_samples_per_network = total_num_samples // len(self._roots)

        for network_index, root in enumerate(self._roots):
            if self.batch_axis_choice == "scene":
                # arr WILL have shape <merged>(#scenes, #nodes_or_#links, #statics + time_dims * #dynamics)
                if self.split_fixed:
                    relative_scene_ids = np.arange(root_sizes[network_index])
                    relative_scene_ids = relative_scene_ids[:num_samples_per_network]
                    num_samples = len(relative_scene_ids)  # min(num_samples_per_network, root_sizes[network_index])
                else:
                    num_samples = root.compute_first_size() if self.num_records is None else min(self.num_records, root.compute_first_size())
                    relative_scene_ids = np.arange(num_samples)
                tuples = (relative_scene_ids, None)
            elif self.batch_axis_choice == "temporal":
                if self.split_fixed:
                    relative_time_ids = np.arange(root.time_dim)
                    relative_time_ids = relative_time_ids[:num_samples_per_network]
                    num_samples = len(relative_time_ids)
                else:
                    num_samples = root.time_dim
                    relative_time_ids = np.arange(num_samples)
                tuples = (None, relative_time_ids)

            elif self.batch_axis_choice == "snapshot":
                if self.split_fixed:
                    time_dim = root.time_dim
                    num_scenes = root_sizes[network_index]
                    relative_scene_ids = np.arange(num_scenes).repeat(time_dim)  # .reshape([-1, 1])
                    relative_time_ids = np.tile(np.arange(time_dim), reps=num_scenes)  # .reshape([-1, 1])
                    relative_scene_ids = relative_scene_ids[:num_samples_per_network]
                    relative_time_ids = relative_time_ids[:num_samples_per_network]
                    num_samples = len(relative_scene_ids)
                else:
                    num_scenes = root.compute_first_size() if self.num_records is None else min(self.num_records, root.compute_first_size())
                    time_dim = root.time_dim
                    relative_scene_ids = np.arange(num_scenes).repeat(time_dim)  # .reshape([-1, 1])
                    relative_time_ids = np.tile(np.arange(time_dim), reps=num_scenes)  # .reshape([-1, 1])
                    tuples = (relative_scene_ids, relative_time_ids)
                    num_samples = len(relative_scene_ids)
                tuples = (relative_scene_ids, relative_time_ids)
            else:
                raise NotImplementedError
            extended_network_ids = np.full([num_samples], network_index)
            flatten_ids = np.arange(flatten_index, flatten_index + num_samples)

            network_index_map: dict[int, tuple[int | None, int | None]] = {}
            # fid_nid_map: dict[int, int] = {}
            lefts: np.ndarray | None = tuples[0] if tuples[0] is not None else None
            rights: np.ndarray | None = tuples[1] if tuples[1] is not None else None
            for i, fid in enumerate(flatten_ids):
                left = lefts[i] if lefts is not None else None
                right = rights[i] if rights is not None else None
                network_index_map[fid] = (left, right)
                # fid_nid_map[fid] = network_index
            # update the global map
            index_map.update(network_index_map)
            network_map.update(zip(flatten_ids, extended_network_ids))  # network_map.update(fid_nid_map)  #
            # update flatten index indicator and network index
            flatten_index += num_samples
            num_samples_per_network_list.append(num_samples)

        length = flatten_index
        return length, index_map, network_map, num_samples_per_network_list

    def is_node(self, component: str) -> bool:
        return component in ["junction", "tank", "reservoir", "node"]

    def get_component(self, attr: str) -> str:
        has_component = "_" in attr
        component = attr.split("_")[0] if has_component else "node" if is_node_simulation_output(attr) else "link"
        return component

    def get_component_by_attr(self, attr: str | tuple, axis=0) -> str:
        """if attr is str, return component| if attr is tuple, reuturn component of the axis-th item in the tuple"""

        if isinstance(attr, str):
            return self.get_component(attr)
        elif isinstance(attr, tuple):
            return self.get_component(attr[axis])
        else:
            raise TypeError(f"attr {attr} has type {type(attr)} which is illegal.")

    def get_available_node_components(self, okeys: OrderedDict, root_components: list[str]) -> list[str]:
        return [k for k in okeys if self.is_node(k) and k in root_components]

    def get_available_link_components(self, okeys: OrderedDict, root_components: list[str]) -> list[str]:
        return [k for k in okeys if not self.is_node(k) and k in root_components]

    def get_object_names_by_component(self, root: Root, component: str, wn: Optional[WaterNetworkModel] = None) -> list[str]:
        if wn is not None:
            return get_object_name_list_by_component(component, wn)
        else:
            return root.attrs["onames"][component] if component in root.attrs["onames"] else []

    def compute_selected_components(
        self,
        root: Root,
        sorted_attrs: list[str | tuple],
        has_asterisks: list[bool],
        skip_types: list[str] = [],
    ) -> list[str]:
        if len(sorted_attrs) <= 0:
            return []

        okeys: OrderedDict = root.attrs["okeys"]

        placeholder_components: list[str] = []

        root_components = list({self.get_component(attr) for attr in root.array_keys})

        for attr, has_asterisk in zip(sorted_attrs, has_asterisks):
            if isinstance(attr, str):
                # if it is a pure str, we gather data w.r.t. str key
                assert attr in root.array_keys, f"Root has no attr {attr}! Check spelling"
                com = self.get_component(attr)
                if not has_asterisk:
                    if com == "node":
                        placeholder_components.extend(self.get_available_node_components(okeys=okeys, root_components=root_components))
                    elif com == "link":
                        placeholder_components.extend(self.get_available_link_components(okeys=okeys, root_components=root_components))
                    else:
                        placeholder_components.append(com)

            elif isinstance(attr, tuple):
                for sub_attr in attr:
                    assert sub_attr in root.array_keys, f"Root has no attr {attr}! Check spelling"
                    sub_component = self.get_component(sub_attr)
                    if sub_component not in placeholder_components:
                        placeholder_components.append(sub_component)

        # sort and deduplicate by okeys
        selected_placeholder_components = [k for k in okeys if k in placeholder_components and k not in skip_types]

        if len(selected_placeholder_components) <= 0:
            assert all(has_asterisks)
            selected_placeholder_components.extend([self.get_component(a) for a in sorted_attrs if isinstance(a, str)])

        return selected_placeholder_components

    def _get_all_wdn_folders(self, repo_id: str) -> list[str]:
        # collect all possible file names
        files = list_repo_files(repo_id, repo_type="dataset")
        # retrieve folder name
        wdn_folders = [f.split("/")[0] for f in files if "/" in f]
        # take unique name and convert back to list
        wdn_folders = list(set(wdn_folders))
        return wdn_folders

    def load_roots(
        self,
        wdn_names: list[str],
    ) -> None:
        # avail_wdn_names = self._get_all_wdn_folders(repo_id=HUGGING_FACE_REPO)
        for i, wdn_name in enumerate(wdn_names):
            # assert os.path.isfile(zip_file_path) and zip_file_path[-4:] == ".zip", f"{zip_file_path} is not a zip file"
            assert wdn_name in AVAIL_WDN_NAMES, f"Error! {wdn_name} is not found in the wdn folder list: {AVAIL_WDN_NAMES}"

            if self._is_init_root:
                root = self._roots[i]
            else:
                root = GidaV7.Root(name=wdn_name, num_cpus=self.num_cpus, cache_dir=self.cache_dir)
                self._roots.append(root)
            if self.verbose:
                print(f"config keys = {root.attrs.keys()}")

            # sort user-define attrs w.r.t okeys
            sorted_node_attrs, node_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.node_attrs)
            sorted_edge_attrs, edge_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.edge_attrs)
            sorted_label_attrs, label_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.label_attrs)
            sorted_edge_label_attrs, edge_label_has_asterisks = self.sort_and_filter_key_order(root=root, attrs=self.edge_label_attrs)

            assert not any(set(GidaV7.Pump_Curve).intersection(sorted_edge_attrs)) and not any(
                set(GidaV7.Pump_Curve).intersection(sorted_edge_label_attrs)
            ), "Curve-related parameters are not supported currently"

            # convert wdn to edge_index or load adj_list
            adj_list: list[tuple[str, str, str]] = root.attrs["adj_list"]

            okeys = root.attrs["okeys"]
            root_components = list({self.get_component(attr) for attr in root.array_keys})
            avail_total_node_components = self.get_available_node_components(okeys=okeys, root_components=root_components)
            avail_total_link_components = self.get_available_link_components(okeys=okeys, root_components=root_components)

            skip_nodes: list[str] = self.skip_nodes_list[i] if len(self.skip_nodes_list) > i else []  # type:ignore

            # skip nodes
            if self.bypass_skip_names_in_config:
                if self.verbose:
                    print(
                        "WARN! Flag bypass_skip_names_in_config is (True)! Consequently, GiDa includes invalid nodes that may cause negative pressures!"
                    )
            else:
                skip_nodes = list(set(skip_nodes).union(root.attrs["skip_names"]))

            # update adj according to skip_nodes
            skip_types: list[str] = self.skip_types_list[i] if len(self.skip_types_list) > i else []  # type:ignore

            # get node_components (in-order)
            node_components = self.compute_selected_components(
                root,
                sorted_attrs=sorted_node_attrs,
                has_asterisks=node_has_asterisks,
                skip_types=skip_types,
            )
            edge_components = self.compute_selected_components(
                root,
                sorted_attrs=sorted_edge_attrs,
                has_asterisks=edge_has_asterisks,
                skip_types=skip_types,
            )

            label_node_skip_types = list(set(avail_total_node_components).difference(node_components))
            label_edge_skip_types = list(set(avail_total_link_components).difference(edge_components))
            node_skip_types = list(set(skip_types).union(label_node_skip_types))
            edge_skip_types = list(set(skip_types).union(label_edge_skip_types))

            # update node mask
            node_masks = []
            node_names = []
            for node_com in node_components:
                object_names = self.get_object_names_by_component(root=root, component=node_com)
                if node_com not in node_skip_types:
                    mask_list = [name not in skip_nodes for name in object_names]
                    mask = np.asarray(mask_list, dtype=bool)
                    node_names.extend(list(compress(object_names, mask_list)))
                else:
                    mask = np.zeros(len(object_names), dtype=bool)
                    skip_nodes.extend(object_names)
                node_masks.append(mask)
            node_mask = np.concatenate(node_masks)

            if self.verbose:
                print(f"Skip nodes = {skip_nodes}")

            # update edge mask
            adj_masks = []
            edge_masks = []

            edge_names = []
            for edge_com in edge_components:
                if edge_com not in edge_skip_types:
                    edge_names.extend(self.get_object_names_by_component(root=root, component=edge_com))

            is_edge_names_empty = len(edge_names) <= 0
            for src, dst, link_name in adj_list:
                is_selected = src in node_names and dst in node_names
                is_in_edge_names = is_edge_names_empty or link_name in edge_names
                if is_in_edge_names:
                    edge_masks.append(is_selected)

                adj_masks.append(is_selected and is_in_edge_names)

            edge_names = list(compress(edge_names, edge_masks))
            edge_mask = np.asarray(edge_masks, dtype=bool)
            adj_mask = np.asarray(adj_masks, dtype=bool)

            root.node_mask = node_mask
            root.adj_mask = adj_mask
            root.edge_mask = edge_mask
            root.node_names = node_names
            root.edge_names = edge_names
            root.node_components = node_components
            root.edge_components = edge_components

            root.sorted_node_attrs = sorted_node_attrs
            root.node_has_asterisks = node_has_asterisks

            root.sorted_edge_attrs = sorted_edge_attrs
            root.edge_has_asterisks = edge_has_asterisks

            root.sorted_label_attrs = sorted_label_attrs
            root.label_has_asterisks = label_has_asterisks

            root.sorted_edge_label_attrs = sorted_edge_label_attrs
            root.edge_label_has_asterisks = edge_label_has_asterisks

            # perform skiping
            adj_list = list(compress(adj_list, adj_mask.tolist()))

            # after loading features and obtaining sorted attrs,
            # we update the adjacency list to ensure the nodal consistency between nodal features and adj

            num_nodes = len(node_names)
            num_edges = len(adj_list)
            if num_edges > 0:
                mapping = dict(zip(node_names, range(0, num_nodes)))
                edge_index = np.zeros((2, num_edges), dtype=np.int_)  # type:ignore
                for i, (src, dst, lkn) in enumerate(adj_list):
                    # if src in mapping and dst in mapping :
                    edge_index[0, i] = mapping[src]
                    edge_index[1, i] = mapping[dst]
            else:
                edge_index = np.zeros([2, 0], dtype=np.int_)  # type:ignore

            root.edge_index = edge_index

            # after all, we start download given refined attributes
            root.download_from_hf()

        self._is_init_root = True

    def strip_asterisk(self, attrs: list[str | tuple]) -> list[str | tuple]:
        new_attrs = []
        for attr in attrs:
            if isinstance(attr, str):
                new_attrs.append(attr.strip("*"))
            elif isinstance(attr, tuple):
                assert all("*" not in sub_attr for sub_attr in attr), f"tuple must have no asterisk, but get {attr}"
                new_attrs.append(attr)
        return new_attrs

    def sort_key_order_in_tuple(self, okeys: OrderedDict, compo_params: tuple) -> tuple[tuple, list]:
        if len(compo_params) <= 0:
            return (), []
        ordered_compoparams = []
        for k, vs in okeys.items():
            for v in vs:
                ordered_compoparams.append(f"{k}_{v}")

        indices = [ordered_compoparams.index(compo_param) if compo_param in ordered_compoparams else LARGE_NUMBER for compo_param in compo_params]
        arg_indices = np.argsort(np.array(indices))
        return tuple([compo_params[a] for a in arg_indices]), arg_indices.tolist()

    def sort_and_filter_key_order(self, root: Root, attrs: list[str | tuple]) -> tuple[list[str | tuple], list[bool]]:
        okeys: OrderedDict = root.attrs["okeys"]
        sorted_list = []
        has_asterisks: list[bool] = []
        for attr in attrs:
            if isinstance(attr, str):
                stripped_attr = self.strip_asterisk([attr])[0]
                if stripped_attr in root.array_keys:
                    sorted_list.append(stripped_attr)
                    has_asterisks.append("*" in attr)
                else:
                    com = root._get_component(stripped_attr)  # type:ignore
                    actual_key = f"{com}_{stripped_attr}"
                    if actual_key in root.array_keys:
                        sorted_list.append(actual_key)
                        has_asterisks.append("*" in attr)
            elif isinstance(attr, tuple):
                attr_mask = [sub_attr in root.array_keys for sub_attr in attr]
                attr = tuple(compress(attr, attr_mask))
                sorted_attr, sorted_idx = self.sort_key_order_in_tuple(okeys, compo_params=attr)
                sorted_list.append(sorted_attr)
                has_asterisks.append(False)

        return sorted_list, has_asterisks

    def len(self) -> int:
        r"""Returns the number of data objects stored in the dataset."""
        return self.length

    def get_set(
        self,
        ids: list[int],
        num_records: Optional[int] = None,
        sampling_strategy: Literal["sequential", "interval"] = "sequential",
        **kwargs: Any,
    ) -> Dataset:
        """get subset by computed ids

        Args:
            ids (list[int]): id list
            num_records (Optional[int], optional): pick only num_records samples. Set None to take all. Defaults to None.

        Returns:
            Dataset: a subset from GiDA
        """
        dataset = copy.copy(self)
        for k, v in kwargs.items():
            setattr(dataset, k, v)
        if sampling_strategy == "sequential":
            dataset._indices = ids[:num_records]
        elif sampling_strategy == "interval":
            if num_records is None or len(ids) <= num_records:
                dataset._indices = ids
            else:
                # step size  depends on by #neworks (len(self._roots)) or num_records
                # step_size = max(1, len(ids) // max(len(self._roots), num_records))
                step_size = max(1, len(ids) // num_records)
                tmp_ids = ids[::step_size]

                dataset._indices = tmp_ids[:num_records]
        dataset.length = len(dataset._indices)

        return dataset

    def query(
        self,
        my_ds: hfd.Dataset,
        time_or_one: int,
        scene_ids: list[int] | None = None,
        time_ids: list[int] | None = None,
        # drop_sid_col: bool = True,
        # drop_tid_col: bool = True,
        num_scenes: int = 1000,
    ) -> np.ndarray:
        """We exclude nan columns and aggregate columns w.r.t. the order of ds.column_names

        Args:
            my_ds (hfd.Dataset): datasets.Dataset object
            time_or_one (int): time dim for dynamic or 1 for static
            row_indices (list[int] | None): the row ids. Default is None (take all)
            drop_sid_col  (bool): drop scene id column. Default is True
            drop_tid_col  (bool): drop time id column. Default is True
            num_scenes (int): the number of generated scenes
        Returns:
            np.ndarray: the big matrix in np format
        """
        # col_list = [col for col in my_ds.column_names if (drop_sid_col and col != "scenario_id") and (drop_tid_col and col != "time_id")]
        # col_list = my_ds.column_names
        ds: hfd.Dataset = my_ds  # .select_columns(col_list)

        assert scene_ids is not None or time_ids is not None

        if scene_ids is not None and time_ids is None:
            if time_or_one > 1:
                # [t0_sx,t1_sx,.....,t8759_sx,t0_s(x+1),...]
                cids = np.concatenate([np.arange(time_or_one * sid, time_or_one * sid + time_or_one) for sid in scene_ids])
            else:
                # cids == scene_ids
                cids = np.asarray(scene_ids)

            ds = ds.select(cids)
            # arr should have shape (len(scene_ids), time_or_one, num_components)
            # arr: np.ndarray = np.concatenate([ds[col].reshape([-1, time_or_one, 1]) for col in col_list], axis=-1)  # type:ignore
            arr: np.ndarray = ds.to_pandas().to_numpy().reshape([-1, time_or_one, ds.shape[-1]])  # type:ignore
        elif scene_ids is None and time_ids is not None:
            if time_or_one > 1:
                # [tx_s0,tx_s1,.....,tx_s999,t(x+1)_s0,...]
                cids = np.concatenate([np.arange(tid, num_scenes * time_or_one, time_or_one) for tid in time_ids])
            else:
                # [tx_s0,tx_s1,.....,tx_s999,t(x+1)_s0,...]
                cids = np.tile(np.arange(num_scenes), len(time_ids))

            ds = ds.select(cids)
            # arr should have shape (len(time_ids), num_scenes, num_components)
            # arr: np.ndarray = np.concatenate([ds[col].reshape([-1, num_scenes, 1]) for col in col_list], axis=-1)  # type:ignore
            arr: np.ndarray = ds.to_pandas().to_numpy().reshape([-1, num_scenes, ds.shape[-1]])  # type:ignore
        else:
            assert scene_ids is not None and time_ids is not None
            if time_or_one > 1:
                # [ [t0_sx,t1_sx,.....,t8759_sx][time_ids] ,[t0_s(x+1),...t_8759_s(x+1)[time_ids]],...]
                # cids = np.concatenate([np.arange(time_or_one * sid, time_or_one * sid + time_or_one)[time_ids] for sid in scene_ids])
                cids = np.concatenate(
                    [np.arange(time_or_one * sid, time_or_one * sid + time_or_one)[tid].reshape([-1]) for sid, tid in zip(scene_ids, time_ids)]
                )
                # cids = []
                # for sid, tid in zip(scene_ids,time_ids):
                #     time_block = np.arange(time_or_one * sid, time_or_one * sid + time_or_one)
                #     cids.append(time_block[tid])
                # cids = np.asarray(cids)
            else:
                # [ [ty_sx,t(y+1)_sx,...], [ty_s(x+1), t(y+1)_s(x+1),...] ,...]
                # cids = np.repeat(np.asarray(scene_ids), len(time_ids))
                cids = np.asarray(scene_ids)
            ds = ds.select(cids)

            # arr should have shape (len(scene_ids)* len(time_ids), 1, num_components)
            # arr: np.ndarray = np.concatenate([ds[col].reshape([len(scene_ids), -1, 1]) for col in col_list], axis=-1)  # type:ignore
            arr: np.ndarray = ds.to_pandas().to_numpy().reshape([-1, 1, ds.shape[-1]])  # type:ignore
        return arr

    def get_and_cache_if_need(self, root: Root, key: str, indices: list[tuple[int | None, int | None]], do_cache: bool = False) -> np.ndarray:
        if do_cache:
            if not hasattr(self, "_array_cache"):
                setattr(self, "_array_cache", {})
            is_dynamic_key = key in GidaV7.DYNAMIC_PARAMS
            array_cache: dict[str, np.ndarray] = getattr(self, "_array_cache")
            time_or_one = root.time_dim if is_dynamic_key else 1

            cache_key = "_".join([root.name, key])
            if cache_key not in array_cache:
                my_ds: hfd.Dataset = root.get_array_by_key(key)
                # original arr has shape (num_scenes* time_or_one, num_components)
                # convert to arr has shaoe (num_scenes, time_or_one, num_components)
                # OOM is acceptable
                arr: np.ndarray = self.query(my_ds=my_ds, time_or_one=time_or_one, scene_ids=None, num_scenes=root.attrs["num_samples"])
                # each has shape (num_scenes, num_objects, time_or_one)
                arr = arr.transpose(0, 2, 1)
                array_cache[cache_key] = arr

            optional_scene_ids, optional_time_ids = tuple(map(list, zip(*indices)))
            non_scene_dim = optional_scene_ids[0] is None
            non_time_dim = optional_time_ids[0] is None
            assert non_time_dim is not None or non_time_dim is not None

            if not non_scene_dim and non_time_dim:  # scene
                return array_cache[cache_key][optional_scene_ids]
            elif non_scene_dim and not non_time_dim:  # temporal
                arr = array_cache[cache_key]
                if arr.shape[-1] < len(optional_time_ids) and not is_dynamic_key:
                    return np.broadcast_to(arr, (arr.shape[0], arr.shape[1], len(optional_time_ids)))
                else:
                    return arr[..., optional_time_ids]
            else:  # snapshot
                arr = array_cache[cache_key]
                if arr.shape[-1] == 1 and not is_dynamic_key:
                    return arr[optional_scene_ids]
                else:
                    indexed_arr = arr[optional_scene_ids, :, optional_time_ids]
                    ret_arr = np.expand_dims(indexed_arr, axis=-1)
                    return ret_arr

        else:
            return self.get_numpy_array(root=root, key=key, indices=indices)

    def get_numpy_array(self, root: Root, key: str, indices: list[tuple[int | None, int | None]]) -> np.ndarray:
        # my_arr has shape (num_scenes,  time_or_one* num_objects)
        my_ds: hfd.Dataset = root.get_array_by_key(key)

        optional_scene_ids, optional_time_ids = tuple(map(list, zip(*indices)))

        non_scene_dim = optional_scene_ids[0] is None
        non_time_dim = optional_time_ids[0] is None
        assert non_time_dim is not None or non_time_dim is not None

        time_or_one = root.time_dim if key in GidaV7.DYNAMIC_PARAMS else 1
        arr = self.query(
            my_ds=my_ds,
            time_or_one=time_or_one,
            scene_ids=optional_scene_ids if not non_scene_dim else None,
            time_ids=optional_time_ids if not non_time_dim else None,
            num_scenes=root.attrs["num_samples"],
        )
        # arr has shape (len(indices), num_components, time_or_one)
        arr = arr.transpose(0, 2, 1)

        # arr must have shape of (batch_size, num_objects, time_dim)
        return arr

    def stack_features(
        self,
        root: Root,
        indices: list[tuple[int | None, int | None]],
        which_array: Literal["node", "edge", "label", "edge_label"] = "node",
    ) -> np.ndarray | None:
        if which_array == "node":
            sorted_attrs: list[str | tuple] = root.sorted_node_attrs
            has_asterisks: list[bool] = root.node_has_asterisks
            selected_placeholder_components = root.node_components
            is_node = True
        elif which_array == "edge":
            sorted_attrs: list[str | tuple] = root.sorted_edge_attrs
            has_asterisks: list[bool] = root.edge_has_asterisks
            selected_placeholder_components = root.edge_components
            is_node = False
        elif which_array == "label":
            sorted_attrs: list[str | tuple] = root.sorted_label_attrs
            has_asterisks: list[bool] = root.label_has_asterisks
            is_label_nodal = self.is_node(self.get_component_by_attr(sorted_attrs[0])) if len(sorted_attrs) > 0 else True
            selected_placeholder_components = root.node_components if is_label_nodal else root.edge_components
            is_node = is_label_nodal
        elif which_array == "edge_label":
            sorted_attrs: list[str | tuple] = root.sorted_edge_label_attrs
            has_asterisks: list[bool] = root.edge_label_has_asterisks
            selected_placeholder_components = root.edge_components
            is_node = False

        if len(sorted_attrs) <= 0:
            return None

        merging_arrs = []

        max_dim: int = -1
        for att, has_asterisk in zip(sorted_attrs, has_asterisks):
            if isinstance(att, str):
                # my_arr has shape (batch_size, num_objects, time_dim) if axis=0
                my_arr: np.ndarray = self.get_and_cache_if_need(root=root, key=att, indices=indices, do_cache=self.do_cache)
                if not has_asterisk:
                    max_dim = max(max_dim, my_arr.shape[1])
            elif isinstance(att, tuple):
                # each has shape (batch_size, DIFF num_objects, time_dim) if axis=0
                my_arrs: list[np.ndarray] = [
                    self.get_and_cache_if_need(root=root, key=sub_att, indices=indices, do_cache=self.do_cache) for sub_att in att
                ]
                my_arr = np.concatenate(my_arrs, axis=1)
                max_dim = max(max_dim, my_arr.shape[1])
            else:
                raise NotImplementedError

            merging_arrs.append(my_arr)

        if max_dim == -1:  # no non-asterisk attr
            max_dim = sum(a.shape[1] for a in merging_arrs)

        required_padding = any(has_asterisks)

        if required_padding:
            for i in range(len(merging_arrs)):
                feature: np.ndarray = merging_arrs[i]
                has_asterisk = has_asterisks[i]
                attr = sorted_attrs[i]
                if has_asterisk and feature.shape[1] < max_dim:
                    if self.verbose:
                        f"WARN! Attribute {attr} has feature.shape[1]: {feature.shape[1]} >= max dim: {max_dim}! It does not need asterisk sign!"
                    assert isinstance(attr, str)
                    component_i = self.get_component(attr)
                    placeholder_id = selected_placeholder_components.index(component_i)
                    prev_pad = 0
                    if placeholder_id > 0:
                        for j in range(0, placeholder_id):
                            component_j = selected_placeholder_components[j]
                            num_obj_dim_j = len(self.get_object_names_by_component(root, component_j, wn=None))
                            prev_pad += num_obj_dim_j

                    post_pad = 0
                    if placeholder_id < len(selected_placeholder_components) - 1:
                        for j in range(placeholder_id + 1, len(selected_placeholder_components)):
                            component_j = selected_placeholder_components[j]
                            num_obj_dim_j = len(self.get_object_names_by_component(root, component_j, wn=None))
                            post_pad += num_obj_dim_j

                    arr = np.pad(feature, ((0, 0), (prev_pad, post_pad), (0, 0)), constant_values=self.unstackable_pad_value)
                    assert arr.shape[1] == max_dim, (
                        "Error! Padding does not match! Check the consistency between sub attributes in tuple, and other attributes (e.g. curve attr)"
                    )

                    merging_arrs[i] = arr
                    if self.verbose:
                        print(f"vstack-[pad]- key {attr} shape: {feature.shape}(old) -> {arr.shape}(new)")
        cat_array = np.concatenate(merging_arrs, axis=-1)

        if is_node:
            cat_array = cat_array[:, root.node_mask]
        else:
            cat_array = cat_array[:, root.edge_mask]
        if self.verbose:
            print(f"vstack-prevskip:shape> all:{cat_array.shape}")

            print("*" * 40)
        return cat_array

    def get(self, idx: int | Sequence) -> Any:
        assert self._indices is not None
        if isinstance(idx, int):
            fids: list[int] = self._indices[idx]  # [idx]  #
        else:
            idx_list = np.asarray(idx).flatten().tolist()

            fids: list[int] = [self._indices[i] for i in idx_list]  # idx_list  #

        batch_size = len(fids)

        # group temporal ids by network id to save querying time
        root_vindices_dict = defaultdict(list)
        for fid in fids:
            network_id = self._network_map[fid]
            root_vindices_dict[network_id].append(self._index_map[fid])

        batch = []
        counter: int = 0
        for nid in root_vindices_dict.keys():
            root: GidaV7.Root = self._roots[nid]
            # print(f"Querying network nid=({nid})- zip_path=({os.path.basename(root.zip_file_path)}) ...")
            if counter >= batch_size:
                break
            index_tup_list: list[tuple[int | None, int | None]] = root_vindices_dict[nid]
            node_array = self.stack_features(root=root, indices=index_tup_list, which_array="node")
            assert node_array is not None

            edge_array = self.stack_features(root=root, indices=index_tup_list, which_array="edge")

            label_array = self.stack_features(root=root, indices=index_tup_list, which_array="label")

            edge_label_array = self.stack_features(root=root, indices=index_tup_list, which_array="edge_label")

            for i in range(node_array.shape[0]):
                if counter >= batch_size:
                    break
                x = node_array[i]
                edge_attr = edge_array[i] if edge_array is not None else None
                y = label_array[i] if label_array is not None else None
                edge_y = edge_label_array[i] if edge_label_array is not None else None
                edge_index: np.ndarray = root.edge_index  # self._roots[self._network_map[i]].edge_index  # type:ignore

                dat = parse2data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_y=edge_y)
                batch.append(dat)
                counter += 1
        assert counter == batch_size
        return batch  # type:ignore

    def indices(self) -> Sequence:
        assert self._indices is not None
        return self._indices

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ) -> Union["Dataset", BaseData]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        if isinstance(idx, (int, np.integer)) or (isinstance(idx, Tensor) and idx.dim() == 0) or (isinstance(idx, np.ndarray) and np.isscalar(idx)):
            # data = self.get(self.indices()[idx])[0]  # type:ignore
            data = self.get([idx])[0]  # type:ignore
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            batch = self.get(idx)  # type:ignore
            batch = batch if self.transform is None else [self.transform(data) for data in batch]  # type:ignore
            return batch  # type:ignore

    def __getitems__(self, idx: Union[int, np.integer, IndexType]) -> list[BaseData]:
        # return self.get(idx)  # type:ignore
        # batch: list[BaseData] = self.get(idx[0])  # type:ignore
        batch: list[BaseData] = self.get(idx)  # type:ignore

        batch = batch if self.transform is None else [self.transform(dat) for dat in batch]
        return batch

    def gather_statistic(
        self,
        which_array: Literal["node", "edge", "label", "edge_label"],
        norm_dim: Optional[int] = 0,
        to_tensor: bool = True,
        num_batches: int = 10,
    ) -> tuple[
        np.ndarray | Tensor,
        np.ndarray | Tensor,
        np.ndarray | Tensor,
        np.ndarray | Tensor,
    ]:
        """
        norm_dim only supports 0 or -1. Otherwise, we cannot concatenate arrays from different-in-size networks </br>
        if compute, stat tuple with be saved in `self.dataset_log_pt_path` (if non-empty), a .pt file. </br>
        return: (min_val, max_val, mean_val, std_val)
        """
        channel_dim = -1
        assert self._indices is not None
        which_array_attr_map = {
            "node": "x",
            "edge": "edge_attr",
            "label": "y",
            "edge_label": "edge_y",
        }
        which_array_attrs_map = {
            "node": getattr(self._roots[0], "sorted_node_attrs"),
            "edge": getattr(self._roots[0], "sorted_edge_attrs"),
            "label": getattr(self._roots[0], "sorted_label_attrs"),
            "edge_label": getattr(self._roots[0], "sorted_edge_label_attrs"),
        }

        time_dim = self._roots[0].time_dim
        param_attrs = which_array_attrs_map[which_array]
        assert param_attrs is not None and len(param_attrs) > 0, f"ERROR! No found paramattrs from which_array=({which_array}): ({param_attrs})"
        channel_splitters = [
            time_dim if (isinstance(pa, str) and pa in GidaV7.DYNAMIC_PARAMS) or (pa[0] in GidaV7.DYNAMIC_PARAMS) else 1 for pa in param_attrs
        ]
        sum_channels = sum(channel_splitters)

        are_all_same_as_sum_channels: bool = True

        cat_arrays: list[dac.Array] = []

        batch_indices = np.array_split(np.arange(len(self.indices())), num_batches)

        selected_attr_key = which_array_attr_map[which_array]
        for bids in batch_indices:
            bids_list = bids.tolist()
            if bids_list:
                data_list: list = self.get(bids_list)
                batch = Batch.from_data_list(data_list)
                batch_att_array: Tensor = getattr(batch, selected_attr_key)
                if batch_att_array.shape[-1] != sum_channels:
                    are_all_same_as_sum_channels = False
                dac_arr: dac.Array = dac.from_array(batch_att_array.numpy())
                cat_arrays.append(dac_arr)

        is_norm_dim_different_from_channel_dim: bool = norm_dim != -1 and norm_dim != len(cat_arrays[0].shape) - 1
        do_group_norm: bool = is_norm_dim_different_from_channel_dim and are_all_same_as_sum_channels and self.batch_axis_choice != "snapshot"
        if is_norm_dim_different_from_channel_dim and not are_all_same_as_sum_channels and self.batch_axis_choice != "snapshot":
            print(
                f"WARN! Networks' arrays do not match their channels. Group norm by channels is prohibited! Flatten norm along {norm_dim} instead!\
                This could change behavior and lead to unwanted statistic!"
            )

        for i in range(len(cat_arrays)):
            arr = cat_arrays[i]
            cat_arrays[i] = arr.reshape([-1, arr.shape[-1]], limit=self.chunk_limit)

        # flatten_array has shape (-1, t+ 1 + t + ...)  or (-1)
        flatten_array = dac.concatenate(cat_arrays, axis=0)

        if do_group_norm:
            std_vals, mean_vals, min_vals, max_vals = [], [], [], []
            current_idx = 0
            for i in range(len(channel_splitters)):
                num_channels = channel_splitters[i]
                t = flatten_array[:, current_idx : current_idx + num_channels]

                # t = t.flatten()
                current_idx += num_channels

                t_std_val, t_mean_val = t.std(axis=norm_dim), t.mean(axis=norm_dim)
                t_min_val, t_max_val = t.min(axis=norm_dim), t.max(axis=norm_dim)

                if norm_dim is not None:
                    t_std_val = np.expand_dims(t_std_val, axis=norm_dim)

                    t_mean_val = np.expand_dims(t_mean_val, axis=norm_dim)

                    t_min_val = np.expand_dims(t_min_val, axis=norm_dim)

                    t_max_val = np.expand_dims(t_max_val, axis=norm_dim)
                else:
                    t_std_val = t_std_val.reshape([-1]).repeat(num_channels)
                    t_mean_val = t_mean_val.reshape([-1]).repeat(num_channels)
                    t_min_val = t_min_val.reshape([-1]).repeat(num_channels)
                    t_max_val = t_max_val.reshape([-1]).repeat(num_channels)

                std_vals.append(t_std_val)
                mean_vals.append(t_mean_val)
                min_vals.append(t_min_val)
                max_vals.append(t_max_val)

            std_val = dac.concatenate(std_vals, axis=channel_dim)
            mean_val = dac.concatenate(mean_vals, axis=channel_dim)
            min_val = dac.concatenate(min_vals, axis=channel_dim)
            max_val = dac.concatenate(max_vals, axis=channel_dim)
        else:
            std_val, mean_val = flatten_array.std(axis=norm_dim), flatten_array.mean(axis=norm_dim)
            min_val, max_val = flatten_array.min(axis=norm_dim), flatten_array.max(axis=norm_dim)

        (min_val, max_val, mean_val, std_val) = compute(*(min_val, max_val, mean_val, std_val))

        if not isinstance(min_val, np.ndarray):
            min_val = np.asarray(min_val, dtype=np.float32).reshape([1, -1])
            max_val = np.asarray(max_val, dtype=np.float32).reshape([1, -1])
            mean_val = np.asarray(mean_val, dtype=np.float32).reshape([1, -1])
            std_val = np.asarray(std_val, dtype=np.float32).reshape([1, -1])
        else:
            min_val = min_val.astype(np.float32)
            max_val = max_val.astype(np.float32)
            mean_val = mean_val.astype(np.float32)
            std_val = std_val.astype(np.float32)

        if to_tensor:
            std_val = from_numpy(std_val)
            mean_val = from_numpy(mean_val)
            min_val = from_numpy(min_val)
            max_val = from_numpy(max_val)

        return (min_val, max_val, mean_val, std_val)

    def save_dataset_checkpoint(self, **kwargs: Any) -> str:
        assert self.dataset_log_pt_path[-4:] == ".pth" or self.dataset_log_pt_path[-3:] == ".pt"

        if os.path.exists(self.dataset_log_pt_path):
            # prevent overriden
            cp_dict: dict = load(self.dataset_log_pt_path)
            cp_dict.update(kwargs)
            save(cp_dict, self.dataset_log_pt_path)
        else:
            save(kwargs, self.dataset_log_pt_path)

        return self.dataset_log_pt_path
