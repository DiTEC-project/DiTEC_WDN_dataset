#
# Created on Fri Jun 06 2025
# Copyright (c) 2025 Huy Truong
# ------------------------------
# Purpose: This is a minimal example how to load the data using interface
# Use GidaV7 for .parquet (HF) - This is OPTIONAL and slower than V6 empirically.
# Use GidaV6 for .zarr - This is the optimal approach but you need .zarr file.
# ------------------------------
#

from ditec_wdn_dataset.core.datasets_large import GidaV6
from ditec_wdn_dataset.hf.dataset import GidaV7
from ditec_wdn_dataset.utils.configs import GidaConfig


def tutorial_v6(gida_yaml_path: str) -> list[GidaV6]:
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())
    actual_num_records: int = gida_config.num_records if gida_config.num_records is not None else full_gida.length

    # test
    print(next(iter(full_gida)))

    # this is optional, let us know if you wish these code should be included in the interface
    if gida_config.subset_shuffle or gida_config.num_records is None:
        # we perform subset_shuffle prior the custom file. If it is empty, we try to load from dataset_log.pt. If it is failed, we create some and save to dataset_log.pt
        custom_subset_shuffle_pt_path = ""
        full_gida.process_subset_shuffle(custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, create_and_save_to_dataset_log_if_nonexist=True)

    train_samples = int(actual_num_records * full_gida.split_ratios[0])
    val_samples = int(actual_num_records * full_gida.split_ratios[1])
    test_samples = actual_num_records - train_samples - val_samples

    ret_datasets = []
    # init subsets
    if gida_config.split_set in ["train", "all"]:
        train_set = full_gida.get_set(full_gida.train_ids, num_records=train_samples)
        ret_datasets.append(train_set)

    if gida_config.split_set in ["val", "all"]:
        valid_set = full_gida.get_set(full_gida.val_ids, num_records=val_samples, transform=None)
        ret_datasets.append(valid_set)

    if gida_config.split_set in ["test", "all"]:
        test_set = full_gida.get_set(full_gida.test_ids, num_records=test_samples, transform=None)
        ret_datasets.append(test_set)

    return ret_datasets


def tutorial_v7(gida_yaml_path: str) -> list[GidaV7]:
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)

    full_gida: GidaV7 = GidaV7(**gida_config.as_dict())
    actual_num_records: int = gida_config.num_records if gida_config.num_records is not None else full_gida.length

    # test
    print(next(iter(full_gida)))

    # this is optional, let us know if you wish these code should be included in the interface
    if gida_config.subset_shuffle or gida_config.num_records is None:
        # we perform subset_shuffle prior the custom file. If it is empty, we try to load from dataset_log.pt. If it is failed, we create some and save to dataset_log.pt
        custom_subset_shuffle_pt_path = ""
        full_gida.process_subset_shuffle(custom_subset_shuffle_pt_path=custom_subset_shuffle_pt_path, create_and_save_to_dataset_log_if_nonexist=True)
    else:
        # we perform sampling on training and validation sets
        step = full_gida.length // gida_config.num_records
        assert step > 0
        full_gida.train_ids = full_gida.train_ids[::step]
        full_gida.val_ids = full_gida.val_ids[::step]

    train_samples = int(actual_num_records * full_gida.split_ratios[0])
    val_samples = int(actual_num_records * full_gida.split_ratios[1])
    test_samples = actual_num_records - train_samples - val_samples

    ret_datasets = []
    # init subsets
    if gida_config.split_set in ["train", "all"]:
        train_set = full_gida.get_set(full_gida.train_ids, num_records=train_samples)
        ret_datasets.append(train_set)

    if gida_config.split_set in ["val", "all"]:
        valid_set = full_gida.get_set(full_gida.val_ids, num_records=val_samples, transform=None)
        ret_datasets.append(valid_set)

    if gida_config.split_set in ["test", "all"]:
        test_set = full_gida.get_set(full_gida.test_ids, num_records=test_samples, transform=None)
        ret_datasets.append(test_set)

    return ret_datasets


if __name__ == "__main__":
    # they are minimal examples loaded from configs
    # tutorial_v6("ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml")
    # tutorial_v7("ditec_wdn_dataset/arguments/test_data_interface_v7_config.yaml")

    # this is a minimal example for the data interface
    full_gida = GidaV7(
        wdn_names=["CTOWN_1GB_24H"],
        node_attrs=["pressure", GidaV7.Node_Elevation],
        edge_attrs=["pipe_diameter"],
        edge_label_attrs=[],  # keep it empty if unsed,
        label_attrs=["demand"],
        num_records=100,  # keep it small to prevent OOM
        batch_axis_choice="scene",  # record unit (scenario)
        verbose=True,  # for more details
    )
    print(len(full_gida.train_ids))
    train_set = full_gida.get_set(full_gida.train_ids)
    print(next(iter(train_set)))
    print(len(full_gida.val_ids))
    valid_set = full_gida.get_set(full_gida.val_ids)
    print(next(iter(valid_set)))
    print(len(full_gida.test_ids))
    test_set = full_gida.get_set(full_gida.test_ids)
    print(next(iter(test_set)))
