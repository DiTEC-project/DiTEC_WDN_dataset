from typing import Literal
import numpy as np
import zarr
from ditec_wdn_dataset.core.datasets_large import GidaV6
from ditec_wdn_dataset.utils.configs import GidaConfig
from torch import as_tensor, isclose, not_equal, tensor, allclose, equal
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import pytest
import timeit


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "scene", "temporal"])
def test_check_x_equal_y_multiple_equal_nets(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",
        r"G:/My Drive/Dataset/huy_v3/simgen_epanet2_20241004_1246.zip",
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.indexing = "dynamic"
    gida_config.batch_axis_choice = batch_axis_choice
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())
    sample = next(iter(full_gida))
    assert allclose(sample.x, sample.y)


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "scene", "temporal"])
def test_interleaving_on_multinets(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",
        r"G:/My Drive/Dataset/huy_v3/simgen_epanet2_20241004_1246.zip",
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.indexing = "dynamic"

    gida_config.batch_axis_choice = batch_axis_choice
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())

    # print(f"flat indices[:20] = {full_gida._indices[:20]}")
    nid_rid_tuples = [full_gida._unflatten(i, full_gida.num_networks) for i in full_gida._indices[:20]]
    nids, rids = zip(*nid_rid_tuples)

    # print(f"nids[:20] = {nids}")
    # print(f"rids[:20] = {rids}")

    assert np.equal(nids, np.tile(np.arange(full_gida.num_networks), reps=len(nids) // full_gida.num_networks)).all()
    rids_gt = np.repeat(np.arange(10), repeats=2)
    assert np.equal(rids, rids_gt).all()


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "scene", "temporal"])
def test_node_single_multiple_equal_nets(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    """single nodal attribute"""
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = []
    gida_config.edge_label_attrs = []
    gida_config.indexing = "dynamic"

    gida_config.batch_axis_choice = batch_axis_choice
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())
    loader = DataLoader(full_gida, batch_size=20)

    next_batch = next(iter(loader))

    if batch_axis_choice == "scene":
        shape = [-1, 19, 8760]
    elif batch_axis_choice == "snapshot":
        shape = [-1, 19, 1]
    else:
        shape = [-1, 19, 1000]

    first_sample = next_batch.x.reshape(shape)[0]

    store = zarr.open(store=gida_config.zip_file_paths[0], mode="r")
    # actual_sample has shape (1000, time_dim * num_components)
    actual_sample = store[gida_config.node_attrs[0]]

    if batch_axis_choice == "scene":
        # actual_sample has shape (time_dim * num_components)
        actual_sample = tensor(actual_sample[0])
        # actual_sample has shape (time_dim, num_components)
        actual_sample = actual_sample.reshape([8760, -1])
        # actual_sample has shape (time_dim, selected_num_components)
        actual_sample = actual_sample[..., full_gida._roots[0].node_mask]  # type:ignore
        # actual_sample has shape (selected_num_components,time_dim)
        actual_sample = actual_sample.transpose(1, 0)
        assert allclose(actual_sample, first_sample)
    elif batch_axis_choice == "snapshot":
        # actual_sample has shape (time_dim * num_components)
        actual_sample = tensor(actual_sample[0])
        # actual_sample has shape (time_dim, num_components)
        actual_sample = actual_sample.reshape([8760, -1])
        # actual_sample has shape (time_dim, selected_num_components)
        actual_sample = actual_sample[..., full_gida._roots[0].node_mask]  # type:ignore
        # actual_sample has shape (selected_num_components,time_dim)
        actual_sample = actual_sample.transpose(1, 0)
        assert allclose(actual_sample[..., 0].unsqueeze(-1), first_sample)
    else:
        # actual_sample has shape (1000, time_dim * num_components)
        actual_sample = tensor(actual_sample[:])
        # actual_sample has shape (1000, time_dim, num_components)
        actual_sample = actual_sample.reshape([1000, 8760, -1])
        # actual_sample has shape (1000, time_dim, selected_num_components)
        actual_sample = actual_sample[..., full_gida._roots[0].node_mask]  # type:ignore
        # actual_sample has shape (1000, selected_num_components)
        actual_sample = actual_sample[:, 0, :]  # type:ignore
        # actual_sample has shape (selected_num_components, 1000)
        actual_sample = actual_sample.transpose(1, 0)
        assert allclose(actual_sample, first_sample)


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "scene", "temporal"])
def test_shuffle_multiple_nets(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",
        r"G:/My Drive/Dataset/huy_v3/simgen_epanet2_20241004_1246.zip",
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.indexing = "dynamic"

    gida_config.batch_axis_choice = batch_axis_choice
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())

    shuffle_full_gida = full_gida.shuffle()

    assert (as_tensor(shuffle_full_gida._indices, dtype=int) != as_tensor(full_gida._indices, dtype=int)).any()
    # not equal(as_tensor(shuffle_full_gida._indices, dtype=int), as_tensor(full_gida._indices, dtype=int))

    ns_x = next(iter(DataLoader(full_gida))).x
    s_x = next(iter(DataLoader(shuffle_full_gida))).x

    # print(f"ns_x shape = {ns_x.shape}")
    # print(f"s_x shape = {s_x.shape}")
    # print(f"ns_x[...,0] = {ns_x[..., 0]}")
    # print(f"s_x[...,0] = {s_x[..., 0]}")
    assert (ns_x != s_x).any()


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "scene", "temporal"])
def test_subset_split(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",
        r"G:/My Drive/Dataset/huy_v3/simgen_epanet2_20241004_1246.zip",
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = []
    gida_config.edge_label_attrs = []
    gida_config.indexing = "dynamic"

    gida_config.batch_axis_choice = batch_axis_choice
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())

    train_set = full_gida.get_set(full_gida.train_ids, num_records=10)
    val_set = full_gida.get_set(full_gida.val_ids, num_records=10)
    test_set = full_gida.get_set(full_gida.test_ids, num_records=10)

    assert (as_tensor(train_set._indices, dtype=int) != as_tensor(val_set._indices, dtype=int)).all()
    assert (as_tensor(test_set._indices, dtype=int) != as_tensor(val_set._indices, dtype=int)).all()
    assert (as_tensor(test_set._indices, dtype=int) != as_tensor(train_set._indices, dtype=int)).all()

    assert (as_tensor(train_set._indices, dtype=int) == as_tensor(full_gida._indices, dtype=int)[:10]).all()
    assert (
        as_tensor(val_set._indices, dtype=int) == as_tensor(full_gida._indices, dtype=int)[len(full_gida.train_ids) : len(full_gida.train_ids) + 10]
    ).all()
    assert (
        as_tensor(test_set._indices, dtype=int) == as_tensor(full_gida._indices, dtype=int)[-len(full_gida.test_ids) : -len(full_gida.test_ids) + 10]
    ).all()


def _iterate_over_a_set(batch_size: int, gida_config: GidaConfig, shuffle_at_set: bool, shuffler_at_loader: bool):
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())
    train_set = full_gida.get_set(full_gida.train_ids)

    if shuffle_at_set:
        train_set = train_set.shuffle()

    data_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffler_at_loader)
    for batch in data_loader:
        batch.y = batch.x  # do random job


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "scene", "temporal"])
def test_benchmark_shuffle(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",
        # r"G:/My Drive/Dataset/huy_v3/simgen_epanet2_20241004_1246.zip",
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = []
    gida_config.edge_label_attrs = []
    gida_config.num_records = 100
    gida_config.indexing = "dynamic"

    gida_config.batch_axis_choice = batch_axis_choice
    if batch_axis_choice == "scene":
        batch_size = 10
    elif batch_axis_choice == "snapshot":
        batch_size = 10
    else:
        batch_size = 10
    number = 3
    on_off_execution_time = timeit.timeit(lambda: _iterate_over_a_set(batch_size, gida_config, True, False), number=number)
    on_off_execution_time = on_off_execution_time / number
    print(f"Average execution time: {on_off_execution_time:.6f} seconds")

    off_on_execution_time = timeit.timeit(lambda: _iterate_over_a_set(batch_size, gida_config, False, True), number=number)
    off_on_execution_time = off_on_execution_time / number
    print(f"Average execution time: {off_on_execution_time:.6f} seconds")

    on_on_execution_time = timeit.timeit(lambda: _iterate_over_a_set(batch_size, gida_config, True, True), number=number)
    on_on_execution_time = on_on_execution_time / number
    print(f"Average execution time: {on_on_execution_time:.6f} seconds")

    off_off_execution_time = timeit.timeit(lambda: _iterate_over_a_set(batch_size, gida_config, False, False), number=number)
    off_off_execution_time = off_off_execution_time / number
    print(f"Average execution time: {off_off_execution_time:.6f} seconds")
    import matplotlib.pyplot as plt

    plt.bar(["on_off", "off_on", "on_on", "off_off"], [on_off_execution_time, off_on_execution_time, on_on_execution_time, off_off_execution_time])
    plt.xlabel("Dataset shuffle - DataLoader shuffle")
    plt.ylabel("Exe time (sec)")
    plt.title(
        f"#networks: {len(gida_config.zip_file_paths)}, batch_size: {batch_size}, num_records: {gida_config.num_records} ({gida_config.batch_axis_choice}), repeat in {number} times"
    )
    plt.show()

    assert on_off_execution_time < off_on_execution_time
    assert on_off_execution_time < on_on_execution_time


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "temporal"])
def test_unequal_timelength(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        r"G:\My Drive\Dataset\from_habrok\simgen_EXN_20241119_0325.zip",  # <--------------------------24h
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",  # <-------------------------- 8760h
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.indexing = "dynamic"

    gida_config.batch_axis_choice = batch_axis_choice
    full_gida: GidaV6 = GidaV6(**gida_config.as_dict())

    sample_0 = full_gida[0]
    sample_2 = full_gida[2]
    sample_48 = full_gida[48]
    # sample_49 = full_gida[49]
    sample_50 = full_gida[50]
    # sample_51 = full_gida[51]

    # print(f"sample_0.x.shape = {sample_0.x.shape}")
    # print(f"sample_2.x.shape = {sample_2.x.shape}")
    # print(f"sample_48.x.shape = {sample_48.x.shape}")
    # print(f"sample_49.x.shape = {sample_49.x.shape}")
    # print(f"sample_50.x.shape = {sample_50.x.shape}")
    # print(f"sample_51.x.shape = {sample_51.x.shape}")

    # print(f"sample_0.x[:10] = {sample_0.x[:10]}")
    # print(f"sample_48.x[:10] = {sample_48.x[:10]}")
    # print(f"sample_2.x[:10] = {sample_2.x[:10]}")
    # print(f"sample_50.x[:10] = {sample_50.x[:10]}")
    assert allclose(sample_48.x, sample_0.x)
    assert allclose(sample_50.x, sample_2.x)


@pytest.mark.parametrize("batch_axis_choice", ["snapshot", "temporal"])
def test_static_and_dynamic_indexing(batch_axis_choice: Literal["snapshot", "scene", "temporal"]):
    gida_yaml_path: str = r"ditec_wdn_dataset/arguments/test_data_interface_v6_config.yaml"
    gida_config = GidaConfig()
    gida_config._parsed = True
    gida_config._from_yaml(gida_yaml_path, unsafe_load=True)
    gida_config.zip_file_paths = [
        # r"G:/My Drive/Dataset/huy_v3/simgen_epanet2_20241004_1246.zip",
        r"G:/My Drive/Dataset/huy_v3/simgen_Anytown_20241118_1026.zip",  # <-------------------------- 8760h
    ]
    gida_config.node_attrs = ["demand"]
    gida_config.edge_label_attrs = []
    gida_config.label_attrs = []
    gida_config.edge_label_attrs = []
    gida_config.batch_axis_choice = batch_axis_choice
    gida_config.subset_shuffle = False
    gida_config.num_records = 8760000 * 20

    # gida_config.indexing = "static"
    # static_gida: GidaV6 = GidaV6(**gida_config.as_dict())

    # gida_config.indexing = "dynamic"
    # dynamic_gida: GidaV6 = GidaV6(**gida_config.as_dict())

    # static_sample = next(iter(DataLoader(static_gida)))
    # dynamic_sample = next(iter(DataLoader(dynamic_gida)))

    # assert allclose(static_sample.x, dynamic_sample.x)

    batch_size = 1024
    number = 3

    gida_config.indexing = "static"
    static_execution_time = timeit.timeit(lambda: _iterate_over_a_set(batch_size, gida_config, False, False), number=number)
    static_execution_time = static_execution_time / number
    print(f"Average execution time: {static_execution_time:.6f} seconds")

    gida_config.indexing = "dynamic"
    dynamic_execution_time = timeit.timeit(lambda: _iterate_over_a_set(batch_size, gida_config, False, False), number=number)
    dynamic_execution_time = dynamic_execution_time / number
    print(f"Average execution time: {dynamic_execution_time:.6f} seconds")
    import matplotlib.pyplot as plt

    plt.bar(["static", "dynamic"], [static_execution_time, dynamic_execution_time])
    plt.xlabel("static vs. dynamic")
    plt.ylabel("Exe time (sec)")
    plt.title(
        f"#networks: {len(gida_config.zip_file_paths)}, batch_size: {batch_size}, num_records: {gida_config.num_records} ({gida_config.batch_axis_choice}), repeat in {number} times"
    )
    plt.show()


if __name__ == "__main__":
    test_static_and_dynamic_indexing("scene")
    # test_unequal_timelength("temporal")
    # test_unequal_timelength("snapshot")
    # test_benchmark_shuffle(batch_axis_choice="scene")
    # test_benchmark_shuffle(batch_axis_choice="snapshot")
    # test_benchmark_shuffle(batch_axis_choice="temporal")
    # test_subset_split(batch_axis_choice="scene")
    # test_subset_split(batch_axis_choice="snapshot")
    # test_subset_split(batch_axis_choice="temporal")
#     test_interleaving_on_multinets(batch_axis_choice="scene")
#     test_node_single_multiple_equal_nets(batch_axis_choice="temporal")
#     test_check_x_equal_y_multiple_equal_nets()
#     test_shuffle_multiple_nets("scene")
