#  #
#  Created on Jul 08 2026
#  Copyright (c) 2026 Andrés Tello
#  ------------------------------
#  Purpose: Check the balance between flow and demand.
#  For n in NODES:
#    incoming_flow - outgoing_flow - demand = 0
#
#  ------------------------------

import glob
import zarr
import os
import wntr
from torch import tensor, argsort
from wntr.network import WaterNetworkModel


def build_adj_from_input(wnm: WaterNetworkModel) -> dict:
    """

    Args:
        wnm: WNTR WaterNetworkModel of the network

    Returns: A dictionary with the following keys:
        edge_index: edge_index a Tensor[2, num_edges] in pyg format. It is unsorted.
        edge_index_sorted: sorted edge_index in pyg format. It is sorted by source_id
        adj_dict: dictionary of the original adjacency matrix taken from input file: edge_id: (src, dst)
        adj_list: dictionary of the original adjacency matrix taken from input file: (src_node_name, dst_node_name, link_name)
        sort_idx: Sort index. It has to be used to sort all edge-related attributes the same way (e.g. flow, diameter, length, velocity, etc....)
        nodes_original_pyg_dict: dictionary in the form {original_node_name: pyg_index}
        nodes_pyg_original_dict: dictionary in the form {pyg_index: original_node_name}

    """
    nodes_original_pyg_dict = {name: i for i, name in enumerate(wnm.node_name_list)}
    nodes_pyg_original_dict = {v: k for k, v in nodes_original_pyg_dict.items()}

    adj_dict = {link_id: (str(link.start_node), str(link.end_node)) for link_id, link in wnm.links.items()}
    adj_list: list[tuple[str, str, str]] = [(start_node, end_node, str(link_id)) for link_id, (start_node, end_node) in adj_dict.items()]
    edge_index = tensor([[nodes_original_pyg_dict[str(l.start_node)], nodes_original_pyg_dict[str(l.end_node)]] for n, l in wnm.links.items()])
    edge_index = edge_index.transpose(1, 0)

    sort_idx = argsort(edge_index[0], descending=False)
    edge_index_sorted = edge_index[:, sort_idx]

    output_dict = {
        "edge_index": edge_index,
        "edge_index_sorted": edge_index_sorted,
        "adj_dict": adj_dict,
        "adj_list": adj_list,
        "sort_idx": sort_idx,
        "nodes_original_pyg_dict": nodes_original_pyg_dict,
        "nodes_pyg_original_dict": nodes_pyg_original_dict,
    }

    return output_dict


def update_adjacency(
    zarr_root: str = r"G:/My Drive/Dataset/from_habrok",
    inp_root: str = r"G:/Other computers/My Laptop/PhD/Codebase/DiTEC_WDN_dataset/ditec_wdn_dataset/inputs/public",
    dry_run: bool = False,
) -> None:
    """Given the zarr, this fuction rebuilds the adjacency matrix from the corresponding INP, then
    Args:
        zarr_root (_type_, optional): _description_. Defaults to r"G:/My Drive/Dataset/from_habrok".
        inp_root (_type_, optional): _description_. Defaults to r"G:/Other computers/My Laptop/PhD/Codebase/DiTEC_WDN_dataset/ditec_wdn_dataset/inputs/public".
        debug (int, optional): _description_. Defaults to 0.
    """
    mode = "r" if dry_run else "a"
    zarr_file_paths = glob.glob(f"{zarr_root}/*.zip")
    for zarr_path in zarr_file_paths:
        print(f"UPDATING {zarr_path}...")
        try:
            root = zarr.open(zarr_path, mode=mode)
        except Exception as e:
            print(f"Zarr path {zarr_path} is not a zip file or a bad zip! Skip...")
            continue

        # skip_nodes = root.attrs["skip_names"]
        inp_file_name = root.attrs["inp_paths"][0].split("/")[-1]
        if inp_file_name == "EXN2.inp":
            inp_file_name = "EXN.inp"
        inp_path = f"{inp_root}/{inp_file_name}"
        print(f"CHECKING {inp_path}...")

        if not os.path.exists(inp_path):
            print(f"Input path {inp_path} does not exist! Skip...")
            continue

        wn = wntr.network.WaterNetworkModel(inp_path)
        topology_dict = build_adj_from_input(wn)
        assert "adj_list" in topology_dict

        root.attrs["adj_list"] = topology_dict["adj_list"]


if __name__ == "__main__":
    wn = wntr.network.WaterNetworkModel(r"G:\Other computers\My Laptop\PhD\Codebase\DiTEC_WDN_dataset\ditec_wdn_dataset\inputs\public\Anytown.inp")
    build_adj_from_input(wn)
