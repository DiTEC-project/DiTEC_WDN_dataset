#  #
#  Created on Jul 08 2026
#  Copyright (c) 2026 Andrés Tello
#  ------------------------------
#  Purpose: Check the balance between flow and demand.
#  For n in NODES:
#    incoming_flow - outgoing_flow - demand = 0
#  Modified func name
#  ------------------------------
from collections import defaultdict
import glob
import os
import wntr
import zarr
from torch import Tensor, as_tensor, zeros, where, abs
import numpy as np
from ditec_wdn_dataset.utils.adj_builder import build_adj_from_input
from ditec_wdn_dataset.utils.auxil_v8 import setup_logger
import json
import matplotlib.pyplot as plt


def check_node_mass_balance(
    edge_index: Tensor,
    flow: Tensor,
    demand: Tensor,
    num_nodes: int | None = None,
    atol: float = 1e-5,
):
    """
    Check hydraulic mass balance at each node:

        sum(inflows) - sum(outflows) - demand = 0

    Parameters
    ----------
    edge_index : torch.Tensor
        Shape [2, E]. edge_index[0] = src, edge_index[1] = dst.

    flow : torch.Tensor
        Shape [E] or [B, E].
        Signed edge flow. Positive means src -> dst.

    demand : torch.Tensor
        Shape [N] or [B, N].
        Positive demand means consumption at the node.

    num_nodes : int, optional
        Number of nodes. If None, inferred from edge_index.

    atol : float
        Absolute tolerance for checking residuals.

    Returns
    -------
    dict
        {
            "ok": bool,
            "residual": torch.Tensor,
            "max_abs_residual": float,
            "bad_nodes_pyg": torch.Tensor  (Nodes in pyg index format)
        }
    """

    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    src = edge_index[0].long()
    dst = edge_index[1].long()

    # merged = torch.vstack([src, dst, flow])

    flow = as_tensor(flow)
    demand = as_tensor(demand, device=flow.device, dtype=flow.dtype)

    if edge_index.device != flow.device:
        src = src.to(flow.device)
        dst = dst.to(flow.device)

    # Case 1: single snapshot
    if flow.ndim == 1:
        net_flow = zeros(num_nodes, device=flow.device, dtype=flow.dtype)

        # Incoming flow adds to node balance
        net_flow.index_add_(0, dst, flow)

        # Outgoing flow subtracts from node balance
        net_flow.index_add_(0, src, -flow)

        residual = net_flow - demand

        bad_nodes = where(abs(residual) > atol)[0]

    # Case 2: multiple snapshots, shape [B, E]
    elif flow.ndim == 2:
        B = flow.shape[0]
        net_flow = zeros(B, num_nodes, device=flow.device, dtype=flow.dtype)

        # Incoming
        net_flow.index_add_(1, dst, flow)

        # Outgoing
        net_flow.index_add_(1, src, -flow)

        residual = net_flow - demand

        bad_nodes = where(abs(residual) > atol)

    else:
        raise ValueError("flow must have shape [E] or [B, E].")

    max_abs_residual = abs(residual).max().item()
    ok = max_abs_residual <= atol

    return {
        "ok": ok,
        "residual": residual,
        "max_abs_residual": max_abs_residual,
        "bad_nodes_pyg": bad_nodes,
    }


def check_all_zarr_and_inp(
    zarr_root: str = r"G:/My Drive/Dataset/from_habrok",
    inp_root: str = r"G:/Other computers/My Laptop/PhD/Codebase/DiTEC_WDN_dataset/ditec_wdn_dataset/inputs/public",
    debug: int = 0,
) -> None:
    atol = 1e-2
    save_folder = "ditec_wdn_dataset/outputs"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # log_json_path = f"{save_folder}/check_all_zarr_and_inp_log.json"

    # logger = setup_logger("stdout_logger", log_path)
    # ZARR_PATH = "/home/andres/Dropbox/PhD Smart Environments - RUG/ExternalProjects/WDN_datasets/datasets/data_v3/24hours/simgen_ky8_20241120_230_20241104_175.zip"
    # INPUT_FILE = "/home/andres/Dropbox/PostdocRUG/Projects/GFM4WDN_v2/ditec_wdn_dataset/inputs/public/ky8.inp"

    # ZARR_ROOT_PATH = r"/G:\My Drive\Dataset\from_habrok"
    # INPUT_FILE_ROOT_PATH = r

    zarr_file_paths = glob.glob(f"{zarr_root}/*.zip")

    for zarr_path in zarr_file_paths:
        print(f"CHECKING {zarr_path}...")
        try:
            root = zarr.open(zarr_path, mode="r")
        except Exception as e:
            print(f"Zarr path {zarr_path} is not a zip file or a bad zip! Skip...")
            continue

        # tmps = os.path.basename(zarr_path).split("_")
        # print(f"tmps = {tmps}")
        # if len(tmps) > 4:
        #     tmps.pop(0)
        #     tmps.pop(-1)
        #     inp_file_name = " ".join(tmps)
        # else:
        #     inp_file_name = tmps[1]

        skip_nodes = root.attrs["skip_names"]
        inp_file_name = root.attrs["inp_paths"][0].split("/")[-1]
        if inp_file_name == "EXN2.inp":
            inp_file_name = "EXN.inp"
        inp_path = f"{inp_root}/{inp_file_name}"
        print(f"CHECKING {inp_path}...")

        if not os.path.exists(inp_path):
            print(f"Input path {inp_path} does not exist! Skip...")
            continue

        wn = wntr.network.WaterNetworkModel(inp_path)
        # adj = build_adj_from_input(wn)

        time_steps = root.attrs["duration"]

        demand = np.array(root["demand"]).reshape(1000, time_steps, -1)
        flow = np.array(root["flowrate"]).reshape(1000, time_steps, -1)

        topology_dict = build_adj_from_input(wn)
        nodes_pyg_original_dict = topology_dict["nodes_pyg_original_dict"]

        errors = 0
        print(f"WDN: {inp_file_name}")
        uni_bad_nodes = []
        bad_scenario_snapshots: dict[int, list[int]] = {}
        orgnode_badsnapshotcount_dict = defaultdict(int)
        for scenario_idx in range(flow.shape[0]):
            tmp_list = []
            for snapshot_id in range(flow[0].shape[0]):
                result = check_node_mass_balance(
                    edge_index=topology_dict["edge_index"],
                    flow=as_tensor(flow[scenario_idx][snapshot_id]),
                    demand=as_tensor(demand[scenario_idx][snapshot_id]),
                    atol=atol,
                )

                if len(result["bad_nodes_pyg"]) > 0:
                    bad_nodes = []
                    for node in result["bad_nodes_pyg"]:
                        orgnode_badsnapshotcount_dict[nodes_pyg_original_dict[node.item()]] += 1
                        bad_nodes.append(f"(org: {nodes_pyg_original_dict[node.item()]}, pyg: {node.item()})")
                    # bad_nodes = [f"(org: {nodes_pyg_original_dict[node.item()]}, pyg: {node.item()})" for node in result["bad_nodes_pyg"]]
                    if debug > 2:
                        print(f"scenario: {scenario_idx} \tsnapshot: {snapshot_id} \tbad_nodes: {bad_nodes}")
                    errors += 1

                    if len(uni_bad_nodes) <= 0:
                        uni_bad_nodes = bad_nodes
                    else:
                        uni_bad_nodes = list(set(uni_bad_nodes).union(bad_nodes))
                    tmp_list.append(snapshot_id)
            if len(tmp_list) > 0:
                bad_scenario_snapshots[scenario_idx] = tmp_list

        wdn_dict = {}
        wdn_dict["bad_nodes"] = uni_bad_nodes
        wdn_dict["skip_nodes"] = skip_nodes
        wdn_dict["zarr_path"] = zarr_path
        wdn_dict["bad_scenario_snapshots"] = bad_scenario_snapshots
        wdn_dict["atol"] = atol

        keys = list(orgnode_badsnapshotcount_dict.keys())
        values = list(orgnode_badsnapshotcount_dict.values())

        plt.bar(keys, values)  # bars at each element name
        plt.xlabel("Original node name")
        plt.ylabel("No of bad snapshots")
        plt.title(f"Bad snapshot count per node in network {inp_file_name[:-4]}")

        plt.savefig(f"{save_folder}/element_counts_{inp_file_name[:-4]}.png", dpi=300, bbox_inches="tight")

        with open(f"{save_folder}/check_log_{inp_file_name[:-4]}.json", "w") as f:
            json.dump(wdn_dict, f, indent=4)

        if debug > 0:
            plt.show()
            top_10 = sorted(orgnode_badsnapshotcount_dict.items(), key=lambda kv: kv[1], reverse=True)[:10]

            print("Top 10 nodes that have the largest number of bad snapshots: ")
            for name, count in top_10:
                print(name, "\t:\t", count)

        print(f"\ntotal bad snapshots: {errors} \t unique bad nodes={uni_bad_nodes}")
        print("*" * 80)
