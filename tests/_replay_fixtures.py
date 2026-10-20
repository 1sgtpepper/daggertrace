from __future__ import annotations

import copy
import json

from daggertrace.replay_v2 import _capture
from daggertrace.validation_v2_fixtures import _micrograph


def _write_canonical(path, value):
    path.write_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sync_hop_getitems(raw):
    nodes = {node["id"]: node for node in raw["backward"]["nodes"]}
    for result in nodes.values():
        if result["target"] != "operator.getitem":
            continue
        hop, key = result["args"]["tuple"]
        binding = nodes[hop["ref"]]["kwargs"]["bindings"].get(key)
        if binding is not None:
            alias_group = result["tensor"]["alias_group"]
            result["tensor"] = copy.deepcopy(nodes[binding["ref"]]["tensor"])
            result["tensor"]["alias_group"] = alias_group


def _replayable_micrograph(capture):
    raw = capture.to_dict()
    raw["diagnostics"] = {
        "callbacks": {"forward": 1, "backward": 1},
        "aot_autograd": {"ok": 1, "total": 1},
        "inputs": [
            {
                "shape": [2, 2],
                "stride": [2, 1],
                "storage_offset": 0,
                "dtype": "float32",
                "device": "cuda:0",
                "requires_grad": True,
            }
        ],
        "grad_output": {
            "shape": [2, 2],
            "stride": [2, 1],
            "storage_offset": 0,
            "dtype": "float32",
            "device": "cuda:0",
            "requires_grad": False,
        },
    }
    return _capture(raw, capture.label)


def _canonical_storage_root_case(target):
    contribution = "materialized" if target in {"aten.mul", "aten.sum"} else "none"
    if target == "aten.add":
        contribution = "direct"
    _, control = _micrograph(False, contribution, False, 0, target == "aten.empty")
    capture = control.to_dict()
    if target == "aten.clone":
        root = next(node for node in capture["backward"]["nodes"] if node["target"] == "aten.zeros_like")
        root["target"] = target
    else:
        root = next(
            node
            for node in capture["backward"]["nodes"]
            if node["target"] == ("aten.mul" if target == "aten.sum" else target)
        )
        if target == "aten.sum":
            root["target"] = target
    if target == "aten.sum":
        root["args"] = {"tuple": [{"ref": "n0"}, []]}
    return capture, root
