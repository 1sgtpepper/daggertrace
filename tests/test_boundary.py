from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from daggertrace.capture import capture_aot
from daggertrace.errors import UnsupportedCapture
from daggertrace.validation_v2_fixtures import _micrograph
from daggertrace.version import CAPTURE_SCHEMA


def test_capture_aot_returns_only_admitted_v2_models(monkeypatch):
    import daggertrace.capture as capture_module

    fixture, _ = _micrograph(False, "none", False, 1, False)
    graphs = {"forward": fixture.forward, "backward": fixture.backward}
    fake_torch = ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.__version__ = "2.12.1+cu130"
    fake_torch.version = SimpleNamespace(
        git_version="7269437d655783a26cba32aa88195b741ff496aa",
        cuda="13.0",
    )
    fake_dynamo = ModuleType("torch._dynamo")
    fake_dynamo.__path__ = []
    fake_utils = ModuleType("torch._dynamo.utils")
    fake_utils.counters = {"aot_autograd": {}}
    fake_dynamo.utils = fake_utils
    fake_dynamo.reset = lambda: None
    fake_exc = ModuleType("torch._dynamo.exc")
    fake_exc.Unsupported = RuntimeError
    fake_backends = ModuleType("torch._dynamo.backends")
    fake_backends.__path__ = []
    fake_common = ModuleType("torch._dynamo.backends.common")

    def fake_aot_autograd(*, fw_compiler, bw_compiler):
        return SimpleNamespace(fw_compiler=fw_compiler, bw_compiler=bw_compiler)

    fake_common.aot_autograd = fake_aot_autograd
    fake_dynamo.backends = fake_backends
    fake_backends.common = fake_common
    fake_dynamo.exc = fake_exc
    fake_torch._dynamo = fake_dynamo
    fake_torch.autograd = SimpleNamespace(grad=lambda *args, **kwargs: None)

    class FakeGraphModule:
        def forward(self, *args, **kwargs):
            return None

    def fake_compile(function, *, fullgraph, dynamic, backend):
        assert fullgraph is True
        assert dynamic is False

        def compiled(*args):
            backend.fw_compiler(FakeGraphModule(), args)
            backend.bw_compiler(FakeGraphModule(), args)
            return "fake-output"

        return compiled

    fake_torch.compile = fake_compile
    fake_triton = ModuleType("triton")
    fake_triton.__version__ = "3.7.1"
    fake_functorch = ModuleType("functorch")
    fake_functorch.__path__ = []
    fake_compile_module = ModuleType("functorch.compile")
    fake_compile_module.make_boxed_func = lambda forward: forward
    fake_functorch.compile = fake_compile_module
    for name, module in {
        "torch": fake_torch,
        "torch._dynamo": fake_dynamo,
        "torch._dynamo.utils": fake_utils,
        "torch._dynamo.exc": fake_exc,
        "torch._dynamo.backends": fake_backends,
        "torch._dynamo.backends.common": fake_common,
        "triton": fake_triton,
        "functorch": fake_functorch,
        "functorch.compile": fake_compile_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        capture_module,
        "_normalize_graph",
        lambda _graph_module, kind, _kernels: graphs[kind],
    )
    monkeypatch.setattr(
        capture_module,
        "_runtime_tensor_meta",
        lambda _value: {"synthetic": True},
    )
    result = capture_aot(
        lambda value: value,
        [SimpleNamespace(requires_grad=True)],
        object(),
        label="fake-v2",
    )
    assert result.schema == CAPTURE_SCHEMA
    assert result.to_dict()["schema"] == CAPTURE_SCHEMA
    assert result.capture_config == {
        "fullgraph": True,
        "dynamic": False,
        "stage": "post-dynamo-pre-inductor",
    }

    for label in ("", "\ud800"):
        with pytest.raises(UnsupportedCapture):
            capture_aot(
                lambda value: value,
                [SimpleNamespace(requires_grad=True)],
                object(),
                label=label,
            )
