import random

import numpy as np
import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import (
    GenericBenchmark,
    GenericBenchmark2DOnly,
    generate_tensor_input,
    vendor_name,
)
from flag_gems.utils import shape_utils


class TensorSelectBenchmark(GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        if (
            vendor_name == "kunlunxin"
        ):  # Speed Up Benchmark Test, Big Shape Will Cause Timeout
            return []
        shapes = super().set_more_shapes()
        shapes = [
            # this filter is for scatter
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


def index_select_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    threshold = 0.1
    dim = 0
    index_size = inp.size(dim)
    from math import floor

    index = torch.randint(0, index_size, [floor(index_size * threshold)]).to(device)
    yield inp, dim, index


def index_select_gbps(bench_fn_args, latency):
    inp = bench_fn_args[0]
    dim = bench_fn_args[1]
    io_amount = shape_utils.size_in_bytes(inp) * 2 // inp.size(dim)
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.index_select
@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, gbps_fn, dtypes",
    [
        pytest.param(
            "index_select",
            torch.index_select,
            index_select_input_fn,
            index_select_gbps,
            FLOAT_DTYPES,
            marks=pytest.mark.index_select,
        ),
    ],
)
def test_perf_index_select(op_name, torch_op, input_fn, gbps_fn, dtypes):
    bench = TensorSelectBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=dtypes,
        get_gbps=gbps_fn,
    )
    bench.run()

