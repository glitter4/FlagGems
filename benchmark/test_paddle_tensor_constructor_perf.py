import math
import random

import pytest
import torch

from benchmark.attri_util import BenchLevel
from benchmark.performance_utils import (
    Config,
    GenericBenchmark,
    generate_tensor_input,
    unary_input_fn,
    vendor_name,
)


def generic_constructor_input_fn(shape, dtype, device):
    yield {"size": shape, "dtype": dtype, "device": device},

# Define operations and their corresponding input functions
tensor_constructor_operations = [
    # generic tensor constructor
    
    ("ones", torch.ones, generic_constructor_input_fn),
    ("zeros", torch.zeros, generic_constructor_input_fn),
    # generic tensor-like constructor
    ("zeros_like", torch.zeros_like, unary_input_fn),
    

]


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(op, fn, input_fn, marks=getattr(pytest.mark, op, None))
        for op, fn, input_fn in tensor_constructor_operations
    ],
)
def test_tensor_constructor_benchmark(op_name, torch_op, input_fn):
    if vendor_name == "kunlunxin" and op_name in [
        "linspace",
    ]:
        pytest.skip("RUNTIME TODOFIX.")
    if vendor_name == "mthreads" and op_name == "logspace":
        pytest.skip("Torch MUSA Unsupported Now")
    bench = GenericBenchmark(input_fn=input_fn, op_name=op_name, torch_op=torch_op)
    bench.run()


@pytest.mark.skipif(
    vendor_name == "kunlunxin" or vendor_name == "hygon", reason="RESULT TODOFIX"
)
@pytest.mark.skipif(vendor_name == "mthreads", reason="RuntimeError")
@pytest.mark.randperm
def test_perf_randperm():
    def randperm_input_fn(shape, dtype, device):
        yield {"n": shape[0], "dtype": dtype, "device": device},

    bench = GenericBenchmark(
        input_fn=randperm_input_fn,
        op_name="randperm",
        torch_op=torch.randperm,
        dtypes=[torch.int32, torch.int64],
    )
    bench.run()
