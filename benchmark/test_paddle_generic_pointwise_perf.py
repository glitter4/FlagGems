import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES, INT_DTYPES
from benchmark.conftest import BenchLevel, Config
from benchmark.performance_utils import (
    GenericBenchmark,
    GenericBenchmarkExcluse1D,
    generate_tensor_input,
    unary_input_fn,
    vendor_name,
)

@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, dtypes",
    [
        pytest.param(
            "triu",
            torch.triu,
            unary_input_fn,
            FLOAT_DTYPES,
            marks=pytest.mark.triu,
        ),
    ],
)
def test_generic_pointwise_benchmark_exclude_1d(op_name, torch_op, input_fn, dtypes):
    bench = GenericBenchmarkExcluse1D(
        input_fn=input_fn, op_name=op_name, torch_op=torch_op, dtypes=dtypes
    )
    bench.run()
