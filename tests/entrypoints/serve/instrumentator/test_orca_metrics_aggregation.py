# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for ORCA endpoint-load-metrics aggregation.

Regression tests for https://github.com/vllm-project/vllm/issues/61898:
under data parallel, one API server fronts several engines and every engine
publishes its own gauge sample. The ORCA header must aggregate those samples
into one endpoint-level value instead of reporting whichever engine's sample
happens to come last.
"""

import json

import pytest

from vllm.entrypoints.serve.utils import orca_metrics
from vllm.v1.metrics.reader import Counter, Gauge


def _two_engine_snapshot():
    """Engine 0 is heavily loaded, engine 1 is idle."""
    return [
        Gauge(
            name="vllm:kv_cache_usage_perc",
            labels={"model_name": "m", "engine": "0"},
            value=0.95,
        ),
        Gauge(
            name="vllm:num_requests_waiting",
            labels={"model_name": "m", "engine": "0"},
            value=40.0,
        ),
        Gauge(
            name="vllm:kv_cache_usage_perc",
            labels={"model_name": "m", "engine": "1"},
            value=0.05,
        ),
        Gauge(
            name="vllm:num_requests_waiting",
            labels={"model_name": "m", "engine": "1"},
            value=0.0,
        ),
    ]


@pytest.fixture
def two_engine_metrics(monkeypatch):
    monkeypatch.setattr(orca_metrics, "get_metrics_snapshot", _two_engine_snapshot)


def test_kv_usage_is_averaged_and_waiting_is_summed(two_engine_metrics):
    named_metrics = orca_metrics.get_named_metrics_from_prometheus()
    assert named_metrics == [
        ("kv_cache_usage_perc", 0.5),
        ("num_requests_waiting", 40.0),
    ]


def test_json_header_reports_endpoint_load(two_engine_metrics):
    header = orca_metrics.metrics_header("json")
    value = header["endpoint-load-metrics"]
    assert value.startswith("JSON ")
    assert json.loads(value.removeprefix("JSON ")) == {
        "named_metrics": {
            "kv_cache_usage_perc": 0.5,
            "num_requests_waiting": 40.0,
        }
    }


def test_text_header_reports_each_metric_once(two_engine_metrics):
    header = orca_metrics.metrics_header("text")
    assert header["endpoint-load-metrics"] == (
        "TEXT named_metrics.kv_cache_usage_perc=0.5, "
        "named_metrics.num_requests_waiting=40.0"
    )


def test_single_engine_values_pass_through(monkeypatch):
    monkeypatch.setattr(
        orca_metrics,
        "get_metrics_snapshot",
        lambda: [
            Gauge(
                name="vllm:kv_cache_usage_perc",
                labels={"model_name": "m", "engine": "0"},
                value=0.25,
            ),
            Gauge(
                name="vllm:num_requests_waiting",
                labels={"model_name": "m", "engine": "0"},
                value=3.0,
            ),
        ],
    )
    assert orca_metrics.get_named_metrics_from_prometheus() == [
        ("kv_cache_usage_perc", 0.25),
        ("num_requests_waiting", 3.0),
    ]


def test_unmapped_and_non_gauge_metrics_are_ignored(monkeypatch):
    monkeypatch.setattr(
        orca_metrics,
        "get_metrics_snapshot",
        lambda: [
            Gauge(
                name="vllm:num_requests_running",
                labels={"engine": "0"},
                value=7.0,
            ),
            Counter(
                name="vllm:num_requests_waiting",
                labels={"engine": "0"},
                value=11,
            ),
        ],
    )
    assert orca_metrics.get_named_metrics_from_prometheus() == []


def test_unsupported_format_warns_with_the_requested_format(monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(
        orca_metrics.logger,
        "warning",
        lambda msg, *args: messages.append(msg % args),
    )
    assert orca_metrics.create_orca_header("xml", []) is None
    assert messages and "xml" in messages[0]
