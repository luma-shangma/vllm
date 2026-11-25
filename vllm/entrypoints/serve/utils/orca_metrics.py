# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions that create ORCA endpoint load report response headers."""

import json
from collections.abc import Callable, Mapping

from vllm.logger import init_logger
from vllm.v1.metrics.reader import Gauge, get_metrics_snapshot

logger = init_logger(__name__)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


# Map from prometheus metric names to ORCA named metrics, together with the
# aggregation that folds per-engine samples into one endpoint value. Under
# data-parallel serving one API server fronts several engines and
# `get_metrics_snapshot()` returns one sample per engine, so the ORCA header
# must aggregate them: reporting only one engine's sample misleads load-aware
# gateways (see https://github.com/vllm-project/vllm/issues/61898).
_PROMETHEUS_TO_ORCA_METRICS: dict[str, tuple[str, Callable[[list[float]], float]]] = {
    # Fraction of the endpoint's KV cache in use. Engines behind one server
    # are equally sized, so the endpoint fraction is the mean across engines.
    "vllm:kv_cache_usage_perc": ("kv_cache_usage_perc", _mean),
    # Requests queue independently on every engine, so the endpoint's queue
    # depth is the sum across engines.
    "vllm:num_requests_waiting": ("num_requests_waiting", sum),
}


def create_orca_header(
    metrics_format: str, named_metrics: list[tuple[str, float]]
) -> Mapping[str, str] | None:
    """Creates ORCA headers named 'endpoint-load-metrics' in the specified format
    and adds custom metrics to named_metrics.
    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Args:
        metrics_format (str): The format of the header ('TEXT', 'JSON').
        named_metrics (List[Tuple[str, float]]): List of tuples with metric names
            and their corresponding double values.

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as
    'endpoint-load-metrics' and values as the ORCA header strings with
    format prefix and data in  with named_metrics in.

    """
    if metrics_format.lower() not in ["text", "json"]:
        logger.warning(
            "Warning: `%s` format is not supported in the ORCA response header",
            metrics_format,
        )
        return None

    header = {}
    orca_report = {
        "named_metrics": {
            metric_name: value
            for metric_name, value in named_metrics
            if isinstance(metric_name, str) and isinstance(value, float)
        }
    }
    # output example:
    # endpoint-load-metrics: TEXT named_metrics.kv_cache_utilization=0.4
    if metrics_format.lower() == "text":
        native_http_header = ", ".join(
            [
                f"named_metrics.{metric_name}={value}"
                for metric_name, value in named_metrics
                if isinstance(metric_name, str) and isinstance(value, float)
            ]
        )
        header["endpoint-load-metrics"] = f"TEXT {native_http_header}"

    # output example:
    # endpoint-load-metrics: JSON “named_metrics”: {“custom-metric-util”: 0.4}
    elif metrics_format.lower() == "json":
        header["endpoint-load-metrics"] = f"JSON {json.dumps(orca_report)}"

    logger.debug("Created ORCA header %s", header)

    return header


def get_named_metrics_from_prometheus() -> list[tuple[str, float]]:
    """Collects current metrics from Prometheus and returns some of them
    in the form of the `named_metrics` list for `create_orca_header()`.

    Each ORCA metric is reported once per endpoint: when several engines
    publish their own samples (data-parallel serving), the samples are
    aggregated per `_PROMETHEUS_TO_ORCA_METRICS` instead of letting the
    last engine's sample overwrite the others.

    Returns:
        List of tuples of metric names and their values.

    """
    # If this metric is mapped into ORCA, then collect its per-engine samples.
    # Note: Only Gauge metrics are currently supported.
    samples: dict[str, list[float]] = {}
    for metric in get_metrics_snapshot():
        if metric.name in _PROMETHEUS_TO_ORCA_METRICS and isinstance(metric, Gauge):
            samples.setdefault(metric.name, []).append(float(metric.value))
    # Aggregate each metric's samples into one endpoint-level value, in the
    # stable order of the mapping.
    named_metrics: list[tuple[str, float]] = []
    for prometheus_name, (orca_name, aggregate) in _PROMETHEUS_TO_ORCA_METRICS.items():
        values = samples.get(prometheus_name)
        if values:
            named_metrics.append((orca_name, float(aggregate(values))))
    return named_metrics


def metrics_header(metrics_format: str) -> Mapping[str, str] | None:
    """Creates ORCA headers named 'endpoint-load-metrics' in the specified format.
    Metrics are collected from Prometheus using `get_named_metrics_from_prometheus()`.

    ORCA headers format description: https://docs.google.com/document/d/1C1ybMmDKJIVlrbOLbywhu9iRYo4rilR-cT50OTtOFTs/edit?tab=t.0
    ORCA proto https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto

    Args:
        metrics_format (str): The format of the header ('TEXT', 'JSON').

    Returns:
    - Optional[Mapping[str,str]]: A dictionary with header key as
    'endpoint-load-metrics' and values as the ORCA header strings with
    format prefix and data in  with named_metrics in.

    """
    if not metrics_format:
        return None
    # Get named metrics from prometheus.
    named_metrics = get_named_metrics_from_prometheus()
    return create_orca_header(metrics_format, named_metrics)
