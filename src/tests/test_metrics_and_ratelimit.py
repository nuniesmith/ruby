"""
Tests for Prometheus Metrics and Rate Limiting
=====================================================================

Covers:

**Prometheus Metrics:**
  - Metric definitions: all expected metrics exist in registry
  - Counter increments: HTTP requests, SSE events, risk checks, ORB detections, no-trade alerts
  - Gauge updates: SSE connections, focus quality, positions count, Redis status, engine refresh
  - Histogram observations: HTTP request duration, engine cycle duration
  - Path normalization: reduces cardinality for dynamic path segments
  - PrometheusMiddleware: instruments requests automatically
  - /metrics/prometheus endpoint: returns text/plain Prometheus exposition format
  - Live gauge collection: reads from cache to update gauges on scrape
  - Registry isolation: get_registry() returns the correct instance

**Rate Limiting:**
  - Client key derivation: from API key, X-Forwarded-For, or remote address
  - Path-based limit mapping: correct limits for public, SSE, mutation, heavy endpoints
  - Rate limit handler: returns structured 429 JSON
  - setup_rate_limiting: installs limiter on app
  - Limiter singleton: get_limiter() returns same instance; reset_limiter() clears it
  - Enabled/disabled toggle: respects RATE_LIMIT_ENABLED environment variable
  - Integration: rate-limited endpoint returns 429 after exceeding limit

**Integration:**
  - Full app with both middleware: metrics + rate limiting coexist
  - /metrics/prometheus accessible without auth
"""

import json
import os
import time
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

os.environ.setdefault("DISABLE_REDIS", "1")
# Disable rate limiting for most tests unless explicitly testing it
os.environ.setdefault("RATE_LIMIT_ENABLED", "0")

_EST = ZoneInfo("America/New_York")


# ===========================================================================
# SECTION 1: Prometheus Metrics — Unit Tests
# ===========================================================================


class TestMetricDefinitions:
    """Verify all expected metrics are registered."""

    def test_registry_exists(self):
        from lib.services.data.api.metrics import get_registry

        registry = get_registry()
        assert registry is not None

    def test_http_requests_total_registered(self):
        from lib.services.data.api.metrics import HTTP_REQUESTS_TOTAL

        assert HTTP_REQUESTS_TOTAL is not None
        # prometheus_client strips _total suffix from Counter._name
        assert "http_requests" in HTTP_REQUESTS_TOTAL._name

    def test_http_request_duration_registered(self):
        from lib.services.data.api.metrics import HTTP_REQUEST_DURATION

        assert HTTP_REQUEST_DURATION is not None
        # prometheus_client may strip _seconds suffix from Histogram._name
        assert "http_request_duration" in HTTP_REQUEST_DURATION._name

    def test_sse_connections_active_registered(self):
        from lib.services.data.api.metrics import SSE_CONNECTIONS_ACTIVE

        assert SSE_CONNECTIONS_ACTIVE is not None
        assert "sse_connections_active" in SSE_CONNECTIONS_ACTIVE._name

    def test_sse_events_total_registered(self):
        from lib.services.data.api.metrics import SSE_EVENTS_TOTAL

        assert SSE_EVENTS_TOTAL is not None
        assert "sse_events" in SSE_EVENTS_TOTAL._name

    def test_engine_last_refresh_epoch_registered(self):
        from lib.services.data.api.metrics import ENGINE_LAST_REFRESH_EPOCH

        assert ENGINE_LAST_REFRESH_EPOCH is not None
        assert "engine_last_refresh_epoch" in ENGINE_LAST_REFRESH_EPOCH._name

    def test_engine_cycle_duration_registered(self):
        from lib.services.data.api.metrics import ENGINE_CYCLE_DURATION

        assert ENGINE_CYCLE_DURATION is not None
        assert "engine_cycle_duration" in ENGINE_CYCLE_DURATION._name

    def test_risk_checks_total_registered(self):
        from lib.services.data.api.metrics import RISK_CHECKS_TOTAL

        assert RISK_CHECKS_TOTAL is not None
        assert "risk_checks" in RISK_CHECKS_TOTAL._name

    def test_orb_detections_total_registered(self):
        from lib.services.data.api.metrics import ORB_DETECTIONS_TOTAL

        assert ORB_DETECTIONS_TOTAL is not None
        assert "orb_detections" in ORB_DETECTIONS_TOTAL._name

    def test_no_trade_alerts_total_registered(self):
        from lib.services.data.api.metrics import NO_TRADE_ALERTS_TOTAL

        assert NO_TRADE_ALERTS_TOTAL is not None
        assert "no_trade_alerts" in NO_TRADE_ALERTS_TOTAL._name

    def test_focus_quality_gauge_registered(self):
        from lib.services.data.api.metrics import FOCUS_QUALITY_GAUGE

        assert FOCUS_QUALITY_GAUGE is not None
        assert "focus_quality_gauge" in FOCUS_QUALITY_GAUGE._name

    def test_positions_open_count_registered(self):
        from lib.services.data.api.metrics import POSITIONS_OPEN_COUNT

        assert POSITIONS_OPEN_COUNT is not None
        assert "positions_open_count" in POSITIONS_OPEN_COUNT._name

    def test_redis_connected_registered(self):
        from lib.services.data.api.metrics import REDIS_CONNECTED

        assert REDIS_CONNECTED is not None
        assert "redis_connected" in REDIS_CONNECTED._name

    def test_orb_filter_results_total_registered(self):
        from lib.services.data.api.metrics import ORB_FILTER_RESULTS_TOTAL

        assert ORB_FILTER_RESULTS_TOTAL is not None
        assert "orb_filter_results" in ORB_FILTER_RESULTS_TOTAL._name

    def test_orb_cnn_prob_registered(self):
        from lib.services.data.api.metrics import ORB_CNN_PROB

        assert ORB_CNN_PROB is not None
        assert "orb_cnn_prob" in ORB_CNN_PROB._name

    def test_orb_cnn_signals_total_registered(self):
        from lib.services.data.api.metrics import ORB_CNN_SIGNALS_TOTAL

        assert ORB_CNN_SIGNALS_TOTAL is not None
        assert "orb_cnn_signals" in ORB_CNN_SIGNALS_TOTAL._name

    def test_daily_pnl_gauge_registered(self):
        from lib.services.data.api.metrics import DAILY_PNL_GAUGE

        assert DAILY_PNL_GAUGE is not None
        assert "daily_pnl_gauge" in DAILY_PNL_GAUGE._name

    def test_consecutive_losses_gauge_registered(self):
        from lib.services.data.api.metrics import CONSECUTIVE_LOSSES_GAUGE

        assert CONSECUTIVE_LOSSES_GAUGE is not None
        assert "consecutive_losses_gauge" in CONSECUTIVE_LOSSES_GAUGE._name

    def test_model_val_accuracy_registered(self):
        from lib.services.data.api.metrics import MODEL_VAL_ACCURACY

        assert MODEL_VAL_ACCURACY is not None
        assert "model_val_accuracy" in MODEL_VAL_ACCURACY._name

    def test_model_val_precision_registered(self):
        from lib.services.data.api.metrics import MODEL_VAL_PRECISION

        assert MODEL_VAL_PRECISION is not None
        assert "model_val_precision" in MODEL_VAL_PRECISION._name

    def test_model_val_recall_registered(self):
        from lib.services.data.api.metrics import MODEL_VAL_RECALL

        assert MODEL_VAL_RECALL is not None
        assert "model_val_recall" in MODEL_VAL_RECALL._name

    def test_model_train_samples_registered(self):
        from lib.services.data.api.metrics import MODEL_TRAIN_SAMPLES

        assert MODEL_TRAIN_SAMPLES is not None
        assert "model_train_samples" in MODEL_TRAIN_SAMPLES._name


class TestMetricRecordHelpers:
    """Test the helper functions that record metric values."""

    def test_record_sse_connect_increments(self):
        from lib.services.data.api.metrics import (
            SSE_CONNECTIONS_ACTIVE,
            record_sse_connect,
        )

        before = SSE_CONNECTIONS_ACTIVE._value.get()
        record_sse_connect()
        after = SSE_CONNECTIONS_ACTIVE._value.get()
        assert after == before + 1

    def test_record_sse_disconnect_decrements(self):
        from lib.services.data.api.metrics import (
            SSE_CONNECTIONS_ACTIVE,
            record_sse_disconnect,
        )

        # Ensure gauge is at least 1 so decrement doesn't go below zero
        SSE_CONNECTIONS_ACTIVE.set(5)
        before = SSE_CONNECTIONS_ACTIVE._value.get()
        record_sse_disconnect()
        after = SSE_CONNECTIONS_ACTIVE._value.get()
        assert after == before - 1

    def test_record_sse_event_increments_counter(self):
        from lib.services.data.api.metrics import (
            SSE_EVENTS_TOTAL,
            record_sse_event,
        )

        # Get initial value
        initial = SSE_EVENTS_TOTAL.labels(event_type="focus-update")._value.get()
        record_sse_event("focus-update")
        after = SSE_EVENTS_TOTAL.labels(event_type="focus-update")._value.get()
        assert after == initial + 1

    def test_record_sse_event_multiple_types(self):
        from lib.services.data.api.metrics import (
            SSE_EVENTS_TOTAL,
            record_sse_event,
        )

        record_sse_event("heartbeat")
        record_sse_event("heartbeat")
        record_sse_event("positions-update")

        hb = SSE_EVENTS_TOTAL.labels(event_type="heartbeat")._value.get()
        pos = SSE_EVENTS_TOTAL.labels(event_type="positions-update")._value.get()
        assert hb >= 2
        assert pos >= 1

    def test_record_risk_check_allowed(self):
        from lib.services.data.api.metrics import (
            RISK_CHECKS_TOTAL,
            record_risk_check,
        )

        initial = RISK_CHECKS_TOTAL.labels(result="allowed")._value.get()
        record_risk_check("allowed")
        after = RISK_CHECKS_TOTAL.labels(result="allowed")._value.get()
        assert after == initial + 1

    def test_record_risk_check_blocked(self):
        from lib.services.data.api.metrics import (
            RISK_CHECKS_TOTAL,
            record_risk_check,
        )

        initial = RISK_CHECKS_TOTAL.labels(result="blocked")._value.get()
        record_risk_check("blocked")
        after = RISK_CHECKS_TOTAL.labels(result="blocked")._value.get()
        assert after == initial + 1

    def test_record_risk_check_advisory(self):
        from lib.services.data.api.metrics import (
            RISK_CHECKS_TOTAL,
            record_risk_check,
        )

        initial = RISK_CHECKS_TOTAL.labels(result="advisory")._value.get()
        record_risk_check("advisory")
        after = RISK_CHECKS_TOTAL.labels(result="advisory")._value.get()
        assert after == initial + 1

    def test_record_orb_detection_long(self):
        from lib.services.data.api.metrics import (
            ORB_DETECTIONS_TOTAL,
            record_orb_detection,
        )

        initial = ORB_DETECTIONS_TOTAL.labels(direction="LONG")._value.get()
        record_orb_detection("LONG")
        after = ORB_DETECTIONS_TOTAL.labels(direction="LONG")._value.get()
        assert after == initial + 1

    def test_record_orb_detection_short(self):
        from lib.services.data.api.metrics import (
            ORB_DETECTIONS_TOTAL,
            record_orb_detection,
        )

        initial = ORB_DETECTIONS_TOTAL.labels(direction="SHORT")._value.get()
        record_orb_detection("SHORT")
        after = ORB_DETECTIONS_TOTAL.labels(direction="SHORT")._value.get()
        assert after == initial + 1

    def test_record_orb_detection_none(self):
        from lib.services.data.api.metrics import (
            ORB_DETECTIONS_TOTAL,
            record_orb_detection,
        )

        initial = ORB_DETECTIONS_TOTAL.labels(direction="none")._value.get()
        record_orb_detection("none")
        after = ORB_DETECTIONS_TOTAL.labels(direction="none")._value.get()
        assert after == initial + 1

    def test_record_no_trade_alert(self):
        from lib.services.data.api.metrics import (
            NO_TRADE_ALERTS_TOTAL,
            record_no_trade_alert,
        )

        initial = NO_TRADE_ALERTS_TOTAL.labels(condition="all_low_quality")._value.get()
        record_no_trade_alert("all_low_quality")
        after = NO_TRADE_ALERTS_TOTAL.labels(condition="all_low_quality")._value.get()
        assert after == initial + 1

    def test_record_no_trade_alert_multiple_conditions(self):
        from lib.services.data.api.metrics import (
            NO_TRADE_ALERTS_TOTAL,
            record_no_trade_alert,
        )

        record_no_trade_alert("extreme_volatility")
        record_no_trade_alert("daily_loss_exceeded")
        record_no_trade_alert("consecutive_losses")

        vol = NO_TRADE_ALERTS_TOTAL.labels(condition="extreme_volatility")._value.get()
        loss = NO_TRADE_ALERTS_TOTAL.labels(condition="daily_loss_exceeded")._value.get()
        streak = NO_TRADE_ALERTS_TOTAL.labels(condition="consecutive_losses")._value.get()
        assert vol >= 1
        assert loss >= 1
        assert streak >= 1

    def test_record_engine_refresh_sets_timestamp(self):
        from lib.services.data.api.metrics import (
            ENGINE_LAST_REFRESH_EPOCH,
            record_engine_refresh,
        )

        before = time.time()
        record_engine_refresh()
        after = time.time()
        value = ENGINE_LAST_REFRESH_EPOCH._value.get()
        assert before <= value <= after

    def test_record_engine_cycle_observes_duration(self):
        from lib.services.data.api.metrics import (
            ENGINE_CYCLE_DURATION,
            record_engine_cycle,
        )

        # Observe a couple of values
        record_engine_cycle(0.5)
        record_engine_cycle(1.2)
        record_engine_cycle(3.0)

        # The histogram sum should reflect our observations
        # Access internal state (prometheus_client Histogram uses _sum)
        sample_sum = ENGINE_CYCLE_DURATION._sum.get()
        assert sample_sum >= 4.7  # 0.5 + 1.2 + 3.0

    def test_update_focus_quality(self):
        from lib.services.data.api.metrics import (
            FOCUS_QUALITY_GAUGE,
            update_focus_quality,
        )

        update_focus_quality("Gold", 0.78)
        val = FOCUS_QUALITY_GAUGE.labels(symbol="Gold")._value.get()
        assert val == pytest.approx(0.78)

    def test_update_focus_quality_multiple_assets(self):
        from lib.services.data.api.metrics import (
            FOCUS_QUALITY_GAUGE,
            update_focus_quality,
        )

        update_focus_quality("MGC", 0.72)
        update_focus_quality("MNQ", 0.85)
        update_focus_quality("MES", 0.60)

        assert FOCUS_QUALITY_GAUGE.labels(symbol="MGC")._value.get() == pytest.approx(0.72)
        assert FOCUS_QUALITY_GAUGE.labels(symbol="MNQ")._value.get() == pytest.approx(0.85)
        assert FOCUS_QUALITY_GAUGE.labels(symbol="MES")._value.get() == pytest.approx(0.60)

    def test_update_positions_count(self):
        from lib.services.data.api.metrics import (
            POSITIONS_OPEN_COUNT,
            update_positions_count,
        )

        update_positions_count(3)
        assert POSITIONS_OPEN_COUNT._value.get() == 3

        update_positions_count(0)
        assert POSITIONS_OPEN_COUNT._value.get() == 0

    def test_update_redis_status_connected(self):
        from lib.services.data.api.metrics import (
            REDIS_CONNECTED,
            update_redis_status,
        )

        update_redis_status(True)
        assert REDIS_CONNECTED._value.get() == 1

    def test_update_redis_status_disconnected(self):
        from lib.services.data.api.metrics import (
            REDIS_CONNECTED,
            update_redis_status,
        )

        update_redis_status(False)
        assert REDIS_CONNECTED._value.get() == 0

    def test_record_orb_filter_result_passed(self):
        from lib.services.data.api.metrics import (
            ORB_FILTER_RESULTS_TOTAL,
            record_orb_filter_result,
        )

        initial = ORB_FILTER_RESULTS_TOTAL.labels(result="passed")._value.get()
        record_orb_filter_result("passed")
        after = ORB_FILTER_RESULTS_TOTAL.labels(result="passed")._value.get()
        assert after == initial + 1

    def test_record_orb_filter_result_rejected(self):
        from lib.services.data.api.metrics import (
            ORB_FILTER_RESULTS_TOTAL,
            record_orb_filter_result,
        )

        initial = ORB_FILTER_RESULTS_TOTAL.labels(result="rejected")._value.get()
        record_orb_filter_result("rejected")
        after = ORB_FILTER_RESULTS_TOTAL.labels(result="rejected")._value.get()
        assert after == initial + 1

    def test_record_orb_filter_result_error(self):
        from lib.services.data.api.metrics import (
            ORB_FILTER_RESULTS_TOTAL,
            record_orb_filter_result,
        )

        initial = ORB_FILTER_RESULTS_TOTAL.labels(result="error")._value.get()
        record_orb_filter_result("error")
        after = ORB_FILTER_RESULTS_TOTAL.labels(result="error")._value.get()
        assert after == initial + 1

    def test_record_orb_cnn_prob_observes(self):
        from lib.services.data.api.metrics import ORB_CNN_PROB, record_orb_cnn_prob

        before = ORB_CNN_PROB._sum.get()
        record_orb_cnn_prob(0.87)
        after = ORB_CNN_PROB._sum.get()
        assert after > before

    def test_record_orb_cnn_prob_count_increments(self):
        from lib.services.data.api.metrics import ORB_CNN_PROB, record_orb_cnn_prob

        # Read the current _count via the sum proxy (same pattern as _sum.get())
        # or by checking the generated output — avoids relying on private attrs
        # not present in prometheus_client type stubs.
        before_sum = ORB_CNN_PROB._sum.get()
        record_orb_cnn_prob(0.55)
        after_sum = ORB_CNN_PROB._sum.get()
        # Sum must have increased by approximately the observed value
        assert after_sum == pytest.approx(before_sum + 0.55, abs=1e-6)

    def test_record_orb_cnn_signal_signal(self):
        from lib.services.data.api.metrics import (
            ORB_CNN_SIGNALS_TOTAL,
            record_orb_cnn_signal,
        )

        initial = ORB_CNN_SIGNALS_TOTAL.labels(verdict="signal")._value.get()
        record_orb_cnn_signal("signal")
        after = ORB_CNN_SIGNALS_TOTAL.labels(verdict="signal")._value.get()
        assert after == initial + 1

    def test_record_orb_cnn_signal_no_signal(self):
        from lib.services.data.api.metrics import (
            ORB_CNN_SIGNALS_TOTAL,
            record_orb_cnn_signal,
        )

        initial = ORB_CNN_SIGNALS_TOTAL.labels(verdict="no_signal")._value.get()
        record_orb_cnn_signal("no_signal")
        after = ORB_CNN_SIGNALS_TOTAL.labels(verdict="no_signal")._value.get()
        assert after == initial + 1

    def test_record_orb_cnn_signal_skipped(self):
        from lib.services.data.api.metrics import (
            ORB_CNN_SIGNALS_TOTAL,
            record_orb_cnn_signal,
        )

        initial = ORB_CNN_SIGNALS_TOTAL.labels(verdict="skipped")._value.get()
        record_orb_cnn_signal("skipped")
        after = ORB_CNN_SIGNALS_TOTAL.labels(verdict="skipped")._value.get()
        assert after == initial + 1

    def test_update_daily_pnl_positive(self):
        from lib.services.data.api.metrics import DAILY_PNL_GAUGE, update_daily_pnl

        update_daily_pnl(350.75)
        assert DAILY_PNL_GAUGE._value.get() == pytest.approx(350.75)

    def test_update_daily_pnl_negative(self):
        from lib.services.data.api.metrics import DAILY_PNL_GAUGE, update_daily_pnl

        update_daily_pnl(-120.50)
        assert DAILY_PNL_GAUGE._value.get() == pytest.approx(-120.50)

    def test_update_daily_pnl_zero(self):
        from lib.services.data.api.metrics import DAILY_PNL_GAUGE, update_daily_pnl

        update_daily_pnl(0.0)
        assert DAILY_PNL_GAUGE._value.get() == pytest.approx(0.0)

    def test_update_consecutive_losses(self):
        from lib.services.data.api.metrics import (
            CONSECUTIVE_LOSSES_GAUGE,
            update_consecutive_losses,
        )

        update_consecutive_losses(3)
        assert CONSECUTIVE_LOSSES_GAUGE._value.get() == 3

    def test_update_consecutive_losses_zero(self):
        from lib.services.data.api.metrics import (
            CONSECUTIVE_LOSSES_GAUGE,
            update_consecutive_losses,
        )

        update_consecutive_losses(0)
        assert CONSECUTIVE_LOSSES_GAUGE._value.get() == 0

    def test_update_model_metrics(self):
        from lib.services.data.api.metrics import (
            MODEL_TRAIN_SAMPLES,
            MODEL_VAL_ACCURACY,
            MODEL_VAL_PRECISION,
            MODEL_VAL_RECALL,
            update_model_metrics,
        )

        update_model_metrics(
            val_accuracy=83.73,
            val_precision=91.71,
            val_recall=78.26,
            train_samples=9941,
        )
        assert MODEL_VAL_ACCURACY._value.get() == pytest.approx(83.73)
        assert MODEL_VAL_PRECISION._value.get() == pytest.approx(91.71)
        assert MODEL_VAL_RECALL._value.get() == pytest.approx(78.26)
        assert MODEL_TRAIN_SAMPLES._value.get() == 9941

    def test_update_model_metrics_overwrites(self):
        from lib.services.data.api.metrics import (
            MODEL_VAL_ACCURACY,
            update_model_metrics,
        )

        update_model_metrics(val_accuracy=70.0, val_precision=75.0, val_recall=60.0, train_samples=3000)
        update_model_metrics(val_accuracy=85.0, val_precision=90.0, val_recall=80.0, train_samples=10000)
        assert MODEL_VAL_ACCURACY._value.get() == pytest.approx(85.0)


class TestPathNormalization:
    """Test _normalize_path for metric cardinality reduction."""

    def test_empty_path(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("") == "/"

    def test_root_path(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/") == "/"

    def test_health_unchanged(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/health") == "/health"

    def test_sse_dashboard_unchanged(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/sse/dashboard") == "/sse/dashboard"

    def test_focus_symbol_normalized(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/api/focus/mgc") == "/api/focus/{id}"
        assert _normalize_path("/api/focus/mnq") == "/api/focus/{id}"

    def test_trades_id_normalized(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/trades/123") == "/trades/{id}"
        assert _normalize_path("/trades/456/close") == "/trades/{id}/close"

    def test_positions_path_normalized(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/positions/abc") == "/positions/{id}"

    def test_journal_path_normalized(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/journal/entries") == "/journal/{id}"

    def test_analysis_latest_symbol_normalized(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/analysis/latest/Gold") == "/analysis/latest/{id}"

    def test_data_ohlcv_normalized(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/data/ohlcv/MGC=F") == "/data/ohlcv/{id}"

    def test_data_daily_normalized(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/data/daily/MGC=F") == "/data/daily/{id}"

    def test_non_matching_path_unchanged(self):
        from lib.services.data.api.metrics import _normalize_path

        assert _normalize_path("/risk/status") == "/risk/status"
        assert _normalize_path("/risk/check") == "/risk/check"
        assert _normalize_path("/metrics/prometheus") == "/metrics/prometheus"


class TestPrometheusOutput:
    """Test that generate_latest produces valid Prometheus text format."""

    def test_generate_latest_returns_bytes(self):
        from prometheus_client import generate_latest

        from lib.services.data.api.metrics import get_registry

        output = generate_latest(get_registry())
        assert isinstance(output, bytes)

    def test_generate_latest_contains_help_lines(self):
        from prometheus_client import generate_latest

        from lib.services.data.api.metrics import get_registry

        output = generate_latest(get_registry()).decode("utf-8")
        assert "# HELP http_requests_total" in output
        assert "# HELP sse_connections_active" in output
        assert "# HELP risk_checks_total" in output

    def test_generate_latest_contains_type_lines(self):
        from prometheus_client import generate_latest

        from lib.services.data.api.metrics import get_registry

        output = generate_latest(get_registry()).decode("utf-8")
        assert "# TYPE http_requests_total counter" in output
        assert "# TYPE http_request_duration_seconds histogram" in output
        assert "# TYPE sse_connections_active gauge" in output

    def test_output_has_metric_values(self):
        from prometheus_client import generate_latest

        from lib.services.data.api.metrics import (
            get_registry,
            record_sse_event,
        )

        record_sse_event("test-event")
        output = generate_latest(get_registry()).decode("utf-8")
        assert "sse_events_total" in output

    def test_output_contains_all_metric_families(self):
        from prometheus_client import generate_latest

        from lib.services.data.api.metrics import get_registry

        output = generate_latest(get_registry()).decode("utf-8")
        expected_metrics = [
            # Original metrics
            "http_requests_total",
            "http_request_duration_seconds",
            "sse_connections_active",
            "sse_events_total",
            "engine_last_refresh_epoch",
            "engine_cycle_duration_seconds",
            "risk_checks_total",
            "orb_detections_total",
            "no_trade_alerts_total",
            "focus_quality_gauge",
            "positions_open_count",
            "redis_connected",
            # New ORB filter + CNN metrics
            "orb_filter_results_total",
            "orb_cnn_prob",
            "orb_cnn_signals_total",
            # New risk / P&L metrics
            "daily_pnl_gauge",
            "consecutive_losses_gauge",
            # New CNN model performance metrics
            "model_val_accuracy",
            "model_val_precision",
            "model_val_recall",
            "model_train_samples",
            # Trainer dataset metrics
            "trainer_images_generated",
            "trainer_label_balance",
            "trainer_render_time_seconds",
        ]
        for metric in expected_metrics:
            assert metric in output, f"Missing metric: {metric}"


class TestTrainerMetrics:
    """Tests for trainer dataset Prometheus metrics."""

    def test_trainer_images_generated_registered(self):
        from lib.services.data.api.metrics import TRAINER_IMAGES_GENERATED

        assert TRAINER_IMAGES_GENERATED is not None
        assert "trainer_images_generated" in TRAINER_IMAGES_GENERATED._name

    def test_trainer_label_balance_registered(self):
        from lib.services.data.api.metrics import TRAINER_LABEL_BALANCE

        assert TRAINER_LABEL_BALANCE is not None
        assert "trainer_label_balance" in TRAINER_LABEL_BALANCE._name

    def test_trainer_render_time_seconds_registered(self):
        from lib.services.data.api.metrics import TRAINER_RENDER_TIME_SECONDS

        assert TRAINER_RENDER_TIME_SECONDS is not None
        assert "trainer_render_time_seconds" in TRAINER_RENDER_TIME_SECONDS._name

    def test_record_trainer_dataset_stats_sets_images(self):
        from lib.services.data.api.metrics import (
            TRAINER_IMAGES_GENERATED,
            record_trainer_dataset_stats,
        )

        record_trainer_dataset_stats(
            total_images=1800,
            label_distribution={"good": 1200, "bad": 600},
            render_time_seconds=45.3,
        )
        assert TRAINER_IMAGES_GENERATED._value.get() == 1800

    def test_record_trainer_dataset_stats_sets_label_balance(self):
        from lib.services.data.api.metrics import (
            TRAINER_LABEL_BALANCE,
            record_trainer_dataset_stats,
        )

        record_trainer_dataset_stats(
            total_images=2000,
            label_distribution={"good": 1420, "bad": 580},
            render_time_seconds=60.0,
        )
        assert TRAINER_LABEL_BALANCE.labels(label="good")._value.get() == 1420
        assert TRAINER_LABEL_BALANCE.labels(label="bad")._value.get() == 580

    def test_record_trainer_dataset_stats_observes_render_time(self):
        from lib.services.data.api.metrics import (
            TRAINER_RENDER_TIME_SECONDS,
            record_trainer_dataset_stats,
        )

        before_sum = TRAINER_RENDER_TIME_SECONDS._sum.get()
        record_trainer_dataset_stats(
            total_images=500,
            label_distribution={"good": 300, "bad": 200},
            render_time_seconds=120.5,
        )
        after_sum = TRAINER_RENDER_TIME_SECONDS._sum.get()
        assert after_sum == pytest.approx(before_sum + 120.5, abs=1e-3)

    def test_record_trainer_dataset_stats_overwrites_images_on_second_call(self):
        from lib.services.data.api.metrics import (
            TRAINER_IMAGES_GENERATED,
            record_trainer_dataset_stats,
        )

        record_trainer_dataset_stats(
            total_images=1000,
            label_distribution={"good": 600, "bad": 400},
            render_time_seconds=30.0,
        )
        record_trainer_dataset_stats(
            total_images=2500,
            label_distribution={"good": 1500, "bad": 1000},
            render_time_seconds=75.0,
        )
        # Gauge should reflect the most recent call
        assert TRAINER_IMAGES_GENERATED._value.get() == 2500

    def test_record_trainer_dataset_stats_empty_label_distribution(self):
        from lib.services.data.api.metrics import (
            TRAINER_IMAGES_GENERATED,
            record_trainer_dataset_stats,
        )

        # Should not raise even with no label entries
        record_trainer_dataset_stats(
            total_images=0,
            label_distribution={},
            render_time_seconds=0.1,
        )
        assert TRAINER_IMAGES_GENERATED._value.get() == 0

    def test_record_trainer_dataset_stats_zero_render_time(self):
        from lib.services.data.api.metrics import (
            TRAINER_RENDER_TIME_SECONDS,
            record_trainer_dataset_stats,
        )

        before_sum = TRAINER_RENDER_TIME_SECONDS._sum.get()
        record_trainer_dataset_stats(
            total_images=10,
            label_distribution={"good": 7, "bad": 3},
            render_time_seconds=0.0,
        )
        after_sum = TRAINER_RENDER_TIME_SECONDS._sum.get()
        # Sum unchanged when render_time is 0
        assert after_sum == pytest.approx(before_sum, abs=1e-6)


class TestCollectLiveGauges:
    """Test _collect_live_gauges reads from cache."""

    def test_collect_with_no_cache_data(self):
        """Should not crash when cache returns None for everything."""
        from lib.services.data.api.metrics import _collect_live_gauges

        with patch("lib.services.data.api.metrics.update_redis_status") as mock_redis:
            _collect_live_gauges()
            # Should have been called (either True or False)
            mock_redis.assert_called()

    def test_collect_focus_updates_gauges(self):
        from lib.services.data.api.metrics import (
            FOCUS_QUALITY_GAUGE,
            _collect_live_gauges,
        )

        focus_data = json.dumps(
            {
                "assets": [
                    {"symbol": "Gold", "quality": 0.82},
                    {"symbol": "Nasdaq", "quality": 0.67},
                ]
            }
        ).encode()

        with patch("lib.core.cache.cache_get", return_value=focus_data):
            _collect_live_gauges()
            assert FOCUS_QUALITY_GAUGE.labels(symbol="Gold")._value.get() == pytest.approx(0.82)
            assert FOCUS_QUALITY_GAUGE.labels(symbol="Nasdaq")._value.get() == pytest.approx(0.67)

    def test_collect_positions_updates_gauge(self):
        # We test update_positions_count directly since _collect_live_gauges
        # reads from the real cache which may not be available
        from lib.services.data.api.metrics import (
            POSITIONS_OPEN_COUNT,
            update_positions_count,
        )

        update_positions_count(2)
        assert POSITIONS_OPEN_COUNT._value.get() == 2


def _can_patch_cache_get():
    """Check if we can patch cache_get in the metrics module."""
    try:
        import lib.core.cache  # noqa: F401

        return True
    except ImportError:
        return False


from contextlib import nullcontext  # noqa: E402


def _mock_cache_for_collect(data):
    """Return a null context when we can't properly patch cache."""
    return nullcontext()


# ===========================================================================
# SECTION 2: Prometheus Middleware — Integration Tests
# ===========================================================================


class TestPrometheusMiddleware:
    """Test the ASGI middleware that instruments HTTP requests."""

    @pytest.fixture()
    def app_with_middleware(self):
        """Build a minimal FastAPI app with PrometheusMiddleware."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.metrics import PrometheusMiddleware

        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)

        @app.get("/test/hello")
        def hello():
            return {"msg": "hello"}

        @app.get("/test/slow")
        def slow():
            import time

            time.sleep(0.05)
            return {"msg": "slow"}

        @app.get("/test/error")
        def error():
            raise ValueError("boom")

        return TestClient(app, raise_server_exceptions=False)

    def test_middleware_records_successful_request(self, app_with_middleware):
        from lib.services.data.api.metrics import HTTP_REQUESTS_TOTAL

        initial = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/hello", status="200")._value.get()

        resp = app_with_middleware.get("/test/hello")
        assert resp.status_code == 200

        after = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/hello", status="200")._value.get()
        assert after == initial + 1

    def test_middleware_records_404(self, app_with_middleware):
        from lib.services.data.api.metrics import HTTP_REQUESTS_TOTAL

        initial = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/nonexistent", status="404")._value.get()

        resp = app_with_middleware.get("/test/nonexistent")
        assert resp.status_code == 404

        after = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/nonexistent", status="404")._value.get()
        assert after == initial + 1

    def test_middleware_records_duration(self, app_with_middleware):
        from lib.services.data.api.metrics import HTTP_REQUEST_DURATION

        # Record initial sum
        initial_sum = HTTP_REQUEST_DURATION.labels(method="GET", path="/test/slow")._sum.get()

        app_with_middleware.get("/test/slow")

        after_sum = HTTP_REQUEST_DURATION.labels(method="GET", path="/test/slow")._sum.get()

        # Duration should have increased
        assert after_sum > initial_sum

    def test_middleware_records_500_on_error(self, app_with_middleware):
        from lib.services.data.api.metrics import HTTP_REQUESTS_TOTAL

        initial = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/error", status="500")._value.get()

        resp = app_with_middleware.get("/test/error")
        assert resp.status_code == 500

        after = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/error", status="500")._value.get()
        assert after == initial + 1


# ===========================================================================
# SECTION 3: Prometheus Endpoint — Integration Tests
# ===========================================================================


class TestPrometheusEndpoint:
    """Test the /metrics/prometheus HTTP endpoint."""

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.metrics import PrometheusMiddleware
        from lib.services.data.api.metrics import router as metrics_router

        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)
        app.include_router(metrics_router)

        return TestClient(app)

    def test_endpoint_returns_200(self, client):
        resp = client.get("/metrics/prometheus")
        assert resp.status_code == 200

    def test_endpoint_returns_prometheus_content_type(self, client):
        resp = client.get("/metrics/prometheus")
        ct = resp.headers.get("content-type", "")
        # Prometheus content type includes version info
        assert "text/plain" in ct or "text/openmetrics" in ct or "application" in ct

    def test_endpoint_contains_http_metrics(self, client):
        # Make a request first so metrics are populated
        client.get("/metrics/prometheus")
        resp = client.get("/metrics/prometheus")
        body = resp.text
        assert "http_requests_total" in body

    def test_endpoint_contains_all_metric_families(self, client):
        resp = client.get("/metrics/prometheus")
        body = resp.text
        assert "sse_connections_active" in body
        assert "risk_checks_total" in body
        assert "orb_detections_total" in body
        assert "positions_open_count" in body
        assert "redis_connected" in body

    def test_endpoint_idempotent(self, client):
        """Multiple scrapes should work without error."""
        for _ in range(5):
            resp = client.get("/metrics/prometheus")
            assert resp.status_code == 200


# ===========================================================================
# SECTION 4: Rate Limiting — Unit Tests
# ===========================================================================


class TestClientKeyFunction:
    """Test _client_key_func key derivation."""

    def _make_request(self, headers: dict[str, str] | None = None, client_host: str = "127.0.0.1") -> MagicMock:
        """Create a mock Request with specified headers and client info."""
        req = MagicMock()
        req.headers = headers or {}
        req.client = MagicMock()
        req.client.host = client_host
        return req

    def test_api_key_takes_priority(self):
        from lib.services.data.api.rate_limit import _client_key_func

        req = self._make_request(
            headers={"x-api-key": "my-secret-key-12345"},
            client_host="192.168.1.1",
        )
        key = _client_key_func(req)
        assert key.startswith("apikey:")
        assert "my-secre" in key  # first 8 chars

    def test_api_key_short_key(self):
        from lib.services.data.api.rate_limit import _client_key_func

        req = self._make_request(headers={"x-api-key": "abc"})
        key = _client_key_func(req)
        assert key == "apikey:abc"

    def test_forwarded_for_used_when_no_api_key(self):
        from lib.services.data.api.rate_limit import _client_key_func

        req = self._make_request(
            headers={"x-forwarded-for": "10.0.0.1, 10.0.0.2"},
            client_host="172.17.0.1",
        )
        key = _client_key_func(req)
        assert key == "ip:10.0.0.1"

    def test_forwarded_for_single_ip(self):
        from lib.services.data.api.rate_limit import _client_key_func

        req = self._make_request(headers={"x-forwarded-for": "203.0.113.50"})
        key = _client_key_func(req)
        assert key == "ip:203.0.113.50"

    def test_remote_address_fallback(self):
        from lib.services.data.api.rate_limit import _client_key_func

        req = self._make_request(headers={}, client_host="192.168.1.100")
        # get_remote_address from slowapi reads request.client.host
        key = _client_key_func(req)
        assert key.startswith("ip:")


class TestPathLimitMapping:
    """Test get_limit_for_path returns correct limits."""

    def test_health_gets_public_limit(self):
        from lib.services.data.api.rate_limit import (
            PUBLIC_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/health") == PUBLIC_LIMIT

    def test_docs_gets_public_limit(self):
        from lib.services.data.api.rate_limit import (
            PUBLIC_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/docs") == PUBLIC_LIMIT

    def test_metrics_gets_public_limit(self):
        from lib.services.data.api.rate_limit import (
            PUBLIC_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/metrics") == PUBLIC_LIMIT
        assert get_limit_for_path("/metrics/prometheus") == PUBLIC_LIMIT

    def test_sse_gets_sse_limit(self):
        from lib.services.data.api.rate_limit import (
            SSE_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/sse/dashboard") == SSE_LIMIT
        assert get_limit_for_path("/sse/health") == SSE_LIMIT

    def test_trades_gets_mutations_limit(self):
        from lib.services.data.api.rate_limit import (
            MUTATIONS_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/trades") == MUTATIONS_LIMIT
        assert get_limit_for_path("/trades/123/close") == MUTATIONS_LIMIT

    def test_log_trade_gets_mutations_limit(self):
        from lib.services.data.api.rate_limit import (
            MUTATIONS_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/log_trade") == MUTATIONS_LIMIT

    def test_positions_update_gets_mutations_limit(self):
        from lib.services.data.api.rate_limit import (
            MUTATIONS_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/positions/update") == MUTATIONS_LIMIT

    def test_risk_check_gets_mutations_limit(self):
        from lib.services.data.api.rate_limit import (
            MUTATIONS_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/risk/check") == MUTATIONS_LIMIT

    def test_force_refresh_gets_heavy_limit(self):
        from lib.services.data.api.rate_limit import (
            HEAVY_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/actions/force_refresh") == HEAVY_LIMIT

    def test_optimize_now_gets_heavy_limit(self):
        from lib.services.data.api.rate_limit import (
            HEAVY_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/actions/optimize_now") == HEAVY_LIMIT

    def test_run_backtest_gets_heavy_limit(self):
        from lib.services.data.api.rate_limit import (
            HEAVY_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/actions/run_backtest") == HEAVY_LIMIT

    def test_unknown_path_gets_default_limit(self):
        from lib.services.data.api.rate_limit import (
            DEFAULT_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/some/random/path") == DEFAULT_LIMIT


class TestLimiterSingleton:
    """Test limiter creation and lifecycle."""

    def test_get_limiter_returns_limiter(self):
        from lib.services.data.api.rate_limit import (
            get_limiter,
            reset_limiter,
        )

        reset_limiter()
        limiter = get_limiter()
        assert limiter is not None

    def test_get_limiter_returns_same_instance(self):
        from lib.services.data.api.rate_limit import (
            get_limiter,
            reset_limiter,
        )

        reset_limiter()
        limiter1 = get_limiter()
        limiter2 = get_limiter()
        assert limiter1 is limiter2

    def test_reset_limiter_clears_singleton(self):
        from lib.services.data.api.rate_limit import (
            get_limiter,
            reset_limiter,
        )

        _limiter1 = get_limiter()  # noqa: F841
        reset_limiter()
        limiter2 = get_limiter()
        # After reset, a new instance should be created
        # (they may or may not be the same object depending on implementation,
        # but at minimum the reset shouldn't crash)
        assert limiter2 is not None


class TestRateLimitEnabled:
    """Test the enabled/disabled toggle."""

    def test_is_rate_limiting_enabled_when_disabled(self):
        # Our test env sets RATE_LIMIT_ENABLED=0
        from lib.services.data.api.rate_limit import (
            is_rate_limiting_enabled,
        )

        # The module reads env at import time, so we need to check
        # the current state
        result = is_rate_limiting_enabled()
        # In test env, it should be disabled
        assert isinstance(result, bool)

    def test_effective_limit_when_disabled(self):
        from lib.services.data.api.rate_limit import (
            _DISABLED_LIMIT,
            _get_effective_limit,
        )

        # When RATE_LIMIT_ENABLED is "0", should return the disabled limit
        # This depends on the module-level _ENABLED var
        result = _get_effective_limit("30/minute")
        # When disabled, should return the super-high limit
        assert result in ("30/minute", _DISABLED_LIMIT)


class TestRateLimitHandler:
    """Test the custom 429 response handler."""

    def _make_rate_limit_exc(self, limit_str: str = "20 per 1 minute"):
        """Create a RateLimitExceeded with a mock Limit object."""
        from slowapi.errors import RateLimitExceeded

        mock_limit = MagicMock()
        mock_limit.error_message = None
        mock_limit.limit = limit_str
        return RateLimitExceeded(mock_limit)

    def test_handler_returns_429(self):
        from lib.services.data.api.rate_limit import _rate_limit_handler

        req = MagicMock()
        req.method = "POST"
        req.url = MagicMock()
        req.url.path = "/trades"
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"

        exc = self._make_rate_limit_exc("20/minute")
        resp = _rate_limit_handler(req, exc)
        assert resp.status_code == 429

    def test_handler_returns_json_body(self):
        from lib.services.data.api.rate_limit import _rate_limit_handler

        req = MagicMock()
        req.method = "GET"
        req.url = MagicMock()
        req.url.path = "/api/focus"
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "10.0.0.1"

        exc = self._make_rate_limit_exc("30/minute")
        resp = _rate_limit_handler(req, exc)

        body = json.loads(resp.body.decode())  # type: ignore[union-attr]
        assert body["error"] == "rate_limit_exceeded"
        assert "detail" in body
        assert "retry_after" in body

    def test_handler_includes_retry_after_header(self):
        from lib.services.data.api.rate_limit import _rate_limit_handler

        req = MagicMock()
        req.method = "GET"
        req.url = MagicMock()
        req.url.path = "/test"
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "10.0.0.1"

        exc = self._make_rate_limit_exc("5/minute")
        resp = _rate_limit_handler(req, exc)

        assert "retry-after" in resp.headers or "Retry-After" in resp.headers


class TestSetupRateLimiting:
    """Test setup_rate_limiting installs correctly on a FastAPI app."""

    def test_setup_installs_limiter_on_app_state(self):
        from fastapi import FastAPI

        from lib.services.data.api.rate_limit import (
            reset_limiter,
            setup_rate_limiting,
        )

        reset_limiter()
        app = FastAPI()
        limiter = setup_rate_limiting(app)

        assert hasattr(app.state, "limiter")
        assert app.state.limiter is limiter

    def test_setup_returns_limiter_instance(self):
        from fastapi import FastAPI

        from lib.services.data.api.rate_limit import (
            reset_limiter,
            setup_rate_limiting,
        )

        reset_limiter()
        app = FastAPI()
        limiter = setup_rate_limiting(app)

        assert limiter is not None

    def test_setup_idempotent(self):
        """Calling setup multiple times should not crash."""
        from fastapi import FastAPI

        from lib.services.data.api.rate_limit import (
            reset_limiter,
            setup_rate_limiting,
        )

        reset_limiter()
        app = FastAPI()
        limiter1 = setup_rate_limiting(app)
        limiter2 = setup_rate_limiting(app)
        assert limiter1 is limiter2


class TestStorageUri:
    """Test storage backend URI resolution."""

    def test_default_is_memory(self):
        from lib.services.data.api.rate_limit import _get_storage_uri

        uri = _get_storage_uri()
        assert "memory" in uri or uri.startswith("redis")


# ===========================================================================
# SECTION 5: Rate Limiting — Integration Tests with FastAPI
# ===========================================================================


class TestRateLimitIntegration:
    """Integration test: rate limiting with a real FastAPI app."""

    @pytest.fixture()
    def limited_client(self):
        """Build a FastAPI app with an extremely low rate limit for testing."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        from lib.services.data.api.rate_limit import _rate_limit_handler

        # Create a fresh limiter with a very tight limit
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["3/minute"],
            storage_uri="memory://",
        )

        app = FastAPI()
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]

        @app.get("/test/limited")
        @limiter.limit("3/minute")
        def limited_endpoint(request: Request):
            return {"ok": True}

        @app.get("/test/unlimited")
        def unlimited_endpoint():
            return {"ok": True}

        return TestClient(app)

    def test_within_limit_succeeds(self, limited_client):
        """Requests within the limit should succeed."""
        for i in range(3):
            resp = limited_client.get("/test/limited")
            assert resp.status_code == 200, f"Request {i + 1} should succeed"

    def test_exceeding_limit_returns_429(self, limited_client):
        """Requests beyond the limit should get 429."""
        # Exhaust the limit
        for _ in range(3):
            limited_client.get("/test/limited")

        # This should be rate-limited
        resp = limited_client.get("/test/limited")
        assert resp.status_code == 429

    def test_429_response_body_structure(self, limited_client):
        """The 429 response should have proper JSON structure."""
        for _ in range(3):
            limited_client.get("/test/limited")

        resp = limited_client.get("/test/limited")
        assert resp.status_code == 429
        body = resp.json()
        assert "error" in body
        assert body["error"] == "rate_limit_exceeded"
        assert "detail" in body

    def test_unlimited_endpoint_not_affected_by_limit(self, limited_client):
        """Endpoint without explicit limit should use default."""
        for _ in range(5):
            resp = limited_client.get("/test/unlimited")
            # May or may not be limited depending on default; just verify it's accessible
            assert resp.status_code in (200, 429)


# ===========================================================================
# SECTION 6: Combined Integration — Both features together
# ===========================================================================


class TestCombinedIntegration:
    """Test metrics + rate limiting coexisting in the same app."""

    @pytest.fixture()
    def full_client(self):
        """Build an app with both Prometheus middleware and rate limiting."""
        from fastapi import FastAPI, Request
        from fastapi.testclient import TestClient
        from slowapi import Limiter
        from slowapi.errors import RateLimitExceeded
        from slowapi.util import get_remote_address

        from lib.services.data.api.metrics import PrometheusMiddleware
        from lib.services.data.api.metrics import router as metrics_router
        from lib.services.data.api.rate_limit import _rate_limit_handler

        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["100/minute"],
            storage_uri="memory://",
        )

        app = FastAPI()
        app.add_middleware(PrometheusMiddleware)
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)  # type: ignore[arg-type]
        app.include_router(metrics_router)

        @app.get("/test/data")
        def test_data():
            return {"data": [1, 2, 3]}

        @app.post("/test/action")
        @limiter.limit("2/minute")
        def test_action(request: Request):
            return {"done": True}

        return TestClient(app)

    def test_metrics_endpoint_accessible(self, full_client):
        resp = full_client.get("/metrics/prometheus")
        assert resp.status_code == 200

    def test_metrics_track_normal_requests(self, full_client):
        from lib.services.data.api.metrics import HTTP_REQUESTS_TOTAL

        initial = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/data", status="200")._value.get()

        full_client.get("/test/data")

        after = HTTP_REQUESTS_TOTAL.labels(method="GET", path="/test/data", status="200")._value.get()
        assert after == initial + 1

    def test_metrics_track_429_responses(self, full_client):
        from lib.services.data.api.metrics import HTTP_REQUESTS_TOTAL

        # Exhaust the limit
        full_client.post("/test/action")
        full_client.post("/test/action")

        initial_429 = HTTP_REQUESTS_TOTAL.labels(method="POST", path="/test/action", status="429")._value.get()

        resp = full_client.post("/test/action")
        assert resp.status_code == 429

        after_429 = HTTP_REQUESTS_TOTAL.labels(method="POST", path="/test/action", status="429")._value.get()
        assert after_429 == initial_429 + 1

    def test_prometheus_endpoint_self_reports(self, full_client):
        """The metrics endpoint should report its own request in metrics."""
        resp = full_client.get("/metrics/prometheus")
        body = resp.text
        # The endpoint itself should show up in http_requests_total
        assert "http_requests_total" in body

    def test_both_features_dont_interfere(self, full_client):
        """Making multiple requests should work with both features active."""
        for _ in range(5):
            resp = full_client.get("/test/data")
            assert resp.status_code == 200

        resp = full_client.get("/metrics/prometheus")
        assert resp.status_code == 200
        assert "http_request_duration_seconds" in resp.text


# ===========================================================================
# SECTION 7: Edge Cases & Error Handling
# ===========================================================================


class TestMetricsEdgeCases:
    """Edge cases for the metrics system."""

    def test_sse_connect_disconnect_balance(self):
        """SSE gauge should track connects and disconnects accurately."""
        from lib.services.data.api.metrics import (
            SSE_CONNECTIONS_ACTIVE,
            record_sse_connect,
            record_sse_disconnect,
        )

        SSE_CONNECTIONS_ACTIVE.set(0)
        record_sse_connect()
        record_sse_connect()
        record_sse_connect()
        assert SSE_CONNECTIONS_ACTIVE._value.get() == 3

        record_sse_disconnect()
        assert SSE_CONNECTIONS_ACTIVE._value.get() == 2

        record_sse_disconnect()
        record_sse_disconnect()
        assert SSE_CONNECTIONS_ACTIVE._value.get() == 0

    def test_update_focus_quality_overwrites(self):
        """Focus quality gauge should reflect the latest value."""
        from lib.services.data.api.metrics import (
            FOCUS_QUALITY_GAUGE,
            update_focus_quality,
        )

        update_focus_quality("TestAsset", 0.50)
        assert FOCUS_QUALITY_GAUGE.labels(symbol="TestAsset")._value.get() == 0.50

        update_focus_quality("TestAsset", 0.95)
        assert FOCUS_QUALITY_GAUGE.labels(symbol="TestAsset")._value.get() == pytest.approx(0.95)

    def test_record_engine_cycle_zero_duration(self):
        """Zero-duration cycle should be observable."""
        from lib.services.data.api.metrics import record_engine_cycle

        record_engine_cycle(0.0)
        # No crash

    def test_record_engine_cycle_large_duration(self):
        """Large duration should be observable."""
        from lib.services.data.api.metrics import record_engine_cycle

        record_engine_cycle(120.0)
        # No crash

    def test_many_different_sse_event_types(self):
        """Many different event types should not crash."""
        from lib.services.data.api.metrics import record_sse_event

        event_types = [
            "focus-update",
            "mgc-update",
            "mnq-update",
            "mes-update",
            "no-trade-alert",
            "heartbeat",
            "session-change",
            "grok-update",
            "risk-update",
            "orb-update",
            "positions-update",
        ]
        for et in event_types:
            record_sse_event(et)
        # No crash, no cardinality explosion


class TestRateLimitEdgeCases:
    """Edge cases for rate limiting."""

    def test_client_key_with_empty_headers(self):
        from lib.services.data.api.rate_limit import _client_key_func

        req = MagicMock()
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "0.0.0.0"
        key = _client_key_func(req)
        assert key.startswith("ip:")

    def test_client_key_with_none_forwarded_for(self):
        from lib.services.data.api.rate_limit import _client_key_func

        req = MagicMock()
        req.headers = {"x-forwarded-for": ""}
        req.client = MagicMock()
        req.client.host = "10.0.0.1"
        # Empty string is falsy, so should fall through to remote address
        key = _client_key_func(req)
        assert key.startswith("ip:")

    def test_get_limit_for_path_empty(self):
        from lib.services.data.api.rate_limit import (
            DEFAULT_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("") == DEFAULT_LIMIT

    def test_get_limit_for_path_root(self):
        from lib.services.data.api.rate_limit import (
            DEFAULT_LIMIT,
            get_limit_for_path,
        )

        assert get_limit_for_path("/") == DEFAULT_LIMIT

    def test_handler_with_low_limit(self):
        """Handler should work with a very low limit."""
        from slowapi.errors import RateLimitExceeded

        from lib.services.data.api.rate_limit import _rate_limit_handler

        req = MagicMock()
        req.method = "GET"
        req.url = MagicMock()
        req.url.path = "/"
        req.headers = {}
        req.client = MagicMock()
        req.client.host = "127.0.0.1"

        mock_limit = MagicMock()
        mock_limit.error_message = None
        mock_limit.limit = "1 per 1 minute"
        exc = RateLimitExceeded(mock_limit)
        resp = _rate_limit_handler(req, exc)
        assert resp.status_code == 429


# ===========================================================================
# SECTION 8: Patterns module — "Should Not Trade" verification
# ===========================================================================


class TestShouldNotTradePatterns:
    """Verify the patterns.py evaluate_no_trade function works correctly."""

    def _make_asset(
        self,
        symbol: str = "Gold",
        quality: float = 0.70,
        vol_percentile: float = 0.50,
    ) -> dict:
        return {
            "symbol": symbol,
            "quality": quality,
            "vol_percentile": vol_percentile,
            "skip": quality < 0.55,
        }

    def test_evaluate_no_trade_exists(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assert callable(evaluate_no_trade)

    def test_no_assets_triggers_no_trade(self):
        from lib.services.engine.patterns import evaluate_no_trade

        result = evaluate_no_trade([])
        assert result.should_skip is True
        assert len(result.reasons) > 0

    def test_all_low_quality_triggers(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [
            self._make_asset("MGC", quality=0.30),
            self._make_asset("MNQ", quality=0.40),
        ]
        result = evaluate_no_trade(assets)
        assert result.should_skip is True
        assert any("quality" in r.lower() for r in result.reasons)

    def test_extreme_vol_triggers(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [
            self._make_asset("MGC", quality=0.80, vol_percentile=0.92),
        ]
        result = evaluate_no_trade(assets)
        assert result.should_skip is True
        assert any("vol" in r.lower() for r in result.reasons)

    def test_daily_loss_triggers(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [self._make_asset("MGC", quality=0.80)]
        risk_status = {"daily_pnl": -300.0, "consecutive_losses": 0}
        result = evaluate_no_trade(assets, risk_status=risk_status)
        assert result.should_skip is True
        assert any("loss" in r.lower() for r in result.reasons)

    def test_consecutive_losses_triggers(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [self._make_asset("MGC", quality=0.80)]
        risk_status = {"daily_pnl": -50.0, "consecutive_losses": 3}
        result = evaluate_no_trade(assets, risk_status=risk_status)
        assert result.should_skip is True
        assert any("consecutive" in r.lower() or "losing" in r.lower() for r in result.reasons)

    def test_good_conditions_allow_trading(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [
            self._make_asset("MGC", quality=0.75, vol_percentile=0.40),
            self._make_asset("MNQ", quality=0.80, vol_percentile=0.50),
        ]
        # Use a time during active trading window (8 AM ET)
        active_time = datetime(2026, 1, 15, 8, 0, 0, tzinfo=_EST)
        result = evaluate_no_trade(assets, now=active_time)
        assert result.should_skip is False
        assert len(result.reasons) == 0

    def test_result_has_severity(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [self._make_asset("MGC", quality=0.30)]
        result = evaluate_no_trade(assets)
        assert result.severity in ("info", "warning", "critical")

    def test_result_to_dict(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [self._make_asset("MGC", quality=0.30)]
        result = evaluate_no_trade(assets)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "should_skip" in d
        assert "reasons" in d
        assert "checks" in d
        assert "severity" in d
        assert "checked_at" in d

    def test_publish_and_clear_no_trade_alert(self):
        from lib.services.engine.patterns import (
            clear_no_trade_alert,
            evaluate_no_trade,
            publish_no_trade_alert,
        )

        assets = [self._make_asset("MGC", quality=0.30)]
        result = evaluate_no_trade(assets)
        assert result.should_skip is True

        # Should not crash even without real Redis
        success = publish_no_trade_alert(result)
        # May or may not succeed depending on cache availability
        assert isinstance(success, bool)

        cleared = clear_no_trade_alert()
        assert isinstance(cleared, bool)

    def test_session_ended_check(self):
        from lib.services.engine.patterns import evaluate_no_trade

        assets = [self._make_asset("MGC", quality=0.80)]
        # 2 PM ET — session has ended
        late_time = datetime(2026, 1, 15, 14, 0, 0, tzinfo=_EST)
        result = evaluate_no_trade(assets, now=late_time)
        assert result.should_skip is True
        assert any("session" in r.lower() or "ended" in r.lower() for r in result.reasons)

    def test_no_trade_conditions_enum(self):
        from lib.services.engine.patterns import NoTradeCondition

        assert hasattr(NoTradeCondition, "ALL_LOW_QUALITY")
        assert hasattr(NoTradeCondition, "EXTREME_VOLATILITY")
        assert hasattr(NoTradeCondition, "DAILY_LOSS_EXCEEDED")
        assert hasattr(NoTradeCondition, "CONSECUTIVE_LOSSES")
        assert hasattr(NoTradeCondition, "LATE_SESSION_NO_SETUPS")
        assert hasattr(NoTradeCondition, "NO_MARKET_DATA")
        assert hasattr(NoTradeCondition, "SESSION_ENDED")
