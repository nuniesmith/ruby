"""
Tests for ``validate_dataset()`` in ``lib.services.training.dataset_generator``.

Covers:
  1. Valid CSV + matching images → report shows valid=True
  2. Missing images → correct missing count reported
  3. Empty CSV → total_rows=0, valid=True
  4. No ``label`` column → label_distribution is empty
  5. No ``image_path`` column → missing_images stays 0
  6. CSV file not found → valid=False with error message
  7. Label distribution correctly reported
  8. Symbol list correctly reported

All tests use ``tmp_path`` and synthetic PNGs — no real data or GPU needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from PIL import Image

from lib.services.training.dataset_generator import validate_dataset

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_png(path: Path) -> None:
    """Write a 4×4 solid-colour PNG to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (4, 4), color=(128, 0, 0))
    img.save(str(path))


def _make_csv(
    csv_path: Path,
    rows: list[dict],
) -> None:
    """Write a list of row-dicts as a CSV at *csv_path*."""
    df = pd.DataFrame(rows)
    df.to_csv(str(csv_path), index=False)


# ---------------------------------------------------------------------------
# 1. Valid CSV + matching images
# ---------------------------------------------------------------------------


class TestValidCsvWithImages:
    def test_valid_dataset_reports_true(self, tmp_path: Path) -> None:
        img1 = tmp_path / "images" / "img1.png"
        img2 = tmp_path / "images" / "img2.png"
        _make_tiny_png(img1)
        _make_tiny_png(img2)

        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": str(img1), "label": 1, "symbol": "MGC"},
                {"image_path": str(img2), "label": 0, "symbol": "MES"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=True)

        assert report["valid"] is True
        assert report["total_rows"] == 2
        assert report["missing_images"] == 0
        assert report["empty_image_paths"] == 0
        assert "error" not in report


# ---------------------------------------------------------------------------
# 2. Missing images → reports correct count
# ---------------------------------------------------------------------------


class TestMissingImages:
    def test_missing_images_count(self, tmp_path: Path) -> None:
        img1 = tmp_path / "images" / "img1.png"
        _make_tiny_png(img1)

        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": str(img1), "label": 1, "symbol": "MGC"},
                {"image_path": str(tmp_path / "images" / "missing1.png"), "label": 0, "symbol": "MES"},
                {"image_path": str(tmp_path / "images" / "missing2.png"), "label": 1, "symbol": "MNQ"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=True)

        assert report["valid"] is False
        assert report["missing_images"] == 2
        assert "2 images not found" in report.get("error", "")

    def test_empty_image_path_counted(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": "", "label": 1, "symbol": "MGC"},
                {"image_path": float("nan"), "label": 0, "symbol": "MES"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=True)

        assert report["empty_image_paths"] == 2
        # Empty paths are not counted as "missing" (they never had a path)
        assert report["missing_images"] == 0


# ---------------------------------------------------------------------------
# 3. Empty CSV
# ---------------------------------------------------------------------------


class TestEmptyCsv:
    def test_empty_csv_is_valid(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        # Create an empty CSV with just headers
        pd.DataFrame(columns=["image_path", "label", "symbol"]).to_csv(str(csv_path), index=False)

        report = validate_dataset(str(csv_path), check_images=True)

        assert report["valid"] is True
        assert report["total_rows"] == 0
        assert report["missing_images"] == 0


# ---------------------------------------------------------------------------
# 4. No ``label`` column
# ---------------------------------------------------------------------------


class TestNoLabelColumn:
    def test_label_distribution_empty_without_label_col(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": "x.png", "symbol": "MGC"},
                {"image_path": "y.png", "symbol": "MES"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=False)

        assert report["label_distribution"] == {}
        # The report should still be structurally valid
        assert report["total_rows"] == 2


# ---------------------------------------------------------------------------
# 5. No ``image_path`` column
# ---------------------------------------------------------------------------


class TestNoImagePathColumn:
    def test_missing_images_zero_without_image_path_col(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"label": 1, "symbol": "MGC"},
                {"label": 0, "symbol": "MES"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=True)

        # Without image_path column, the image check is skipped
        assert report["missing_images"] == 0
        assert report["empty_image_paths"] == 0
        assert report["valid"] is True


# ---------------------------------------------------------------------------
# 6. CSV not found
# ---------------------------------------------------------------------------


class TestCsvNotFound:
    def test_returns_invalid_with_error(self, tmp_path: Path) -> None:
        bogus = str(tmp_path / "nonexistent.csv")
        report = validate_dataset(bogus, check_images=True)

        assert report["valid"] is False
        assert "CSV not found" in report["error"]
        assert bogus in report["error"]

    def test_csv_not_found_report_has_no_rows(self, tmp_path: Path) -> None:
        bogus = str(tmp_path / "nonexistent.csv")
        report = validate_dataset(bogus, check_images=True)

        # When CSV doesn't exist, the report should not contain row-level data
        assert "total_rows" not in report


# ---------------------------------------------------------------------------
# 7. Label distribution correctly reported
# ---------------------------------------------------------------------------


class TestLabelDistribution:
    def test_distribution_matches_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": "a.png", "label": 1, "symbol": "MGC"},
                {"image_path": "b.png", "label": 1, "symbol": "MGC"},
                {"image_path": "c.png", "label": 0, "symbol": "MES"},
                {"image_path": "d.png", "label": 2, "symbol": "MNQ"},
                {"image_path": "e.png", "label": 2, "symbol": "MNQ"},
                {"image_path": "f.png", "label": 2, "symbol": "MNQ"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=False)

        dist = report["label_distribution"]
        assert dist[1] == 2
        assert dist[0] == 1
        assert dist[2] == 3

    def test_single_label_class(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": "a.png", "label": 0, "symbol": "MGC"},
                {"image_path": "b.png", "label": 0, "symbol": "MES"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=False)

        dist = report["label_distribution"]
        assert len(dist) == 1
        assert dist[0] == 2


# ---------------------------------------------------------------------------
# 8. Symbol list correctly reported
# ---------------------------------------------------------------------------


class TestSymbolList:
    def test_symbols_sorted_and_unique(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": "a.png", "label": 1, "symbol": "MNQ"},
                {"image_path": "b.png", "label": 0, "symbol": "MGC"},
                {"image_path": "c.png", "label": 1, "symbol": "MES"},
                {"image_path": "d.png", "label": 0, "symbol": "MNQ"},
                {"image_path": "e.png", "label": 1, "symbol": "MGC"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=False)

        assert report["symbols"] == ["MES", "MGC", "MNQ"]

    def test_no_symbol_column_returns_empty(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": "a.png", "label": 1},
                {"image_path": "b.png", "label": 0},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=False)

        assert report["symbols"] == []

    def test_single_symbol(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "labels.csv"
        _make_csv(
            csv_path,
            [
                {"image_path": "a.png", "label": 1, "symbol": "MGC"},
                {"image_path": "b.png", "label": 0, "symbol": "MGC"},
            ],
        )

        report = validate_dataset(str(csv_path), check_images=False)

        assert report["symbols"] == ["MGC"]
