from __future__ import annotations

import unittest

import numpy as np

from map_converter import normalize_large_axis_contours


def contour(points: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)


class BoundaryNormalizationTests(unittest.TestCase):
    def test_near_horizontal_large_segment_is_straightened(self) -> None:
        source = contour([(50, 100), (850, 114), (850, 300), (50, 300)])
        normalized, segments, anomalies = normalize_large_axis_contours([source], 1000, 600, 3)
        points = normalized[0].reshape(-1, 2)
        self.assertEqual(points[0, 1], points[1, 1])
        self.assertTrue(any(item["orientation"] == "horizontal" for item in segments))
        self.assertFalse(anomalies)

    def test_real_diagonal_over_two_degrees_is_untouched(self) -> None:
        source = contour([(50, 100), (850, 150), (850, 300), (50, 300)])
        normalized, segments, _ = normalize_large_axis_contours([source], 1000, 600, 3)
        points = normalized[0].reshape(-1, 2)
        self.assertEqual(points[0, 1], 100)
        self.assertEqual(points[1, 1], 150)
        self.assertFalse(any(item["y1"] < 200 for item in segments if item["orientation"] == "horizontal"))

    def test_eight_pixel_gap_connects_but_nine_does_not(self) -> None:
        joined = contour([(50, 100), (300, 100), (300, 220), (309, 220), (309, 100), (600, 100), (600, 300), (50, 300)])
        _, segments, _ = normalize_large_axis_contours([joined], 800, 500, 3)
        top = [item for item in segments if item["orientation"] == "horizontal" and item["y1"] == 100]
        self.assertEqual(len(top), 1)
        self.assertTrue(top[0]["connected"])

        separate = contour([(50, 100), (300, 100), (300, 220), (310, 220), (310, 100), (600, 100), (600, 300), (50, 300)])
        _, segments, _ = normalize_large_axis_contours([separate], 800, 500, 3)
        top = [item for item in segments if item["orientation"] == "horizontal" and item["y1"] == 100]
        self.assertEqual(len(top), 2)

    def test_correction_over_32_pixels_is_rejected(self) -> None:
        source = contour([(50, 100), (2050, 166), (2050, 500), (50, 500)])
        normalized, segments, anomalies = normalize_large_axis_contours([source], 2200, 700, 3)
        points = normalized[0].reshape(-1, 2)
        self.assertEqual(points[0, 1], 100)
        self.assertEqual(points[1, 1], 166)
        self.assertFalse(any(item["y1"] < 300 for item in segments if item["orientation"] == "horizontal"))
        self.assertEqual(anomalies[0]["type"], "correction_exceeds_limit")


if __name__ == "__main__":
    unittest.main()
