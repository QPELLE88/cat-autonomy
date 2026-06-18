"""
CAT Autonomous Trucks — Obstacle Detection Module
Processes LiDAR point cloud data to detect and classify obstacles
in the autonomous truck's path.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np


class ResponseAction(Enum):
    """Autonomous truck response actions when obstacle detected."""
    SLOW = "slow"
    STOP = "stop"
    REROUTE = "reroute"
    ALERT_OPERATOR = "alert_operator"


@dataclass
class PointCloud:
    """Represents a LiDAR point cloud cluster."""
    points: np.ndarray  # Nx3 array of (x, y, z) coordinates
    intensity: np.ndarray  # N array of return intensities
    timestamp: float
    sensor_id: str


@dataclass
class DetectionResult:
    """Result of obstacle detection."""
    obstacle_type: str
    confidence: float
    bounding_box: tuple  # (x_min, y_min, z_min, x_max, y_max, z_max)
    response_action: ResponseAction
    safety_buffer_meters: float
    position: tuple  # (x, y, z) center


class ObstacleBase(ABC):
    """Base class for all obstacle detectors."""

    def __init__(self, min_confidence: float = 0.85):
        self.min_confidence = min_confidence

    @abstractmethod
    def detect(self, cloud: PointCloud) -> Optional[DetectionResult]:
        """Detect this obstacle type in the given point cloud."""
        pass

    @abstractmethod
    def get_profile_signature(self) -> dict:
        """Return the geometric profile signature for this obstacle type."""
        pass

    def compute_bounding_box(self, points: np.ndarray) -> tuple:
        """Compute axis-aligned bounding box for a point cluster."""
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        return (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])


class RockObstacle(ObstacleBase):
    """Detects rock obstacles using compact spherical profile matching."""

    def get_profile_signature(self) -> dict:
        return {
            "shape": "spherical",
            "aspect_ratio_range": (0.6, 1.4),
            "min_height_m": 0.3,
            "max_height_m": 3.0,
        }

    def detect(self, cloud: PointCloud) -> Optional[DetectionResult]:
        points = cloud.points
        if len(points) < 10:
            return None

        bbox = self.compute_bounding_box(points)
        width = bbox[3] - bbox[0]
        height = bbox[5] - bbox[2]
        depth = bbox[4] - bbox[1]

        # Check for compact spherical profile
        aspect_ratio = max(width, depth) / max(height, 0.01)
        if 0.6 <= aspect_ratio <= 1.4 and 0.3 <= height <= 3.0:
            confidence = min(0.95, len(points) / 100.0)
            if confidence >= self.min_confidence:
                center = points.mean(axis=0)
                return DetectionResult(
                    obstacle_type="rock",
                    confidence=confidence,
                    bounding_box=bbox,
                    response_action=ResponseAction.SLOW,
                    safety_buffer_meters=5.0,
                    position=tuple(center),
                )
        return None


class VehicleObstacle(ObstacleBase):
    """Detects vehicles (trucks, excavators, loaders) using rectangular profile."""

    def get_profile_signature(self) -> dict:
        return {
            "shape": "rectangular",
            "aspect_ratio_range": (1.5, 4.0),
            "min_height_m": 1.5,
            "max_height_m": 6.0,
            "min_length_m": 3.0,
        }

    def detect(self, cloud: PointCloud) -> Optional[DetectionResult]:
        points = cloud.points
        if len(points) < 50:
            return None

        bbox = self.compute_bounding_box(points)
        width = bbox[3] - bbox[0]
        height = bbox[5] - bbox[2]
        length = bbox[4] - bbox[1]

        aspect = max(width, length) / max(min(width, length), 0.01)
        if (1.5 <= aspect <= 4.0 and
            1.5 <= height <= 6.0 and
            max(width, length) >= 3.0):
            confidence = min(0.98, len(points) / 200.0)
            if confidence >= self.min_confidence:
                center = points.mean(axis=0)
                return DetectionResult(
                    obstacle_type="vehicle",
                    confidence=confidence,
                    bounding_box=bbox,
                    response_action=ResponseAction.STOP,
                    safety_buffer_meters=10.0,
                    position=tuple(center),
                )
        return None


class PersonnelObstacle(ObstacleBase):
    """Detects personnel/humans using tall narrow vertical profile."""

    def get_profile_signature(self) -> dict:
        return {
            "shape": "vertical_cylinder",
            "aspect_ratio_range": (0.2, 0.6),
            "min_height_m": 1.2,
            "max_height_m": 2.2,
            "max_width_m": 1.0,
        }

    def detect(self, cloud: PointCloud) -> Optional[DetectionResult]:
        points = cloud.points
        if len(points) < 20:
            return None

        bbox = self.compute_bounding_box(points)
        width = bbox[3] - bbox[0]
        height = bbox[5] - bbox[2]
        depth = bbox[4] - bbox[1]

        max_horizontal = max(width, depth)
        if (max_horizontal <= 1.0 and
            1.2 <= height <= 2.2 and
            max_horizontal / max(height, 0.01) <= 0.6):
            confidence = min(0.99, len(points) / 50.0)
            if confidence >= self.min_confidence:
                center = points.mean(axis=0)
                return DetectionResult(
                    obstacle_type="personnel",
                    confidence=confidence,
                    bounding_box=bbox,
                    response_action=ResponseAction.STOP,
                    safety_buffer_meters=20.0,
                    position=tuple(center),
                )
        return None


class FallenTreeObstacle(ObstacleBase):
    """Detects fallen trees using elongated cylindrical profile for forestry/construction expansion."""

    def get_profile_signature(self) -> dict:
        return {
            "shape": "horizontal_cylinder",
            "aspect_ratio_range": (3.0, 10.0),
            "min_height_m": 0.5,
            "max_height_m": 2.0,
            "min_length_m": 5.0,
            "max_width_m": 1.5,
        }

    def detect(self, cloud: PointCloud) -> Optional[DetectionResult]:
        points = cloud.points
        if len(points) < 30:
            return None

        bbox = self.compute_bounding_box(points)
        width = bbox[3] - bbox[0]
        height = bbox[5] - bbox[2]
        length = bbox[4] - bbox[1]

        # Check for elongated horizontal cylindrical profile (fallen tree)
        max_horizontal = max(width, length)
        min_horizontal = min(width, length)
        aspect_ratio = max_horizontal / max(min_horizontal, 0.01)

        if (3.0 <= aspect_ratio <= 10.0 and
            0.5 <= height <= 2.0 and
            max_horizontal >= 5.0 and
            min_horizontal <= 1.5):
            confidence = min(0.92, len(points) / 150.0)
            if confidence >= self.min_confidence:
                center = points.mean(axis=0)
                return DetectionResult(
                    obstacle_type="fallen_tree",
                    confidence=confidence,
                    bounding_box=bbox,
                    response_action=ResponseAction.REROUTE,
                    safety_buffer_meters=15.0,
                    position=tuple(center),
                )
        return None


# ============================================================
# OBSTACLE DETECTION PIPELINE
# ============================================================

class ObstacleDetectionPipeline:
    """Main pipeline that runs all obstacle detectors against incoming LiDAR data."""

    def __init__(self):
        self.detectors: List[ObstacleBase] = [
            RockObstacle(),
            VehicleObstacle(),
            PersonnelObstacle(),
            FallenTreeObstacle(),
        ]

    def register_detector(self, detector: ObstacleBase):
        """Register a new obstacle detector to the pipeline."""
        self.detectors.append(detector)

    def process_frame(self, clouds: List[PointCloud]) -> List[DetectionResult]:
        """Process a single LiDAR frame (multiple point cloud clusters)."""
        results = []
        for cloud in clouds:
            for detector in self.detectors:
                result = detector.detect(cloud)
                if result is not None:
                    results.append(result)
                    break  # First match wins for this cluster
        return results

    def get_highest_priority_action(self, results: List[DetectionResult]) -> ResponseAction:
        """Determine the highest priority response action from all detections."""
        priority = [ResponseAction.STOP, ResponseAction.REROUTE,
                    ResponseAction.SLOW, ResponseAction.ALERT_OPERATOR]
        for action in priority:
            if any(r.response_action == action for r in results):
                return action
        return ResponseAction.ALERT_OPERATOR
