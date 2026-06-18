"""
Tests for CAT Autonomous Truck Obstacle Detection Pipeline.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from perception.obstacle_detector import (
    ObstacleDetectionPipeline,
    RockObstacle,
    VehicleObstacle,
    PersonnelObstacle,
    FallenTreeObstacle,
    PointCloud,
    ResponseAction,
)


def make_cloud(points_array, sensor_id="lidar_primary"):
    """Helper to create a PointCloud from a numpy array."""
    pts = np.array(points_array, dtype=np.float64)
    return PointCloud(
        points=pts,
        intensity=np.ones(len(pts)),
        timestamp=1000.0,
        sensor_id=sensor_id,
    )


class TestRockObstacle:
    def test_detects_spherical_rock(self):
        # Create a roughly spherical cluster (~1m diameter)
        np.random.seed(42)
        center = np.array([10.0, 5.0, 0.5])
        points = center + np.random.randn(100, 3) * 0.3
        cloud = make_cloud(points)

        detector = RockObstacle()
        result = detector.detect(cloud)

        assert result is not None
        assert result.obstacle_type == "rock"
        assert result.response_action == ResponseAction.SLOW
        assert result.safety_buffer_meters == 5.0

    def test_rejects_flat_object(self):
        # Create a flat cluster (not spherical)
        points = np.zeros((50, 3))
        points[:, 0] = np.linspace(0, 10, 50)  # 10m long
        points[:, 1] = np.random.randn(50) * 0.1  # very narrow
        points[:, 2] = np.random.randn(50) * 0.05  # very flat
        cloud = make_cloud(points)

        detector = RockObstacle()
        result = detector.detect(cloud)
        assert result is None


class TestVehicleObstacle:
    def test_detects_large_vehicle(self):
        # Create a rectangular cluster (~8m x 3m x 4m)
        np.random.seed(123)
        points = np.random.rand(200, 3)
        points[:, 0] *= 3.0   # 3m wide
        points[:, 1] *= 8.0   # 8m long
        points[:, 2] *= 4.0   # 4m tall
        points[:, 2] += 0.5   # offset from ground
        cloud = make_cloud(points)

        detector = VehicleObstacle()
        result = detector.detect(cloud)

        assert result is not None
        assert result.obstacle_type == "vehicle"
        assert result.response_action == ResponseAction.STOP
        assert result.safety_buffer_meters == 10.0

    def test_rejects_small_object(self):
        # Too small to be a vehicle
        points = np.random.rand(60, 3) * 0.5
        cloud = make_cloud(points)

        detector = VehicleObstacle()
        result = detector.detect(cloud)
        assert result is None


class TestPersonnelObstacle:
    def test_detects_standing_person(self):
        # Create a tall narrow cluster (~0.5m x 0.5m x 1.8m)
        np.random.seed(456)
        points = np.random.rand(50, 3)
        points[:, 0] *= 0.4   # narrow
        points[:, 1] *= 0.4   # narrow
        points[:, 2] *= 1.8   # tall
        points[:, 2] += 0.1   # slight offset
        cloud = make_cloud(points)

        detector = PersonnelObstacle()
        result = detector.detect(cloud)

        assert result is not None
        assert result.obstacle_type == "personnel"
        assert result.response_action == ResponseAction.STOP
        assert result.safety_buffer_meters == 20.0


class TestFallenTreeObstacle:
    def test_detects_fallen_tree(self):
        # Create an elongated horizontal cylinder (~8m x 1m x 1m)
        np.random.seed(789)
        points = np.random.rand(150, 3)
        points[:, 0] *= 1.0   # 1m wide (trunk diameter)
        points[:, 1] *= 8.0   # 8m long (tree length)
        points[:, 2] *= 1.0   # 1m tall (fallen height)
        points[:, 2] += 0.5   # offset from ground
        cloud = make_cloud(points)

        detector = FallenTreeObstacle()
        result = detector.detect(cloud)

        assert result is not None
        assert result.obstacle_type == "fallen_tree"
        assert result.response_action == ResponseAction.REROUTE
        assert result.safety_buffer_meters == 15.0

    def test_rejects_short_object(self):
        # Too short to be a fallen tree
        points = np.random.rand(40, 3)
        points[:, 0] *= 1.0
        points[:, 1] *= 2.0   # Only 2m long
        points[:, 2] *= 1.0
        cloud = make_cloud(points)

        detector = FallenTreeObstacle()
        result = detector.detect(cloud)
        assert result is None

    def test_rejects_tall_object(self):
        # Too tall to be a fallen tree (standing tree)
        points = np.random.rand(40, 3)
        points[:, 0] *= 1.0
        points[:, 1] *= 8.0
        points[:, 2] *= 5.0   # 5m tall (not fallen)
        cloud = make_cloud(points)

        detector = FallenTreeObstacle()
        result = detector.detect(cloud)
        assert result is None


class TestPipeline:
    def test_pipeline_processes_multiple_clusters(self):
        pipeline = ObstacleDetectionPipeline()

        # Create a rock and a person
        np.random.seed(789)
        rock_points = np.array([[10, 5, 0.5]]) + np.random.randn(100, 3) * 0.3
        person_points = np.random.rand(50, 3)
        person_points[:, 0] *= 0.4
        person_points[:, 1] *= 0.4
        person_points[:, 2] *= 1.8

        clouds = [make_cloud(rock_points), make_cloud(person_points)]
        results = pipeline.process_frame(clouds)

        assert len(results) >= 1  # At least one detection

    def test_highest_priority_is_stop(self):
        pipeline = ObstacleDetectionPipeline()
        from perception.obstacle_detector import DetectionResult

        results = [
            DetectionResult("rock", 0.9, (0,0,0,1,1,1), ResponseAction.SLOW, 5.0, (0,0,0)),
            DetectionResult("personnel", 0.95, (0,0,0,1,1,1), ResponseAction.STOP, 20.0, (5,5,0)),
        ]

        action = pipeline.get_highest_priority_action(results)
        assert action == ResponseAction.STOP


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
