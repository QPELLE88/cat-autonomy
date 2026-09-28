"""
CAT Autonomous Trucks — Simulation Scenario Generator
Generates YAML scenario files for the AV simulation environment.
Each scenario defines obstacle placement, road geometry, truck state,
and expected autonomous response.
"""
import yaml
import os
from dataclasses import dataclass, asdict
from enum import Enum


class RoadType(Enum):
    STRAIGHT = "straight"
    CURVED_LEFT = "curved_left"
    CURVED_RIGHT = "curved_right"
    INTERSECTION = "intersection"
    GRADE_CHANGE = "grade_change"


class WeatherCondition(Enum):
    CLEAR = "clear"
    RAIN = "rain"
    FOG = "fog"
    DUST = "dust"
    NIGHT = "night"


@dataclass
class ObstaclePlacement:
    obstacle_type: str
    position_x: float  # meters from truck start
    position_y: float  # meters lateral offset
    position_z: float  # meters height
    rotation_deg: float
    scale: float = 1.0


@dataclass
class TruckState:
    speed_kmh: float
    heading_deg: float
    load_tonnes: float
    position_x: float = 0.0
    position_y: float = 0.0


@dataclass
class ExpectedResponse:
    action: str  # "stop", "slow", "reroute", "alert_operator"
    max_response_time_ms: int
    min_stopping_distance_m: float
    safety_buffer_m: float


@dataclass
class SimScenario:
    name: str
    description: str
    road_type: str
    weather: str
    visibility_m: float
    truck_state: TruckState
    obstacles: list[ObstaclePlacement]
    expected_response: ExpectedResponse
    pass_criteria: dict[str, bool]


def generate_rock_scenarios(output_dir: str, count: int = 10) -> list[str]:
    """Generate standard rock obstacle scenarios."""
    scenarios = []
    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        speed = 30.0 + (i * 5)
        scenario = SimScenario(
            name=f"rock_obstacle_{i+1:03d}",
            description=f"Single rock obstacle on haul road at {speed} km/h approach",
            road_type=RoadType.STRAIGHT.value,
            weather=WeatherCondition.CLEAR.value,
            visibility_m=200.0,
            truck_state=TruckState(speed_kmh=speed, heading_deg=0.0, load_tonnes=200.0),
            obstacles=[
                ObstaclePlacement(
                    obstacle_type="rock",
                    position_x=80.0 + i * 10,
                    position_y=0.5,
                    position_z=0.0,
                    rotation_deg=0.0,
                    scale=0.8 + i * 0.2,
                )
            ],
            expected_response=ExpectedResponse(
                action="slow",
                max_response_time_ms=500,
                min_stopping_distance_m=5.0,
                safety_buffer_m=5.0,
            ),
            pass_criteria={
                "truck_stopped_before_obstacle": True,
                "safety_buffer_maintained": True,
                "response_time_within_limit": True,
            },
        )
        filepath = os.path.join(output_dir, f"{scenario.name}.yaml")
        with open(filepath, "w") as f:
            yaml.dump(asdict(scenario), f, default_flow_style=False)
        scenarios.append(filepath)

    return scenarios


def generate_vehicle_scenarios(output_dir: str, count: int = 10) -> list[str]:
    """Generate vehicle obstacle scenarios (parked trucks, excavators)."""
    scenarios = []
    os.makedirs(output_dir, exist_ok=True)

    vehicles = ["haul_truck", "excavator", "loader", "water_truck", "grader"]

    for i in range(count):
        vehicle_type = vehicles[i % len(vehicles)]
        scenario = SimScenario(
            name=f"vehicle_obstacle_{vehicle_type}_{i+1:03d}",
            description=f"Stationary {vehicle_type} blocking haul road",
            road_type=RoadType.STRAIGHT.value if i < 5 else RoadType.CURVED_LEFT.value,
            weather=WeatherCondition.CLEAR.value if i < 7 else WeatherCondition.DUST.value,
            visibility_m=150.0 if i < 7 else 80.0,
            truck_state=TruckState(speed_kmh=40.0, heading_deg=0.0, load_tonnes=250.0),
            obstacles=[
                ObstaclePlacement(
                    obstacle_type=f"vehicle_{vehicle_type}",
                    position_x=100.0,
                    position_y=1.0 if i % 2 == 0 else -1.0,
                    position_z=0.0,
                    rotation_deg=45.0 * (i % 4),
                )
            ],
            expected_response=ExpectedResponse(
                action="stop",
                max_response_time_ms=300,
                min_stopping_distance_m=10.0,
                safety_buffer_m=10.0,
            ),
            pass_criteria={
                "truck_stopped_before_obstacle": True,
                "safety_buffer_maintained": True,
                "response_time_within_limit": True,
                "no_collision": True,
            },
        )
        filepath = os.path.join(output_dir, f"{scenario.name}.yaml")
        with open(filepath, "w") as f:
            yaml.dump(asdict(scenario), f, default_flow_style=False)
        scenarios.append(filepath)

    return scenarios


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scenarios_dir = os.path.join(base_dir, "scenarios")

    print("Generating rock scenarios...")
    rock_files = generate_rock_scenarios(os.path.join(scenarios_dir, "rock"), count=10)
    print(f"  Created {len(rock_files)} rock scenarios")

    print("Generating vehicle scenarios...")
    vehicle_files = generate_vehicle_scenarios(os.path.join(scenarios_dir, "vehicle"), count=10)
    print(f"  Created {len(vehicle_files)} vehicle scenarios")

    print(f"\nTotal scenarios generated: {len(rock_files) + len(vehicle_files)}")
