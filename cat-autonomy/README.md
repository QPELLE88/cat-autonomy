# CAT Autonomous Truck Platform

Internal software platform for Caterpillar's autonomous mining and construction trucks.

## Architecture

```
cat-autonomy/
├── perception/          # Obstacle detection from LiDAR point clouds
│   ├── obstacle_detector.py    # Detection classes (Rock, Vehicle, Personnel)
│   ├── lidar_processor.cpp     # Real-time point cloud clustering (C++)
│   └── config/sensor_config.yaml
├── simulation/          # Test scenario generation and validation
│   ├── scenario_generator.py   # YAML scenario file generation
│   └── scenarios/              # Generated test scenarios
├── fleet/               # Fleet management platform
│   ├── dashboard/              # Real-time monitoring UI
│   └── telemetry/              # Event schemas and data pipeline
└── tests/
    └── perception/             # Unit tests for obstacle detection
```

## Supported Obstacle Types

| Type | Profile | Response | Safety Buffer |
|------|---------|----------|---------------|
| Rock | Spherical, 0.3-3.0m | SLOW | 5m |
| Vehicle | Rectangular, 1.5-6.0m tall | STOP | 10m |
| Personnel | Vertical cylinder, 1.2-2.2m | STOP | 20m |
| FallenTree | Horizontal cylinder, 5.0m+ length | REROUTE | 15m |

## Requirements

Python 3.12+ is required (declared in `pyproject.toml` and `.python-version`).

Install dependencies from the `cat-autonomy/` directory with either:

```bash
pip install -e ".[dev]"          # from pyproject.toml (runtime + pytest)
pip install -r requirements.txt  # convenience alternative
```

## Running Tests

```bash
cd tests/perception
python -m pytest test_obstacle_detector.py -v
```

## Generating Simulation Scenarios

```bash
python simulation/scenario_generator.py
```

## TODO

- [ ] Generate simulation scenarios for new obstacle types
- [ ] Build fleet dashboard widget for obstacle event monitoring
