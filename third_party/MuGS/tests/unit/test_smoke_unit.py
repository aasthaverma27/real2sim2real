"""
Smoke tests that always run regardless of optional deps (mjlab, gsplat, ...).

Their purpose is twofold:
1. Sanity-check that the standalone `mugs` package imports cleanly in CI.
2. Guarantee at least one test is collected so pytest doesn't exit 5
   ("no tests collected") when all mjlab-gated tests get skipped.
"""

import importlib


def test_mugs_version():
    import mugs

    assert isinstance(mugs.__version__, str)
    assert mugs.__version__


def test_standalone_sensor_imports():
    sensors = importlib.import_module("mugs.sensors")
    assert hasattr(sensors, "GaussianSensor")
    assert hasattr(sensors, "GaussianSensorConfig")


def test_mask_config_imports():
    importlib.import_module("mugs.utils.mask_config")
