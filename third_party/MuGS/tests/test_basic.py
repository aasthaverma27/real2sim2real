"""
Basic sanity tests

Simple tests to verify test infrastructure works.
"""

import pytest


def test_python_version():
    """Test Python version is 3.10+"""
    import sys
    assert sys.version_info >= (3, 10)


def test_numpy_available():
    """Test numpy is available"""
    import numpy as np
    arr = np.array([1, 2, 3])
    assert len(arr) == 3
    assert arr.sum() == 6


def test_yaml_available():
    """Test yaml is available"""
    import yaml
    data = {"test": "value"}
    yaml_str = yaml.dump(data)
    loaded = yaml.safe_load(yaml_str)
    assert loaded["test"] == "value"


class TestBasicMath:
    """Test basic math operations"""

    def test_addition(self):
        assert 1 + 1 == 2

    def test_multiplication(self):
        assert 2 * 3 == 6

    def test_division(self):
        assert 10 / 2 == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
