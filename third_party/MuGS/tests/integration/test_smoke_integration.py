"""
Smoke tests that don't require mjlab.

Guarantees pytest collects at least one test even when the mjlab-gated
modules in this folder get skipped via `pytest.importorskip("mjlab")`,
so the integration job doesn't fail with exit code 5 ("no tests collected").
"""

import mujoco


def test_mujoco_compiles_minimal_scene():
    spec = mujoco.MjSpec.from_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
          </worldbody>
        </mujoco>
        """
    )
    model = spec.compile()
    assert model.ngeom == 1
