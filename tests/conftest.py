import numpy as np
import pytest

from fissure_annotation.slicing import Case

AIR_HU = -800.0


@pytest.fixture
def case() -> Case:
    """A 60 x 60 x 60 volume of air with a lung block of two lobes touching at x = 30.

    Spacing differs per axis on purpose, so tests catch code that assumes isotropic voxels.
    """
    vol = np.full((60, 60, 60), AIR_HU, dtype=np.float32)
    lobes = np.zeros((60, 60, 60), dtype=np.uint8)
    lobes[10:30, 10:50, 10:50] = 1
    lobes[30:50, 10:50, 10:50] = 3
    return Case(vol, lobes, (1.0, 1.0, 2.0))
