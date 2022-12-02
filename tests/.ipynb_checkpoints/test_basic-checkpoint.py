import pytest

import mubind as mb


def test_package_has_version():
    mb.__version__


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_example():
    assert 1 == 0  # This test is designed to fail.
