import pytest

def test_import():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import multibind as mb

def test_package_has_version():
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import multibind as mb
    mb.__version__