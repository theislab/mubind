from importlib.metadata import version

from . import pl, pp, tl, get, set

__all__ = ["pl", "pp", "tl", "get", "set", "datasets"]

__version__ = version("mubind")

from . import datasets
from . import models
from . import pl
from . import tl
from . import get

try:
    import bindome
    mubind.bindome = bindome

except ImportError:
    print("bindome has not been installed. Please check at https://github.com/theislab/bindome")
