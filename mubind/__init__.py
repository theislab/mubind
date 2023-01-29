from importlib.metadata import version

from . import pl, pp, tl, get, set

__all__ = ["pl", "pp", "tl", "get", "set", "datasets"]

__version__ = version("mubind")


import mubind
import mubind.datasets
import mubind.models
import mubind.pl
import mubind.tl
import mubind.get

try:
    import bindome

    mubind.bindome = bindome
except ImportError:
    print("bindome has not been installed. Please check at https://github.com/theislab/bindome")
