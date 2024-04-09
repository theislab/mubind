from importlib.metadata import version

__all__ = ["pl", "pp", "tl", "get", "set", "datasets", 'models']

__version__ = version("mubind")

from . import datasets
from . import models
from . import pl
from . import tl
from . import get

try:
    import bindome
    import mubind
    mubind.bindome = bindome

except ImportError:
    print("bindome has not been installed. Please check at https://github.com/theislab/bindome")
