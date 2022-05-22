from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl", "datasets"]

__version__ = version("multibind")


import multibind
import multibind.datasets
import multibind.models
import multibind.pl
import multibind.tl

try:
    import bindome

    multibind.bindome = bindome
except ImportError:
    print(
        "bindome has not been installed. Please check at https://github.com/theislab/bindome"
    )  # module doesn't exist, deal with it.
