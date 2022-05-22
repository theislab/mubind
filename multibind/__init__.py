

from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl"]

__version__ = version("multibind")



import multibind
import multibind.pl
import multibind.tl
import multibind.datasets
import multibind.models

try:
    import bindome
    multibind.bindome = bindome
except ImportError as e:
    print('bindome has not been installed. Please check at https://github.com/theislab/bindome') # module doesn't exist, deal with it.
