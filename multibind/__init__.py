

from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl"]

__version__ = version("multibind")



import multibind
import multibind.pl
import multibind.tl
import multibind.datasets
import multibind.models

import bindome

multibind.bindome = bindome