from importlib import metadata


try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None  # type: ignore[assignment]


__all__ = ["pl", "pp", "tl", "get", "set", "datasets", 'models']

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
