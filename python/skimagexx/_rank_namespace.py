"""Own public rank modules; supported functions never resolve upstream names."""
import importlib
from . import _rank
from ._compat import module_attribute


def initialize(module):
    relative=module.__name__.split('.',1)[1]
    module.__doc__='Native rolling-histogram rank filters with explicit compatibility boundaries.'
    module.__path__=[]
    # The pinned generic module exposes these callables as identity aliases
    # but omits them from star imports. Keep availability and star exports separate.
    names=(_rank._GENERIC_FUNCTIONS if relative=='filters.rank.generic' else
           _rank._BILATERAL_FUNCTIONS if relative=='filters.rank.bilateral' else _rank.__all__)
    module.__all__=[name for name in names
                    if relative!='filters.rank.generic' or name not in ('sum','windowed_histogram')]
    for name in names:
        setattr(module,name,getattr(_rank,name))
    def missing(name):
        if relative=='filters.rank' and name in ('generic','bilateral'):
            return importlib.import_module(module.__name__+'.'+name)
        return module_attribute(relative,name)
    module.__getattr__=missing
    module.__dir__=lambda:sorted(set(module.__dict__)|set(module.__all__)|({'generic','bilateral'} if relative=='filters.rank' else set()))
