"""Own public rank modules; supported functions never resolve upstream names."""
import importlib
from . import _rank
from ._compat import module_attribute


def initialize(module):
    relative=module.__name__.split('.',1)[1]
    module.__doc__='Native rolling-histogram rank filters with explicit compatibility boundaries.'
    module.__path__=[]
    # The pinned generic module exposes this callable as an identity alias but
    # omits it from star imports. Keep availability and star exports separate.
    module.__all__=[name for name in _rank.__all__
                    if relative!='filters.rank.generic' or name!='windowed_histogram']
    for name in _rank.__all__:
        setattr(module,name,getattr(_rank,name))
    def missing(name):
        if relative=='filters.rank' and name=='generic':
            return importlib.import_module(module.__name__+'.generic')
        return module_attribute(relative,name)
    module.__getattr__=missing
    module.__dir__=lambda:sorted(set(module.__dict__)|set(module.__all__)|({'generic'} if relative=='filters.rank' else set()))
