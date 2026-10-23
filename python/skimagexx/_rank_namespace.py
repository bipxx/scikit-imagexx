"""Own public rank modules; supported functions never resolve upstream names."""
import importlib
from . import _rank
from ._compat import module_attribute


def initialize(module):
    relative=module.__name__.split('.',1)[1]
    module.__doc__='Native rolling-histogram rank filters with explicit compatibility boundaries.'
    module.__path__=[]
    module.__all__=list(_rank.__all__)
    for name in module.__all__:
        setattr(module,name,getattr(_rank,name))
    def missing(name):
        if relative=='filters.rank' and name=='generic':
            return importlib.import_module(module.__name__+'.generic')
        return module_attribute(relative,name)
    module.__getattr__=missing
    module.__dir__=lambda:sorted(set(module.__dict__)|set(module.__all__)|({'generic'} if relative=='filters.rank' else set()))
