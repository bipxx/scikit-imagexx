"""NumPy primitive policy controls only; no original IGG or image/native calls."""
import ast
import json
from pathlib import Path
import warnings
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def policy():
    import numpy as np
    source=ast.parse((ROOT/'python/skimagexx/_inverse_gaussian_gradient.py').read_bytes())
    node=next(v for v in source.body if isinstance(v,ast.FunctionDef)and v.name=='_policy')
    namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),'<exact IGG _policy>','exec'),namespace)
    return np,namespace['_policy']

CASES=[]
for dtype in ('float32','float64','complex64','complex128'):
    for operation,kind,flags in [('divide','invalid',1),('divide','divide',2),('divide','overflow',4),('divide','underflow',8),('multiply','invalid',1),('multiply','overflow',4),('multiply','underflow',8),('add','invalid',1),('add','overflow',4)]:
        CASES.append((dtype,operation,kind,flags))
    if dtype.startswith('complex'):CASES.append((dtype,'divide','complex-zero',3))
    else:CASES.append((dtype,'sqrt','invalid',1))

def operands(np,dtype,operation,kind):
    limits=np.finfo(dtype)
    if kind=='complex-zero':a,b=1+0j,0+0j
    elif kind=='invalid':a,b=(-1.,None)if operation=='sqrt'else (0.,0.)if operation=='divide'else (np.inf,-np.inf)if operation=='add'else (np.inf,0.)
    elif kind=='divide':a,b=(1+1j if dtype.startswith('complex')else 1.),0.
    elif kind=='overflow':a,b=limits.max,(.5 if operation=='divide'else limits.max if operation=='add'else 2.)
    else:a,b=limits.tiny,(limits.max if operation=='divide'else limits.tiny)
    return np.array([a],dtype=dtype),None if b is None else np.array([b],dtype=dtype)

def observe_policy(np,call,settings):
    notices=[];events=[];failure=None;old=np.geterrcall();outer=np.geterr().copy()
    class Log:
        def write(self,text):events.append(('log',text))
    def callback(error,flags):events.append(('call',error,flags))
    handler=Log()if 'log'in settings.values()else callback
    try:
        np.seterrcall(handler)
        with np.errstate(**settings),warnings.catch_warnings(record=True)as notices:
            warnings.simplefilter('always')
            try:call()
            except FloatingPointError as e:failure=(type(e).__module__,type(e).__qualname__,e.args)
            assert np.geterrcall()is handler and np.geterr()==settings
        return dict(warnings=[(v.category.__module__,v.category.__qualname__,str(v.message))for v in notices],events=events,error=failure)
    finally:
        np.seterrcall(old)
        assert np.geterr()==outer and np.geterrcall()is old

@pytest.mark.parametrize('dtype,operation,kind,flags',CASES,ids=lambda v:str(v))
@pytest.mark.parametrize('mode',('warn','raise','call','log','ignore'))
def test_primitive_policy_wording_order_masks_and_handlers(policy,dtype,operation,kind,flags,mode,monkeypatch):
    np,deliver=policy;a,b=operands(np,dtype,operation,kind);settings=dict.fromkeys(('divide','over','under','invalid'),mode)
    reference=lambda:getattr(np,operation)(a)if b is None else getattr(np,operation)(a,b)
    expected=observe_policy(np,reference,settings)
    actual=observe_policy(np,lambda:deliver(flags,operation,np.dtype(dtype)),settings)
    assert actual==expected

@pytest.mark.parametrize('dtype',('complex64','complex128'))
@pytest.mark.parametrize('settings',[
 dict(divide='ignore',over='ignore',under='ignore',invalid='raise'),
 dict(divide='raise',over='ignore',under='ignore',invalid='ignore'),
 dict(divide='call',over='ignore',under='ignore',invalid='warn'),
 dict(divide='ignore',over='ignore',under='ignore',invalid='call'),
 dict(divide='log',over='ignore',under='ignore',invalid='log')])
def test_joint_complex_division_preserves_full_error_mask(policy,dtype,settings):
    np,deliver=policy;a,b=operands(np,dtype,'divide','complex-zero')
    expected=observe_policy(np,lambda:np.divide(a,b),settings)
    assert observe_policy(np,lambda:deliver(3,'divide',np.dtype(dtype)),settings)==expected

@pytest.mark.parametrize('dtype',('float32','float64','complex64','complex128'))
def test_combined_multiply_flags_are_one_policy_dispatch(policy,dtype):
    np,deliver=policy;limits=np.finfo(dtype);a=np.array([limits.max,limits.tiny,np.inf],dtype=dtype);b=np.array([2.,limits.tiny,0.],dtype=dtype)
    settings=dict.fromkeys(('divide','over','under','invalid'),'call')
    assert observe_policy(np,lambda:deliver(13,'multiply',np.dtype(dtype)),settings)==observe_policy(np,lambda:np.multiply(a,b),settings)
