"""Genuine installed-core wide OUTER saved domain; Linux GNU FP scope only."""
import ctypes,importlib,inspect,os,sys
import numpy as np
import pytest
import mark_wide_outer_capture as observer
from mark_wide_outer_private_compare import load_saved,seed_bytes,semantic,assess
SPECS=observer.case_specs()

@pytest.fixture(scope='module')
def genuine():
 assert sys.platform=='linux'and not os.environ.get('MARK_COMPONENT_DIR')
 assert os.environ.get('UTIL_VIEWS_FENV_MODE')=='source-built-v1'
 from util_views_fenv import backend
 import skimagexx._core as core
 find=importlib.import_module('skimagexx._find_boundaries');mark=importlib.import_module('skimagexx._mark_boundaries')
 assert find._core is mark._core is core and core.get_num_threads()==1
 names=('find_boundaries_into','mark_boundaries_prepare','mark_boundaries_mask','mark_boundaries_color','mark_boundaries_count','mark_boundaries_paint')
 assert all(inspect.isbuiltin(getattr(core,n))for n in names)
 return core,{'segmentation.find_boundaries':find.find_boundaries,'segmentation.mark_boundaries':mark.mark_boundaries},backend(),names

@pytest.mark.parametrize('workers',(1,4))
@pytest.mark.parametrize('spec',SPECS,ids=lambda s:s['id'])
def test_genuine_wide_outer_saved(spec,workers,genuine):
 core,functions,(libc,helper),names=genuine
 _,rows=load_saved();profile={'1.24.4':'minimum','2.5.4':'modern'}[np.__version__]
 expected=next(r for r in rows[profile]if r['id']==spec['id'])
 def env():
  raw=(ctypes.c_ubyte*32)();assert libc.fegetenv(ctypes.byref(raw))==0;return bytes(raw)
 def restore(raw):
  buffer=ctypes.create_string_buffer(raw,32);assert helper.reference_fenv_restore(buffer,32)==0 and env()==raw
 caller=env();old_worker=core.get_num_threads();core.set_num_threads(workers);oldtrace=sys.gettrace();oldprofile=sys.getprofile();immediate=[];returned=[];inside=False
 def deny(frame,event,value):
  if event=='call'and frame.f_globals.get('__name__','').startswith(('skimage.','scipy.ndimage')):raise AssertionError('ORIGINAL_PROVIDER_BODY_FORBIDDEN')
  return deny
 def count(frame,event,value):
  if inside and event=='c_return'and any(value is getattr(core,n)for n in names):returned.append(value.__name__)
 def target(*args,**options):
  nonlocal inside
  restore(seed_bytes(env(),expected['target_immediate']['before']));before=env();policy=observer.np_state(np);inside=True
  try:return functions[spec['api']](*args,**options)
  finally:
   after=env();inside=False;immediate.append(dict(before={n:int.from_bytes(before[o:o+w],'little')for n,o,w in [('control',0,2),('status',4,2),('mxcsr',28,4)]},after={n:int.from_bytes(after[o:o+w],'little')for n,o,w in [('control',0,2),('status',4,2),('mxcsr',28,4)]},numpy_before=policy,numpy_after=observer.np_state(np)))
 try:
  sys.settrace(deny);sys.setprofile(count);actual=observer.observe(target,spec,np)
 finally:
  sys.settrace(oldtrace);sys.setprofile(oldprofile);core.set_num_threads(old_worker);restore(caller)
 assert len(immediate)==1 and not assess(expected,actual,immediate[0])
 required={'find_boundaries_into'}if spec['api'].endswith('find_boundaries')else set(names[1:])
 assert required<=set(returned)and core.get_num_threads()==old_worker
