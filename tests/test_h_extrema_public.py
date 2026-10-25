"""Installed partial H-extrema gate; frozen816 partition, no private adapter.

The fresh child authenticates originals before importing the actual production
package. Integer observations are numerical gates; unsupported float/promotion
rows remain mandatory followups and unsafe originals remain14 blocked metadata
rows. This source/collection preparation does not claim a qualified product.
"""
import ast
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
MIGRATION=ROOT/'coverage/evidence/h-extrema-public-source-composition.json'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def public_assertion_nodes():
    source=ROOT/'tests/test_h_extrema_native_pipeline.py'
    nodes=ast.parse(source.read_bytes()).body;names=('initial_native_spec','retain','test_exact_private_pipeline','test_safe_metadata_only_intake_guards')
    selected=[copy.deepcopy(n) for n in nodes if isinstance(n,ast.FunctionDef) and n.name in names];assert len(selected)==len(names)
    numerical=next(n for n in selected if n.name=='test_exact_private_pipeline');before=copy.deepcopy(numerical);changes=[]
    for node in ast.walk(numerical):
        if isinstance(node,ast.If) and ast.unparse(node.test)=="name in PLAN['legacy_unsafe_ids']":
            last=node.body[-1];assert isinstance(last,ast.Expr) and ast.unparse(last.value)=="pytest.skip('unsafe original intake is not an upstream exception contract')"
            node.body[-1]=ast.Return(value=None);changes.append((node,last))
    assert len(changes)==1
    inverse=copy.deepcopy(numerical)
    for node in ast.walk(inverse):
        if isinstance(node,ast.If) and ast.unparse(node.test)=="name in PLAN['legacy_unsafe_ids']":node.body[-1]=changes[0][1]
    assert ast.dump(inverse,include_attributes=False)==ast.dump(before,include_attributes=False)
    return selected


def audit_installed(output):
    assert not any(n=='skimagexx' or n.startswith('skimagexx.') for n in sys.modules)
    assert not any(n in os.environ for n in ('SKIMAGEXX_H_EXTREMA_CORE','SKIMAGEXX_EXPECT_OPENMP_OFF','SKIMAGEXX_RECONSTRUCTION_CORE'))
    evidence=json.loads(MIGRATION.read_bytes());policy=json.loads((ROOT/'coverage/contracts/h-extrema-integer.json').read_bytes())
    for name,want in evidence['effective_source_raw_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name
    profile='windows' if os.name=='nt' else 'minimum' if importlib.metadata.version('numpy')=='1.24.4' else 'modern'
    reference=load('installed_h_reference_auth',ROOT/'tests/h_extrema_reference.py');plan,authentication=reference.authenticate(profile)
    legacy=load('installed_h_exact_legacy',ROOT/'tests/test_h_extrema_contracts.py');boundaries=load('installed_h_exact_boundaries',ROOT/'tests/test_h_extrema_boundaries.py')
    import numpy as np
    matrix=json.loads((ROOT/'coverage/contracts/h-extrema.json').read_bytes());specs=legacy.cases()+boundaries.cases()
    assert [s['id'] for s in specs]==policy['all_case_ids']
    assert hashlib.sha256((ROOT/'coverage/contracts/h-extrema.json').read_bytes()).hexdigest()==policy['reference_matrix_sha256']
    import skimagexx as xx
    core=xx._core;wrapper=importlib.import_module('skimagexx._h_extrema');reconstruction=importlib.import_module('skimagexx._reconstruction');morphology=importlib.import_module('skimagexx.morphology')
    from skimagexx.coverage import native_functions
    assert core.__name__=='skimagexx._core' and wrapper._core is reconstruction._core is core
    assert wrapper.reconstruction is reconstruction.reconstruction
    parent=Path(xx.__file__).resolve().parent
    assert all(Path(module.__file__).resolve().parent==parent for module in (core,wrapper,reconstruction,morphology))
    assert core.get_num_threads()==xx.get_num_threads()==1 and core.has_openmp is True
    assert core.h_extrema_interface=='h-extrema-integer-v1' and core.h_extrema_full_API is False
    assert core.h_extrema_integer_widths==(8,16,32,64)
    assert core.reconstruction_interface=='reconstruction-v1' and type(core.reconstruction_capabilities) is int and core.reconstruction_capabilities==3
    assert type(core._consumer_gil_assertions) is bool
    for name in ('h_extrema_validate_metadata','h_extrema_integer_range','h_extrema_integer_seed','h_extrema_integer_mask','h_extrema_counts','reconstruction_run','reconstruction_counts'):
        entry=getattr(core,name);assert inspect.isbuiltin(entry) and entry.__name__==name and entry.__module__==core.__name__
    for name in ('h_maxima','h_minima'):
        function=vars(morphology)[name];assert function is vars(wrapper)[name] and function.__module__=='skimagexx._h_extrema'
        assert inspect.signature(function)==inspect.signature(legacy.ORIGINAL[name])
        assert morphology.__all__.count(name)==1 and native_functions().count('morphology.'+name)==1
    for name in ('python/skimagexx/_h_extrema.py','python/skimagexx/_reconstruction.py'):
        installed=wrapper if name.endswith('_h_extrema.py') else reconstruction
        assert hashlib.sha256(Path(installed.__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==evidence['effective_source_canonical_lf_sha256'][name]
    nodes=public_assertion_nodes();rows=[];guards=[];journal_path=output.with_suffix('.jsonl');journal=journal_path.open('x',encoding='utf-8',newline='\n')
    namespace=dict(pytest=pytest,os=os,sys=sys,json=json,hashlib=hashlib,np=np,CORE=core,OWNED=wrapper,PLAN=plan,POLICY=policy,SPECS=specs,
        MATRIX=matrix,LEGACY=legacy,BOUNDARIES=boundaries,_profile=profile,ROWS=rows,GUARDS=guards,JOURNAL=journal,
        FORBIDDEN={f.__code__ for f in legacy.ORIGINAL.values()}|{legacy.ORIGINAL_REC.__code__,legacy.R.SOURCE.rank_order.__code__})
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<authenticated-installed-H-assertions>','exec'),namespace)
    assert [s['id'] for s in specs if namespace['initial_native_spec'](s) and s['id'] not in plan['legacy_unsafe_ids']]==policy['initial_native_ids']
    previous=xx.set_fallback_enabled(False);assert xx.set_fallback_enabled(False) is False
    failure=None
    try:
        for workers in (1,4):
            for spec in specs:namespace['test_exact_private_pipeline'](spec,workers)
        namespace['test_safe_metadata_only_intake_guards']()
    except BaseException as error:
        failure=dict(type=type(error).__name__,message=str(error));raise
    finally:
        journal.close();core.set_num_threads(1);xx.set_fallback_enabled(previous)
        report=dict(schema='h-extrema-installed-public-partition-v1',profile=profile,source_authentication=authentication,
            core=dict(path=str(Path(core.__file__).resolve()),sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest()),
            rows=rows,guards=guards,failure=failure,full_API_complete=False,private_adapter=False,operation_timings=0)
        with output.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    from collections import Counter
    counts=Counter(r['status'] for r in rows)
    assert counts==dict(native=848,**{'mandatory-followup':756,'source-unsafe-blocked':28}) and len(rows)==1632
    per_worker={str(w):dict(Counter(r['status'] for r in rows if r['workers']==w)) for w in (1,4)}
    assert all(v==dict(native=424,**{'mandatory-followup':378,'source-unsafe-blocked':14}) for v in per_worker.values())
    assert all(r.get('actual_calls',0)==0 for r in rows if r['status']=='source-unsafe-blocked')
    assert not any(r.get('forwarding_attempts') for r in rows)
    print(json.dumps(dict(profile=profile,partition=dict(counts),per_worker=per_worker,logical_ids=len(specs),rows=len(rows),private_adapter=False,full_API_complete=False,operation_timings=0),sort_keys=True))


def test_installed_h_extrema_exact_partition(tmp_path):
    output=tmp_path/'installed-h-extrema.json'
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--audit-installed',str(output)],capture_output=True,text=True,encoding='utf-8',timeout=600)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout);assert report['rows']==1632 and report['logical_ids']==816 and report['partition']==dict(native=848,**{'mandatory-followup':756,'source-unsafe-blocked':28})
    assert report['private_adapter'] is False and report['full_API_complete'] is False


if __name__=='__main__':
    assert len(sys.argv)==3 and sys.argv[1]=='--audit-installed'
    audit_installed(Path(sys.argv[2]))
