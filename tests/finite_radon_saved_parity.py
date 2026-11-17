"""Prospective saved-only consumer. No expected rows exist; fail unbound."""
import hashlib,inspect,json
from pathlib import Path
from finite_radon_contract import cases

def portable_record(v):
    if isinstance(v,list):return [portable_record(x)for x in v]
    if not isinstance(v,dict):return v
    skip=set()
    if v.get('pointer_delta_role')=='nonsharing-diagnostic-only':skip.add('pointer_delta')
    if {'raw','control','status','mxcsr'}<=v.keys():skip.add('raw')
    if 'handler_id'in v:skip.add('handler_id')
    return {k:portable_record(x)for k,x in v.items()if k not in skip}

def run_saved(binding,call,observe,envelope,np):
    assert binding is not None,'FINITE_RADON_UNBOUND: root-admitted originals required'
    d=json.loads(Path(binding).read_bytes());assert d['status']=='ROOT_ADMITTED_FINITE_RADON_SAVED_ONLY'
    for stem in('observations','raw_admission','candidate_source','observer_source','envelope_source'):
        assert hashlib.sha256(Path(d[stem+'_path']).read_bytes()).hexdigest()==d[stem+'_sha256']
    assert Path(inspect.getsourcefile(call)).resolve()==Path(d['candidate_source_path']).resolve()
    assert Path(inspect.getsourcefile(observe)).resolve()==Path(d['observer_source_path']).resolve()
    assert Path(inspect.getsourcefile(envelope)).resolve()==Path(d['envelope_source_path']).resolve()
    admission=json.loads(Path(d['raw_admission_path']).read_bytes());assert admission['passed']and admission['source']==d['reference_source']
    assert admission['observations_sha256']==d['observations_sha256']and np.__version__==d['numpy']
    rows=json.loads(Path(d['observations_path']).read_bytes());specs=[s for s in cases()if s['api']==d['api']]
    assert [r['id']for r in rows]==[s['id']for s in specs]==d['case_ids']
    for spec,gold in zip(specs,rows):
        actual=envelope(call,observe,spec,np,gold['caller_restore']['saved'])
        assert actual['caller_restore']['exact32']and actual['caller_restore']['saved']['raw']==actual['caller_restore']['actual']['raw']
        assert portable_record(actual['target_immediate'])==portable_record(gold['target_immediate'])
        assert portable_record(actual)==portable_record(gold)
    return dict(rows=len(rows),scope='Saved observed scope only; genuine native ownership separate')
