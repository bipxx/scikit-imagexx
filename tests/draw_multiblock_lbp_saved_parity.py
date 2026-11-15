"""Saved-only94row parity consumer. No original target is resolved or called."""
import hashlib,inspect,json
from pathlib import Path
import draw_multiblock_lbp_overlap_contract as contract
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def portable_record(value):
    """Derived comparison view; raw observations remain immutable.

    Exclude only explicitly diagnostic nonsharing offsets, callback identity
    and FENV instruction/data-address bytes; preserve control/status/MXCSR.
    """
    if isinstance(value,list):return [portable_record(v)for v in value]
    if not isinstance(value,dict):return value
    excluded=set()
    if value.get('pointer_delta_role')=='nonsharing-diagnostic-only':excluded.add('pointer_delta')
    if {'raw','control','status','mxcsr'}<=set(value):excluded.add('raw')
    if 'handler_id'in value:excluded.add('handler_id')
    return {k:portable_record(v)for k,v in value.items()if k not in excluded}


def run_saved_comparison(binding_path,candidate,observer,envelope,np):
    assert binding_path is not None,'UNBOUND: independently admitted authentic94observations required'
    data=json.loads(Path(binding_path).read_bytes())
    assert data['status']=='ROOT_ADMITTED_LBP_SAVED_ONLY'
    for stem in('observations','terminal','raw_admission','observer_source','candidate_source','envelope_source'):
        assert H(data[stem+'_path'])==data[stem+'_sha256'],stem
    admission=json.loads(Path(data['raw_admission_path']).read_bytes())
    assert admission['passed']is True and admission['status']=='PASS_SCOPED_94_ORIGINAL_OBSERVATIONS_EACH_PROFILE'
    assert H(data['raw_admission_path'])=='1bcfff2be1b241b989a6142a756131474e89e756dd30047b0e653dde3d822726'
    assert data['profile']in('minimum','modern')
    raw_key=f"build/draw-multiblock-lbp-codec-capture-v1/capture-v1/{data['profile']}/observations.json"
    assert admission['raw_files'][raw_key]['sha256']==data['observations_sha256']
    assert admission['source']=='339fe6812ec07997cb9861c7755378ea977fa1d4'
    assert Path(inspect.getsourcefile(candidate)).resolve()==Path(data['candidate_source_path']).resolve()
    assert Path(inspect.getsourcefile(observer)).resolve()==Path(data['observer_source_path']).resolve()
    assert Path(inspect.getsourcefile(envelope)).resolve()==Path(data['envelope_source_path']).resolve()
    assert candidate.__module__.startswith('skimagexx.')and candidate.__name__=='draw_multiblock_lbp'
    assert np.__version__==data['numpy']
    rows=json.loads(Path(data['observations_path']).read_bytes());terminal=json.loads(Path(data['terminal_path']).read_bytes());specs=contract.cases()
    assert len(rows)==94 and [r['id']for r in rows]==[s['id']for s in specs]==data['case_ids']
    assert terminal['recipe_observations']==94 and terminal['status']=='COMPLETE_PENDING_INDEPENDENT_RAW_ADMISSION'
    # Root supplies the already admitted full32/NumPy/callback envelope, seeded
    # from each saved caller state; no new FENV engine or masked comparison.
    for spec,expected in zip(specs,rows):
        actual=envelope(candidate,observer,spec,np,expected['caller_restore']['saved'])
        assert actual['id']==spec['id']and actual['supplied_call_attempts']==1 and all(actual['checks'].values())
        assert actual['caller_restore']['exact32']is True
        assert portable_record(actual)==portable_record(expected),spec['id']
    return dict(rows=94,original_calls=0,candidate_calls=94,scope='Observed assessed-domain parity only; native entry ownership/genuine SDK/performance separate')
