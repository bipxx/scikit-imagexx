"""Actual loop/terminal AST with inert supplied-call and writer failures only."""
import ast,copy,json,traceback
from pathlib import Path
from types import SimpleNamespace
import pytest
import finite_radon_capture_controller as C
TREE=ast.parse(Path(C.__file__).read_text());MAIN=next(n for n in TREE.body if isinstance(n,ast.FunctionDef)and n.name=='main')
LOOP=next(n for n in ast.walk(MAIN)if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='spec'and isinstance(n.iter,ast.Name)and n.iter.id=='specs')
TERMINAL=next(n for n in ast.walk(MAIN)if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)and isinstance(n.value.func,ast.Name)and n.value.func.id=='dump'and 'last_case_diagnostics' in ast.unparse(n))

class FakeNP:
    def __init__(self):self.policy={'all':'warn'};self.handler=None;self.buffer=8192;self.printing={'precision':8}
    def geterr(self):return dict(self.policy)
    def geterrcall(self):return self.handler
    def getbufsize(self):return self.buffer
    def get_printoptions(self):return dict(self.printing)
    def seterr(self,**v):self.policy=v
    def seterrcall(self,v):self.handler=v
    def setbufsize(self,v):self.buffer=v
    def set_printoptions(self,**v):self.printing=v
class Saved:
    tags=0xffff;control=0x37f;mxcsr=0x1f80
    def __init__(self,b):self.raw=b
    def __bytes__(self):return self.raw

@pytest.mark.parametrize('fault',('none','observer','codec','state','postvalidation','completed-writer','failure-writer','started-writer','restore'))
def test_actual_failure_paths_preserve_consumption_immediate_and_restore(tmp_path,fault):
    binding=tmp_path/'binding.json';binding.write_text('{}');go=tmp_path/'go.json';go.write_text('{}')
    saved=Saved(bytes(range(32)));current=[saved];np=FakeNP();args=SimpleNamespace(root_go=go,root_go_sha256=C.H(go),binding=binding,binding_sha256=C.H(binding),output=tmp_path,mode='capture',profile='minimum',prefix_api=None,cache_state='cold')
    spec={'id':'transform.frt2/inert','api':'frt2'};q=dict(source='synthetic',case_ids=[spec['id']],helper={},git_argv=[])
    g=dict(C.__dict__,args=args,q=q,go={},env=lambda:current[0],state=lambda v:dict(raw=bytes(v).hex(),synthetic=True),restore=lambda v:current.__setitem__(0,Saved(b'\xfe'*32)if fault=='restore'else v),np=np,specs=[spec],capture_rows=[],journal=tmp_path/'journal.jsonl',target_call=object(),counts=dict(target_attempted=0),controls=[],entries=[],codec_metadata_controls=[],initial=saved,final=saved,failure=None,identity={'synthetic':True},module_request={'synthetic':True},last_case_diagnostics=None,total_public=0,immediate_calls=[],check_deadline=lambda go:None,LIMITS=())
    pair=dict(before={'raw':bytes(saved).hex()},after={'raw':'ff'*32},numpy_before={'synthetic':'before'},numpy_after={'synthetic':'after'});calls=[]
    def observe(call,spec,np):
        calls.append('inert');g['call_count']=g['public_attempts']=1;g['total_public']=1;g['counts']['target_attempted']=1;g['immediate_calls'].append(copy.deepcopy(pair));current[0]=Saved(b'\xff'*32)
        if fault in('observer','codec','failure-writer','restore'):raise ValueError('INERT_OBSERVER_AFTER_RETURN')
        if fault=='state':np.policy={'changed':'raise'}
        if fault=='postvalidation':binding.write_text('changed')
        return dict(id=spec['id'],synthetic_result_only=True)
    def append(path,row):
        if(fault=='failure-writer'and row['event']=='failed_case_diagnostics')or(fault=='completed-writer'and row['event']=='completed')or(fault=='started-writer'and row['event']=='started'):
            with path.open('a')as f:f.write('{"partial":')
            raise OSError('INERT_PARTIAL_WRITER')
        C.append(path,row)
    g.update(observe=observe,append=append)
    try:exec(compile(ast.Module(body=[copy.deepcopy(LOOP)],type_ignores=[]),C.__file__,'exec'),g)
    except BaseException as error:
        assert fault!='none';g['failure']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
    else:assert fault=='none'
    exec(compile(ast.Module(body=[copy.deepcopy(TERMINAL)],type_ignores=[]),C.__file__,'exec'),g)
    terminal=json.loads((tmp_path/'terminal.json').read_bytes());raw=(tmp_path/'journal.jsonl').read_text()
    if fault=='none':
        assert terminal['recipe_observations']==1 and terminal['last_case_diagnostics']['caller_restore']['exact32']
        assert [json.loads(s)['event']for s in raw.splitlines()]==['started','completed']
    else:
        assert terminal['status']=='FAILED_FINITE_RADON_CONTROLLER'and terminal['recipe_observations']==0
        d=terminal['last_case_diagnostics'];assert d['failure']['traceback']and d['retry']is False
        if fault=='started-writer':assert calls==[]and d['public_attempts']==0
        else:
            assert calls==['inert']and d['public_attempts']==d['target_entries']==1 and d['immediate']==[pair]
            if fault=='restore':
                assert not d['caller_restore']['exact32']and d['failure']['type']=='ValueError'and d['propagated_failure']['type']=='AssertionError'
            else:assert d['caller_restore']['exact32']and d['caller_restore']['saved']==d['caller_restore']['actual']
        if fault=='failure-writer':assert d['failure_writer_error']['type']=='OSError'and terminal['failure']['type']=='ValueError'
        assert '"event": "completed"'not in raw
    if fault!='restore':assert bytes(current[0])==bytes(saved)
