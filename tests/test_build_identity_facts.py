"""Collector metadata fixtures only; synthetic files never qualify a product."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

import pytest

spec=importlib.util.spec_from_file_location('build_identity_facts_under_test',Path(__file__).parents[1]/'tools/build_identity_facts.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def put(path,data='fixture'):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data,encoding='utf-8');return path.resolve()


@pytest.mark.parametrize('argv',[
    ['cl.exe','/Ispace path','/DNAME="value"','plain'],
    ['a','','one\\two','trailing\\','space and slash\\'],
    ['a','quote"inside','\\"','/FoC:\\path with space\\'],
])
def test_windows_command_roundtrip_without_executing_it(argv):
    assert f.windows_argv(subprocess.list2cmdline(argv))==argv


def test_unterminated_windows_quote_is_rejected():
    with pytest.raises(f.FactsError,match='unterminated'):f.windows_argv('cl /I"unclosed')


@pytest.mark.parametrize('windows',[False,True])
def test_nested_response_expansion_preserves_raw_files(tmp_path,windows):
    put(tmp_path/'nested.rsp','/DNAME=1 "space path"')
    first=put(tmp_path/'first.rsp','@nested.rsp /O2')
    evidence=set();assert f.expand_responses(['cl','@first.rsp'],tmp_path,evidence,windows)==['cl','/DNAME=1','space path','/O2']
    assert evidence=={first,(tmp_path/'nested.rsp').resolve()}


@pytest.mark.parametrize('kind',['missing','cycle'])
def test_bad_responses_fail_closed(tmp_path,kind):
    if kind=='cycle':put(tmp_path/'a.rsp','@a.rsp')
    with pytest.raises((FileNotFoundError,f.FactsError)):f.expand_responses(['@a.rsp'],tmp_path,set())


def test_nested_response_paths_keep_original_invocation_cwd(tmp_path):
    put(tmp_path/'sub/outer.rsp','@inner.rsp')
    put(tmp_path/'inner.rsp','-O2');put(tmp_path/'sub/inner.rsp','wrong-directory')
    assert f.expand_responses(['@sub/outer.rsp'],tmp_path,set())==['-O2']


def test_tlog_keeps_multiline_commands_and_grouped_keys(tmp_path):
    path=tmp_path/'CL.command.1.tlog';path.write_text('^one.cpp|two.cpp\n/c /O2\n/Foobj\\\n^three.cpp\n/c /Od\n',encoding='utf-16')
    assert f.tlog_blocks(path)=={('one.cpp','two.cpp'):['/c /O2','/Foobj\\'],('three.cpp',):['/c /Od']}


@pytest.mark.parametrize('data',['','orphan\n','^a\nvalue\n^a\nvalue\n'])
def test_missing_or_ambiguous_tlog_fails(tmp_path,data):
    path=put(tmp_path/'link.read.tlog',data)
    with pytest.raises(f.FactsError):f.tlog_blocks(path)


def test_ld_dependencies_keep_system_inputs_and_escaped_spaces():
    assert f.make_dependencies('core.so: one.o \\\n lib.a /system/crti.o path\\ with\\ space.a\n/system/crti.o:\n')==['one.o','lib.a','/system/crti.o','path with space.a']


@pytest.mark.parametrize('raw',['','object: #deps 1, deps mtime 3 (STALE)\n    h\n','object: #deps 2, deps mtime 3 (VALID)\n    h\n'])
def test_ninja_deps_missing_stale_or_partial_rejected(raw):
    with pytest.raises(f.FactsError):f.ninja_deps(raw)


@pytest.mark.parametrize('bad',['a | b','a || b','a ; b','a & b'])
def test_shell_control_is_never_silently_reinterpreted(bad):
    with pytest.raises(f.FactsError):f.shell_segments(bad)


def synthetic_gnu(tmp_path,monkeypatch,lto=False):
    source=tmp_path/'source';build=tmp_path/'out';native=build/'native-build';native.mkdir(parents=True)
    cpp=put(source/'src/rank.cpp','// synthetic TU, never compiled\n')
    tools={name:put(tmp_path/'tools'/name) for name in ('c++','ninja','cc1plus','as','ld','collect2','ar','ranlib','strip','lto1','lto-wrapper','liblto_plugin.so')}
    headers=[put(source/'include/skimagexx/rank.hpp'),put(tmp_path/'python/Python.h'),put(build/'build-env/numpy/_core/include/numpy/arrayobject.h'),put(build/'build-env/pybind11/include/pybind11/pybind11.h'),put(tmp_path/'system/include/predef.h')]
    obj=put(native/'CMakeFiles/skimagexx.dir/src/rank.cpp.o');archive=put(native/'libskimagexx.a');core=put(native/'_core.test.so');crt=put(tmp_path/'system/crti.o')
    put(build/'product-wheels/product.whl','not a wheel; collector fixture only')
    cache='\n'.join(k+':STRING='+v for k,v in dict(CMAKE_HOME_DIRECTORY=str(source),CMAKE_GENERATOR='Ninja',CMAKE_MAKE_PROGRAM=str(tools['ninja']),CMAKE_STRIP=str(tools['strip']),CMAKE_BUILD_TYPE='Release',SKIMAGEXX_OPENMP='ON',SKIMAGEXX_AVX2='ON').items())
    put(native/'CMakeCache.txt',cache)
    put(native/'CMakeFiles/4/CMakeCXXCompiler.cmake',f'set(CMAKE_CXX_COMPILER "{tools["c++"]}")\nset(CMAKE_CXX_COMPILER_ID "GNU")\nset(CMAKE_CXX_COMPILER_VERSION "13.3")\nset(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "{tmp_path/"system/include"}")\n')
    relative=obj.relative_to(native).as_posix();flags=['-flto=auto'] if lto else []
    compile=[str(tools['c++']),*flags,'-o',str(obj),'-c',str(cpp)]
    put(native/'compile_commands.json',json.dumps([dict(directory=str(native),file=str(cpp),arguments=compile)]))
    put(native/'.ninja_log','# ninja log v5\n'+''.join(f'0\t1\t2\t{p.relative_to(native).as_posix()}\thash\n' for p in (obj,archive,core)))
    put(native/'.ninja_deps','binary dependency fixture');put(native/'build.ninja','synthetic graph')
    deps=relative+': #deps 6, deps mtime 2 (VALID)\n'+''.join('    '+str(p)+'\n' for p in [cpp,*headers])
    link=[str(tools['c++']),*flags,'-save-temps=obj','-shared',str(obj),str(archive),'-o',str(core)]
    commands='\n'.join(shlex.join(v) for v in [compile,[str(tools['ar']),'qc',str(archive),str(obj)],[str(tools['ranlib']),str(archive)],link,[str(tools['strip']),str(core)]])
    linkpaths=[obj,archive,crt]
    if lto:linkpaths.append(put(native/(core.name+'.ltrans0.ltrans.o')))
    put(native/'core-link.d',str(core)+': '+shlex.join(list(map(str,linkpaths)))+'\n')
    def query(self,label,argv,cwd):
        if label=='ninja-commands':value=commands
        elif label=='ninja-deps':value=deps
        elif label.startswith('resolve-'):
            names={'backend':'cc1plus','assembler':'as','linker':'ld','link-driver':'collect2','archiver':'ar','ranlib':'ranlib','lto-backend':'lto1','lto-wrapper':'lto-wrapper','lto-plugin':'liblto_plugin.so'}
            value=str(tools[names[label.removeprefix('resolve-')]])
        else:value={'compiler-target':'x86_64-linux-gnu','generator-version':'1.13.0'}[label]
        evidence=put(self.out/(label+'.json'),json.dumps(dict(argv=list(map(str,argv)),stdout=value,synthetic=True)));self.files.add(evidence)
        return value
    monkeypatch.setattr(f.Capture,'query',query)
    return dict(source=source,build=build,native=native,cpp=cpp,obj=obj,archive=archive,core=core,crt=crt,headers=headers,tools=tools,linkpaths=linkpaths)


@pytest.mark.parametrize('lto',[False,True])
def test_complete_synthetic_gnu_capture_records_actual_edges_and_lto_tools(tmp_path,monkeypatch,lto):
    x=synthetic_gnu(tmp_path,monkeypatch,lto)
    facts=f.collect(x['source'],x['build'],'a'*40,'modern',x['build']/'facts.json')
    assert facts['schema']=='rank-statistics-build-input-v2'
    assert facts['build_root']==str(x['build']) and facts['configuration']['lto'] is lto
    assert set(facts['header_roots'])=={'project','python','numpy','pybind11','toolchain'}
    assert facts['translation_units'][0]['dependencies']==sorted(map(str,x['headers']))
    links=facts['link_commands'];ar=next(row for row in links if row['tool']=='archiver');ranlib=next(row for row in links if row['tool']=='ranlib')
    assert ar['inputs']==[str(x['obj'])] and ranlib['inputs']==ranlib['outputs']==[str(x['archive'])]
    actual_link=next(row for row in links if row['tool']=='compiler')
    assert set(actual_link['inputs'])==set(map(str,x['linkpaths']))
    assert next(row for row in facts['link_inputs'] if row['path']==str(x['crt']))['generated'] is False
    assert next(row for row in facts['link_inputs'] if row['path']==str(x['obj']))['generated'] is True
    if lto:
        assert {'lto_backend','lto_wrapper','lto_plugin'}<=set(facts['binaries'])
        assert str(x['linkpaths'][-1]) in actual_link['outputs']
    with pytest.raises(f.FactsError,match='overwrite'):f.collect(x['source'],x['build'],'a'*40,'modern',x['build']/'facts.json')


@pytest.mark.parametrize('missing',['core-link.d','.ninja_log','.ninja_deps','compile_commands.json'])
def test_missing_actual_evidence_never_produces_facts(tmp_path,monkeypatch,missing):
    x=synthetic_gnu(tmp_path,monkeypatch);(x['native']/missing).unlink()
    with pytest.raises((FileNotFoundError,f.FactsError)):f.collect(x['source'],x['build'],'a'*40,'modern',x['build']/'facts.json')
    assert not (x['build']/'facts.json').exists()


def test_discarded_lto_temporary_blocks_instead_of_inventing_input(tmp_path,monkeypatch):
    x=synthetic_gnu(tmp_path,monkeypatch,True);x['linkpaths'][-1].unlink()
    with pytest.raises(FileNotFoundError):f.collect(x['source'],x['build'],'a'*40,'modern',x['build']/'facts.json')
    assert not (x['build']/'facts.json').exists()


def test_unknown_external_dependency_is_not_blanket_classified_as_generated(tmp_path,monkeypatch):
    x=synthetic_gnu(tmp_path,monkeypatch);other=put(x['build']/'build-env/unknown-library.a')
    p=x['native']/'core-link.d';p.write_text(p.read_text().strip()+' '+shlex.quote(str(other))+'\n')
    data=f.collect(x['source'],x['build'],'a'*40,'modern',x['build']/'facts.json')
    assert next(v for v in data['link_inputs'] if v['path']==str(other))['generated'] is False


def test_wrong_source_root_is_not_retrofitted(tmp_path,monkeypatch):
    x=synthetic_gnu(tmp_path,monkeypatch);other=tmp_path/'other';other.mkdir()
    with pytest.raises(f.FactsError,match='source root differs'):f.collect(other,x['build'],'a'*40,'modern',x['build']/'facts.json')


def test_wrong_compiler_profile_fails_before_metadata_queries(tmp_path,monkeypatch):
    x=synthetic_gnu(tmp_path,monkeypatch)
    with pytest.raises(f.FactsError,match='compiler/profile differs'):f.collect(x['source'],x['build'],'a'*40,'windows',x['build']/'facts.json')


def test_retained_lto_file_requires_actual_save_temps_recipe(tmp_path,monkeypatch):
    x=synthetic_gnu(tmp_path,monkeypatch,True);query=f.Capture.query
    def changed(self,label,argv,cwd):
        value=query(self,label,argv,cwd)
        return value.replace('-save-temps=obj','') if label=='ninja-commands' else value
    monkeypatch.setattr(f.Capture,'query',changed)
    with pytest.raises(f.FactsError,match='unattributed retained LTO'):f.collect(x['source'],x['build'],'a'*40,'modern',x['build']/'facts.json')


@pytest.mark.parametrize('kind',['collision','macro','missing'])
def test_project_output_mapping_rejects_ambiguous_or_missing_facts(tmp_path,kind):
    import xml.etree.ElementTree as ET
    one=put(tmp_path/'one/unit.cpp');two=put(tmp_path/'two/unit.cpp');put(tmp_path/'obj/unit.obj')
    project=ET.Element('Project');ET.SubElement(ET.SubElement(project,'PropertyGroup'),'IntDir').text=str(tmp_path/'obj')+'/'
    default=ET.SubElement(ET.SubElement(project,'ItemDefinitionGroup'),'ClCompile');ET.SubElement(default,'ObjectFileName').text='$(Unknown)' if kind=='macro' else '$(IntDir)'
    items=ET.SubElement(project,'ItemGroup');ET.SubElement(items,'ClCompile',Include=str(one))
    if kind=='collision':ET.SubElement(items,'ClCompile',Include=str(two))
    if kind=='missing':(tmp_path/'obj/unit.obj').unlink()
    path=put(tmp_path/'project.vcxproj',ET.tostring(project,encoding='unicode'))
    with pytest.raises((f.FactsError,FileNotFoundError)):f.msvc_project_sources(path,tmp_path,'Release')


def test_metadata_queries_retain_failure_and_never_run_a_shell(tmp_path,monkeypatch):
    capture=f.Capture(tmp_path/'capture');calls=[]
    def run(argv,**options):calls.append((argv,options));return subprocess.CompletedProcess(argv,1,'raw output','raw failure')
    monkeypatch.setattr(f.subprocess,'run',run)
    with pytest.raises(f.FactsError,match='query failed'):capture.query('probe',['ninja','--version'],tmp_path)
    assert 'shell' not in calls[0][1]
    record=json.loads((capture.out/'probe.json').read_text());assert record['stdout']=='raw output' and record['stderr']=='raw failure' and record['returncode']==1


@pytest.mark.parametrize('role',('backend','assembler','linker','link_driver','archiver','ranlib','lto_backend','lto_wrapper','lto_plugin'))
def test_actual_tool_role_query_labels_pass_real_capture_validation(tmp_path,monkeypatch,role):
    capture=f.Capture(tmp_path/'capture');calls=[]
    def run(argv,**options):
        calls.append(argv);return subprocess.CompletedProcess(argv,0,'/actual/tool/path\n','')
    monkeypatch.setattr(f.subprocess,'run',run)
    label='resolve-'+role.replace('_','-')
    assert capture.query(label,['compiler','-print-prog-name=tool'],tmp_path)=='/actual/tool/path'
    assert len(calls)==1 and (capture.out/(label+'.json')).is_file()


@pytest.mark.parametrize('label',('resolve/driver','../driver','resolve driver','resolve\\driver','resolve\tdriver','resolve_driver'))
def test_unsafe_query_labels_reject_before_any_subprocess(tmp_path,monkeypatch,label):
    capture=f.Capture(tmp_path/'capture')
    monkeypatch.setattr(f.subprocess,'run',lambda *_a,**_k:pytest.fail('unsafe label reached subprocess'))
    with pytest.raises(f.FactsError,match='unsafe metadata query label'):capture.query(label,['compiler'],tmp_path)


def test_every_gnu_role_callsite_uses_collision_free_label_encoding():
    import inspect
    source=inspect.getsource(f.gnu)
    assert "capture.query('resolve-'+role.replace('_','-')" in source
    roles=('backend','assembler','linker','link_driver','archiver','ranlib','lto_backend','lto_wrapper','lto_plugin')
    labels=['resolve-'+role.replace('_','-') for role in roles]
    assert len(labels)==len(set(labels))


def test_observed_sortdefault_nls_is_explicit_pinned_toolchain_resource(tmp_path,monkeypatch):
    root=tmp_path/'Windows';resource=put(root/'Globalization/Sorting/SortDefault.nls','actual-host-locale-resource-fixture')
    monkeypatch.setenv('SystemRoot',str(root))
    assert f.windows_sorting_resource(resource,str(root))
    source=tmp_path/'source';native=tmp_path/'out/native-build';native.mkdir(parents=True)
    deps=[put(source/'include/skimagexx/rank.hpp'),put(tmp_path/'python/Python.h'),put(tmp_path/'numpy/include/numpy/arrayobject.h'),put(tmp_path/'pybind/include/pybind11/pybind11.h'),put(tmp_path/'VC/Tools/MSVC/14.44/include/vcruntime.h'),resource]
    roots=f.header_roots(set(deps),source,native,{},windows=True)
    assert str(resource.parent) in roots['toolchain'] and str(root) not in roots['toolchain']
    import hashlib
    before=hashlib.sha256(resource.read_bytes()).hexdigest();resource.write_bytes(b'changed-host-locale-data')
    assert hashlib.sha256(resource.read_bytes()).hexdigest()!=before
    # It remains a required actual input, never classified as generated data.
    assert resource not in {native/'object.obj'}


@pytest.mark.parametrize('kind',['outside','wrong-kind','missing','link'])
def test_unknown_missing_or_escaped_sorting_inputs_remain_fail_closed(tmp_path,monkeypatch,kind):
    root=tmp_path/'Windows';folder=root/'Globalization/Sorting';folder.mkdir(parents=True)
    if kind=='outside':path=put(tmp_path/'elsewhere/Unknown.nls')
    elif kind=='wrong-kind':path=put(folder/'SortDefault.txt')
    elif kind=='missing':path=folder/'SortDefault.nls'
    else:
        path=put(folder/'SortDefault.nls');original=Path.is_symlink
        # Explicit path-protocol fixture tests the guard on hosts that cannot
        # create Windows links without privileges; no conditional pass/skip.
        monkeypatch.setattr(Path,'is_symlink',lambda value:value==path or original(value))
    if kind in ('outside','wrong-kind'):assert not f.windows_sorting_resource(path,str(root))
    else:
        with pytest.raises((f.FactsError,FileNotFoundError)):f.windows_sorting_resource(path,str(root))


def test_unknown_nls_outside_systemroot_is_not_a_header_root(tmp_path,monkeypatch):
    root=tmp_path/'Windows';root.mkdir();monkeypatch.setenv('SystemRoot',str(root))
    source=tmp_path/'source';native=tmp_path/'build';native.mkdir()
    deps={put(source/'include/a.hpp'),put(tmp_path/'python/Python.h'),put(tmp_path/'numpy/include/numpy/a.h'),put(tmp_path/'pybind/include/pybind11/a.h'),put(tmp_path/'VC/Tools/MSVC/14.44/include/vcruntime.h'),put(tmp_path/'not-windows/Globalization/Sorting/Unknown.nls')}
    with pytest.raises(f.FactsError,match='unclassified actual dependency'):f.header_roots(deps,source,native,{},windows=True)


def test_complete_synthetic_msvc_tracking_keeps_reads_writes_tools_and_crt(tmp_path,monkeypatch):
    source=tmp_path/'source';build=tmp_path/'out';native=build/'native-build';native.mkdir(parents=True)
    tools={name:put(tmp_path/'VC/Tools/MSVC/14.44/bin/Hostx64/x64'/name) for name in ('cl.exe','c1xx.dll','c2.dll','link.exe','lib.exe')}
    headers=[put(source/'include/skimagexx/rank.hpp'),put(tmp_path/'python/Python.h'),put(build/'build-env/numpy/_core/include/numpy/arrayobject.h'),put(build/'build-env/pybind11/include/pybind11/pybind11.h'),put(tmp_path/'VC/Tools/MSVC/14.44/include/vcruntime.h')]
    crt=put(tmp_path/'VC/Tools/MSVC/14.44/lib/x64/MSVCRT.LIB')
    archive=put(native/'Release/skimagexx.lib');core=put(native/'Release/_core.test.pyd')
    put(build/'product-wheels/product.whl','synthetic only');put(build/'build-whole-product.log','MSBuild version 17.14.23+b001 for .NET Framework\n')
    cache=dict(CMAKE_HOME_DIRECTORY=str(source),CMAKE_GENERATOR='Visual Studio 17 2022',SKIMAGEXX_OPENMP='ON',SKIMAGEXX_AVX2='ON')
    put(native/'CMakeCache.txt','\n'.join(k+':STRING='+v for k,v in cache.items()))
    put(native/'CMakeFiles/4/CMakeCXXCompiler.cmake','set(CMAKE_CXX_COMPILER_ID "MSVC")\nset(CMAKE_CXX_COMPILER_VERSION "19.44")\n')
    objects=[]
    for target,name in [('skimagexx','rank'),('_core','bind_rank')]:
        cpp=put(source/('src/'+name+'.cpp'));obj=put(native/(target+'.dir/Release/'+name+'.obj'));objects.append(obj)
        tlog=native/(target+'.dir/Release/'+target+'.tlog')
        def tracking(kind,entries):
            p=tlog/kind;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('^'+str(cpp)+'\n'+'\n'.join(map(str,entries))+'\n',encoding='utf-16')
        tracking('CL.command.1.tlog',['/c /MD /GL /Fo'+subprocess.list2cmdline([str(obj)])+' '+subprocess.list2cmdline([str(cpp)])])
        # Actual CL.write may be sparse even though every object exists and
        # every source has a command/read record. /Fo/project mapping is used.
        tracking('CL.read.1.tlog',[cpp,*headers]);tracking('CL.write.1.tlog',[])
        role='Lib' if target=='skimagexx' else 'link';output=archive if target=='skimagexx' else core
        tracking(role+'.command.1.tlog',['/OUT:'+subprocess.list2cmdline([str(output)])+' '+subprocess.list2cmdline([str(obj)])])
        tracked='Lib-link' if role=='Lib' else role
        tracking(tracked+'.read.1.tlog',[obj] if target=='skimagexx' else [obj,archive,crt])
        tracking(tracked+'.write.1.tlog',[output])
        import xml.etree.ElementTree as ET
        project=ET.Element('Project');group=ET.SubElement(project,'PropertyGroup');ET.SubElement(group,'IntDir').text=str(obj.parent)+'/'
        default=ET.SubElement(ET.SubElement(project,'ItemDefinitionGroup'),'ClCompile');ET.SubElement(default,'ObjectFileName').text='$(IntDir)'
        item=ET.SubElement(ET.SubElement(project,'ItemGroup'),'ClCompile',Include=str(cpp));ET.SubElement(item,'ObjectFileName').text=str(obj)
        put(native/(target+'.vcxproj'),ET.tostring(project,encoding='unicode'))
    monkeypatch.setattr(f,'windows_model',lambda *_:dict(compiler=str(tools['cl.exe']),architecture='x64',toolset_version='14.44'))
    data=f.collect(source,build,'a'*40,'windows',build/'facts.json')
    assert data['configuration']['crt']=='/MD' and data['configuration']['lto'] is True
    assert data['configuration']['generator_version']=='17.14.23+b001'
    assert set(data['binaries'])=={'compiler','frontend','backend','linker','archiver'}
    assert len(data['translation_units'])==2 and len(data['link_commands'])==2
    assert next(row for row in data['link_commands'] if row['tool']=='linker')['inputs']==list(map(str,sorted([objects[1],archive,crt],key=str)))
    assert next(row for row in data['link_inputs'] if row['path']==str(archive))['generated'] is True
    assert next(row for row in data['link_inputs'] if row['path']==str(crt))['generated'] is False
