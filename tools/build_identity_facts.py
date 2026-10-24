"""Collect prospective build-host facts, never reconstruct them from binaries.

Only metadata queries are executed. Missing dependency logs, stale Ninja deps,
unreadable inputs (including discarded LTO temporaries), wrappers, or unsupported
command syntax block capture. This collector does not establish public parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET


SCHEMA = 'rank-statistics-build-input-v2'


class FactsError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise FactsError(message)


def actual(path, cwd=None):
    path = Path(path)
    if not path.is_absolute():
        require(cwd is not None, 'relative path without recorded working directory')
        path = Path(cwd) / path
    resolved = path.resolve(strict=True)
    require(resolved.is_file(), 'missing actual input file: ' + str(path))
    return resolved


def text(path):
    data = Path(path).read_bytes()
    return data.decode('utf-16') if data.startswith((b'\xff\xfe', b'\xfe\xff')) else data.decode('utf-8-sig')


def windows_argv(command):
    """MS CRT quoting, independent of the platform running metadata tests."""
    result = []; i = 0
    while i < len(command):
        while i < len(command) and command[i].isspace(): i += 1
        if i == len(command): break
        value = []; quoted = False
        while i < len(command) and (quoted or not command[i].isspace()):
            slashes = 0
            while i < len(command) and command[i] == '\\': slashes += 1; i += 1
            if i < len(command) and command[i] == '"':
                value.extend('\\' * (slashes // 2))
                if slashes % 2: value.append('"'); i += 1
                elif quoted and i + 1 < len(command) and command[i + 1] == '"': value.append('"'); i += 2
                else: quoted = not quoted; i += 1
            else:
                value.extend('\\' * slashes)
                if i < len(command) and (quoted or not command[i].isspace()): value.append(command[i]); i += 1
        require(not quoted, 'unterminated MSVC command quote')
        result.append(''.join(value))
    return result


def expand_responses(argv, cwd, evidence, windows=False, chain=()):
    result = []
    for value in argv:
        if not value.startswith('@'):
            result.append(value); continue
        require(len(chain) < 16, 'response expansion too deep')
        path = actual(value[1:], cwd)
        require(path not in chain, 'cyclic response file')
        require(path.stat().st_size <= 16 * 1024 * 1024, 'unbounded response file')
        evidence.add(path)
        tokens = windows_argv(text(path)) if windows else shlex.split(text(path), posix=True)
        # Both GCC and the MS tools open nested @files against the invocation
        # working directory, not the containing response file's directory.
        result.extend(expand_responses(tokens, cwd, evidence, windows, chain + (path,)))
    require(not any(v.startswith('@') for v in result), 'unexpanded response argument')
    return result


def cmake_cache(path):
    result = {}
    for line in text(path).splitlines():
        match = re.fullmatch(r'([^/#][^:=]*):[^=]+=(.*)', line)
        if match: result[match[1]] = match[2]
    return result


def cmake_sets(path):
    return {m[1]: m[2] if m[2] is not None else m[3]
            for m in re.finditer(r'set\((\w+)\s+(?:"([^"]*)"|([^\s()]+))\)', text(path))}


def tlog_blocks(path):
    """Keep complete keys and multiline commands; never discard grouped rows."""
    records = {}; key = None
    for line in text(path).splitlines():
        if line.startswith('^'):
            key = tuple(line[1:].split('|'))
            require(all(key) and key not in records, 'duplicate/empty tlog key')
            records[key] = []
        elif line:
            require(key is not None, 'tlog content without tracking key')
            records[key].append(line)
    require(records, 'empty tracking log: ' + str(path))
    return records


def ninja_log(path):
    lines = text(path).splitlines()
    require(lines and re.fullmatch(r'# ninja log v[567]', lines[0]), 'unsupported/missing Ninja execution log')
    outputs = set()
    for line in lines[1:]:
        fields = line.split('\t')
        require(len(fields) == 5 and all(v.isdigit() for v in fields[:3]), 'malformed Ninja execution row')
        outputs.add(fields[3])
    require(outputs, 'no executed Ninja outputs')
    return outputs


def ninja_deps(raw):
    result = {}; key = None; count = None
    for line in raw.splitlines() + ['']:
        if not line:
            if key is not None:
                require(len(result[key]) == count, 'Ninja dependency count differs')
            key = None; continue
        if not line.startswith('    '):
            match = re.fullmatch(r'(.+): #deps (\d+), deps mtime \d+ \(VALID\)', line)
            require(match is not None, 'stale/unsupported Ninja dependency block')
            key, count = match[1], int(match[2]); require(key not in result, 'duplicate Ninja dependency output')
            result[key] = []
        else:
            require(key is not None, 'orphan Ninja dependency')
            result[key].append(line[4:])
    require(result, 'empty actual Ninja dependencies')
    return result


def make_dependencies(raw):
    """GNU ld --dependency-file: retain every prerequisite, not phony rules."""
    joined = raw.replace('\\\n', '')
    result = []
    for line in joined.splitlines():
        if not line.strip(): continue
        fields=re.split(r':(?:\s|$)',line,maxsplit=1)
        require(len(fields)==2, 'malformed linker dependency record')
        _, values = fields
        result.extend(shlex.split(values, posix=True))
    require(result, 'empty actual linker dependencies')
    return result


def shell_segments(command):
    lexer = shlex.shlex(command, posix=True, punctuation_chars=';&|'); lexer.whitespace_split = True
    segments = [[]]
    for token in lexer:
        if token == '&&': segments.append([])
        else:
            require(token not in (';', '|', '||', '&'), 'unsupported shell control in build recipe')
            segments[-1].append(token)
    return [v for v in segments if v and v != [':']]


class Capture:
    def __init__(self, out):
        self.out = Path(out)
        require(not self.out.exists(), 'never overwrite earlier facts evidence')
        self.out.mkdir(parents=True)
        self.files = set()

    def query(self, label, argv, cwd):
        require(re.fullmatch(r'[a-z0-9-]+', label), 'unsafe metadata query label')
        record = self.out / (label + '.json'); require(not record.exists(), 'duplicate metadata query')
        process = subprocess.run(list(map(str, argv)), cwd=cwd, capture_output=True, text=True, check=False)
        record.write_text(json.dumps(dict(argv=list(map(str, argv)), cwd=str(cwd), returncode=process.returncode,
                                          stdout=process.stdout, stderr=process.stderr), indent=2) + '\n')
        self.files.add(record)
        require(process.returncode == 0, 'metadata query failed: ' + label)
        return process.stdout.strip()


def single(paths, message):
    paths = list(paths); require(len(paths) == 1, message); return paths[0]


def windows_sorting_resource(path, system_root):
    """Observed CL locale reads, pinned as data; no whole-Windows allowance."""
    if system_root is None:return False
    root=Path(system_root)
    require(root.is_absolute() and root.is_dir(),'actual SystemRoot unavailable')
    folder=root/'Globalization/Sorting';path=Path(path)
    if path.parent!=folder or path.suffix.lower()!='.nls':return False
    require(re.fullmatch(r'[A-Za-z0-9_-]+\.nls',path.name,re.I) is not None,'unsupported sorting resource name')
    # Follow neither a file link nor a parent junction/symlink outside the
    # declared host directory. The normal producer hashes the actual read file.
    for entry in (root,root/'Globalization',folder,path):
        require(not entry.is_symlink() and not (hasattr(entry,'is_junction') and entry.is_junction()),'sorting resource symlink/junction rejected')
    require(path.absolute()==path.resolve(strict=True),'sorting resource escapes declared host directory')
    actual(path)
    return True


def header_roots(dependencies, source, native, compiler_info, windows=False):
    roots = dict(project=[source, native], python=[], numpy=[], pybind11=[], toolchain=[])
    for path in dependencies:
        parts = path.parts
        if path.name.lower() == 'python.h': roots['python'].append(path.parent)
        for i, part in enumerate(parts):
            if part.lower() == 'numpy' and i and parts[i-1].lower() == 'include': roots['numpy'].append(Path(*parts[:i]))
            if part.lower() == 'pybind11' and i and parts[i-1].lower() == 'include': roots['pybind11'].append(Path(*parts[:i]))
    if windows:
        resource_folders=set()
        for path in dependencies:
            match = re.match(r'(.+[/\\]MSVC[/\\][^/\\]+[/\\]include)(?:[/\\]|$)', str(path), re.I)
            kit = re.match(r'(.+[/\\]Windows Kits[/\\]\d+[/\\]Include[/\\][^/\\]+)(?:[/\\]|$)', str(path), re.I)
            if match or kit: roots['toolchain'].append(Path((match or kit)[1]))
            # MSVC tracking also records resources/support DLLs actually read
            # by CL. Pin these as toolchain inputs rather than dropping them.
            if path.suffix.lower() in ('.dll','.exe'): roots['toolchain'].append(path.parent)
            if windows_sorting_resource(path,os.environ.get('SystemRoot')):
                resource_folders.add(path.parent);roots['toolchain'].append(path.parent)
        # A directory root is required by the identity schema. Limit this
        # special root to the observed NLS resource kind, not arbitrary files.
        for path in dependencies:
            if path.parent in resource_folders:require(windows_sorting_resource(path,os.environ.get('SystemRoot')),'unclassified input beside sorting resource')
    else:
        roots['toolchain'] = [Path(v) for v in compiler_info.get('CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES', '').split(';') if v]
    roots = {k: sorted({p.resolve(strict=True) for p in values if p.is_dir()}, key=str) for k, values in roots.items()}
    require(all(roots.values()), 'missing actual project/Python/NumPy/pybind11/toolchain header roots')
    observed = set()
    for path in dependencies:
        matches = [(len(str(folder)), name) for name, folders in roots.items() for folder in folders if path.is_relative_to(folder)]
        require(matches, 'unclassified actual dependency: ' + str(path))
        observed.add(max(matches)[1])
    require(observed == set(roots), 'actual used dependency closure does not cover all five header providers')
    return {name: list(map(str, values)) for name, values in roots.items()}


def gnu(native, cache, info, capture):
    require(cache.get('CMAKE_GENERATOR') == 'Ninja', 'GNU capture requires actual Ninja execution/dependency evidence')
    compiler = actual(info['CMAKE_CXX_COMPILER']); ninja = actual(cache['CMAKE_MAKE_PROGRAM'])
    log = native / '.ninja_log'; compdb = native / 'compile_commands.json'; binarydeps = native / '.ninja_deps'
    capture.files.update((log, compdb, binarydeps, native/'build.ninja'))
    executed = {str(actual(v, native)) for v in ninja_log(log)}
    commands = capture.query('ninja-commands', [ninja, '-C', native, '-t', 'commands'], native)
    deps = ninja_deps(capture.query('ninja-deps', [ninja, '-C', native, '-t', 'deps'], native))
    recipes = [expand_responses(v, native, capture.files) for line in commands.splitlines() for v in shell_segments(line)]
    units = []; used = set(); binaries = dict(compiler=str(Path(info['CMAKE_CXX_COMPILER']).absolute()),generator=str(Path(cache['CMAKE_MAKE_PROGRAM']).absolute())); matched_objects = set()
    for row in json.loads(text(compdb)):
        cwd = Path(row['directory']).resolve(strict=True)
        recorded = row.get('arguments') or shlex.split(row['command'])
        require('-o' in recorded and '-c' in recorded, 'compdb lacks explicit source/object')
        obj = Path(recorded[recorded.index('-o')+1]);obj=(cwd/obj).resolve() if not obj.is_absolute() else obj.resolve()
        if str(obj) not in executed: continue
        obj=actual(obj);source=actual(row['file'],cwd)
        matches = [v for v in recipes if '-c' in v and '-o' in v
                   and actual(v[v.index('-c')+1], native) == source and actual(v[v.index('-o')+1], native) == obj]
        argv = single(matches, 'missing/ambiguous actual compile recipe')
        require(actual(argv[0], native) == compiler, 'compiler wrapper or conflicting compiler')
        dep_key = single((k for k in deps if actual(k, native) == obj), 'missing actual used headers for object')
        headers = sorted({actual(p, native) for p in deps[dep_key]} - {source}, key=str)
        require(headers, 'empty used header closure'); used.update(headers); matched_objects.add(str(obj))
        target = obj.relative_to(native).as_posix()
        units.append(dict(source=str(source), target=target, argv=argv, dependencies=list(map(str, headers)), outputs=[str(obj)]))
    require(units, 'no actually executed translation units')
    require({p for p in executed if p.endswith(('.o','.obj'))} == matched_objects, 'executed object/TU coverage differs')
    # Query the exact compiler driver configuration; record this resolution as
    # metadata, without pretending it is a trace of backend execution.
    all_args = [a for row in units for a in row['argv']]
    require(not any(a.startswith(('-B','-specs','--specs','--sysroot','-isysroot','-fuse-ld=','-fplugin=','-wrapper')) for a in all_args), 'custom compiler tool/header routing needs an explicit capture provider')
    for role, name in [('backend','cc1plus'),('assembler','as'),('linker','ld'),('link_driver','collect2'),('archiver','ar'),('ranlib','ranlib')]:
        resolved = capture.query('resolve-'+role.replace('_','-'), [compiler, '-print-prog-name='+name], native)
        selected = Path(resolved) if Path(resolved).is_absolute() else Path(shutil.which(resolved) or '')
        require(selected.is_absolute(), 'unresolved actual compiler tool: '+name); actual(selected);binaries[role] = str(selected)
    if cache.get('CMAKE_STRIP'):actual(cache['CMAKE_STRIP']);binaries['stripper']=str(Path(cache['CMAKE_STRIP']).absolute())
    links = []
    for argv in recipes:
        if not argv or argv[0] in ('cd', ':'): continue
        try: tool = actual(argv[0], native)
        except (FileNotFoundError, FactsError): continue
        roles = [role for role,p in binaries.items() if actual(p) == tool]
        if not roles or '-c' in argv: continue
        role = roles[0]; outputs = []
        if role == 'compiler' and '-o' in argv: outputs = [actual(argv[argv.index('-o')+1],native)]
        elif role == 'archiver':
            # GNU ar recipe is actual `ar <flags> archive object...`.
            candidates=[actual(v,native) for v in argv[1:] if v.endswith('.a')]
            outputs=[single(candidates,'ambiguous actual archive output')]
        elif role == 'ranlib': outputs=[actual(argv[-1],native)]
        elif role == 'stripper': outputs=[actual(argv[-1],native)]
        if outputs:
            require(all(str(p) in executed for p in outputs),'link recipe output was not executed')
            if role == 'archiver':
                index=next(i for i,v in enumerate(argv) if v.endswith('.a'))
                direct=[actual(v,native) for v in argv[index+1:]]
                require(direct and all(p.suffix in ('.o','.obj') for p in direct),'unsupported actual archive input syntax')
            elif role in ('ranlib','stripper'):direct=[outputs[0]]
            else:direct=[]  # Filled exclusively from the actual ld read closure.
            links.append(dict(target=outputs[0].relative_to(native).as_posix(),tool=role,argv=argv,outputs=list(map(str,outputs)),inputs=list(map(str,direct))))
    require({'archiver','compiler'} <= {v['tool'] for v in links}, 'missing real archive/module link recipes')
    module_links=[row for row in links if row['tool']=='compiler']
    require(len(module_links)==1,'a dedicated actual dependency file is required per module link')
    lto=any(a.startswith('-flto') for row in units+links for a in row['argv'])
    if lto:
        for role,name,option in [('lto_backend','lto1','-print-prog-name='),('lto_wrapper','lto-wrapper','-print-prog-name='),('lto_plugin','liblto_plugin.so','-print-file-name=')]:
            resolved=capture.query('resolve-'+role.replace('_','-'),[compiler,option+name],native)
            selected=Path(resolved) if Path(resolved).is_absolute() else Path(shutil.which(resolved) or '')
            require(selected.is_absolute(),'unresolved actual LTO tool: '+name);actual(selected);binaries[role]=str(selected)
    linkdep = native/'core-link.d'; capture.files.add(linkdep)
    inputs = {actual(p,native) for p in make_dependencies(text(linkdep))}
    for row in links:
        if row['tool']=='compiler':
            # Retained LTO temporaries are attributable only to this actual
            # save-temps link recipe and its output-derived naming. No arbitrary
            # input under BUILD is classified as a generated producer output.
            output=Path(row['outputs'][0]);bases=(output.name,output.stem)
            retained=[p for p in inputs if p.parent==output.parent and any(re.fullmatch(re.escape(base)+r'\.ltrans\d+\.ltrans\.o',p.name) for base in bases)]
            if retained:
                require(lto and '-save-temps=obj' in row['argv'],'unattributed retained LTO inputs')
                row['outputs'].extend(map(str,sorted(retained,key=str)))
            row['inputs']=list(map(str,sorted(inputs,key=str)))
        inputs.update(Path(p) for p in row['inputs'])
    # All archive member objects are real link inputs even if ld saw the archive.
    inputs.update(Path(p) for v in units for p in v['outputs'])
    target = capture.query('compiler-target',[compiler,'-dumpmachine'],native)
    crt = sorted(p.name for p in inputs if re.match(r'.*crt[^/]*\.(?:o|a|so(?:\.\d+)*)$',p.name))
    require(crt, 'actual linker CRT inputs missing')
    return units,links,inputs,used,binaries,dict(generator_version=capture.query('generator-version',[ninja,'--version'],native),target_arch=target,toolset=info['CMAKE_CXX_COMPILER_VERSION'],crt=';'.join(crt))


def windows_model(native, config):
    import importlib.util
    spec=importlib.util.spec_from_file_location('build_facts_qualification_metadata',Path(__file__).with_name('public_qualification.py'))
    q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
    return q.read_windows_toolchain(native,config)


def msvc_project_sources(project, native, config):
    tree=ET.fromstring(text(project));tag=lambda n:n.tag.rsplit('}',1)[-1]
    def enabled(node):
        condition=node.get('Condition')
        return condition is None or condition.replace(' ','').split('==')[-1].strip("'\"")==config+'|x64'
    intdirs=[n.text for n in tree.iter() if tag(n)=='IntDir' and enabled(n) and n.text]
    intdir=single(intdirs,'missing/ambiguous actual MSVC intermediate directory')
    defaults=[]
    for group in tree:
        if tag(group)=='ItemDefinitionGroup' and enabled(group):
            defaults.extend(n.text for n in group.iter() if tag(n)=='ObjectFileName' and enabled(n) and n.text)
    default=single(defaults,'missing/ambiguous actual MSVC object rule')
    result={}
    for node in tree.iter():
        if tag(node)!='ClCompile' or 'Include' not in node.attrib:continue
        source=actual(node.get('Include'),native);require(source not in result,'duplicate actual project source')
        overrides=[n.text for n in node if tag(n)=='ObjectFileName' and enabled(n) and n.text]
        value=single(overrides,'ambiguous per-source object rule') if overrides else default
        for key,replacement in [('$(IntDir)',intdir),('$(Configuration)',config),('$(Platform)','x64'),('%(Filename)',source.stem),('%(Extension)',source.suffix)]:value=value.replace(key,replacement)
        require('$(' not in value and '%(' not in value,'unsupported MSVC output macro')
        if value.endswith(('\\','/')):value+=source.stem+'.obj'
        result[source]=actual(value,native)
    require(result and len(set(result.values()))==len(result),'missing/colliding actual MSVC object outputs')
    return result


def msvc(native, config, cache, info, capture):
    # CMake/MSBuild toolchain metadata plus actual CL/link/Lib tracking records.
    model=windows_model(native,config)
    compiler=actual(model['compiler']);folder=compiler.parent
    binaries={role:str(actual(folder/name)) for role,name in [('compiler','cl.exe'),('frontend','c1xx.dll'),('backend','c2.dll'),('linker','link.exe'),('archiver','lib.exe')]}
    units=[];links=[];used=set();inputs=set();crt_flags=set()
    for target in ('skimagexx','_core'):
        project=native/(target+'.vcxproj');capture.files.add(project)
        expected=msvc_project_sources(project,native,config)
        tlog=native/(target+'.dir')/config/(target+'.tlog')
        def tracking(name):
            path=tlog/name;capture.files.add(path);return tlog_blocks(path)
        commands=tracking('CL.command.1.tlog');reads=tracking('CL.read.1.tlog');writes=tracking('CL.write.1.tlog')
        require({actual(s) for key in commands for s in key}==set(expected),'stale/missing CL source producers')
        require({actual(s) for key in reads for s in key}==set(expected),'stale/missing CL dependency producers')
        seen=set()
        for key,lines in commands.items():
            argv=expand_responses(windows_argv(' '.join(lines)),native,capture.files,windows=True)
            # The CL tracking provider records arguments, not an executable
            # token. Its selected executable is pinned by CMake's MSVC model.
            require(argv and argv[0].lower()=='/c','unsupported CL tracking command')
            require(not any(v.lower().startswith(('/b1','/b2')) for v in argv),'custom MSVC frontend/backend needs explicit trace')
            for source_name in key:
                source=actual(source_name)
                require(source not in seen,'ambiguous repeated CL producer');seen.add(source)
                matches=[values for sources,values in reads.items() if source_name in sources]
                headers={actual(v) for values in matches for v in values}-{source};require(headers,'missing CL used dependencies')
                # CL.write is a sparse incremental log, not a TU inventory.
                # Derive the real object from the executed /Fo argument AND
                # independently require the exact generated project mapping.
                fo=[v[3:] for v in argv if v.lower().startswith('/fo')]
                value=single(fo,'missing/ambiguous executed MSVC /Fo')
                if value.endswith(('\\','/')):value+=source.stem+'.obj'
                obj=actual(value,native);require(obj==expected[source],'executed/project object output mismatch')
                require(not (headers & (set(expected)-{source})),'batched source reads cannot be attributed as per-TU headers')
                require(any(v.upper() in ('/MD','/MDD','/MT','/MTD') for v in argv),'actual MSVC CRT flag absent')
                crt_flags.update(v.upper() for v in argv if v.upper() in ('/MD','/MDD','/MT','/MTD'))
                used.update(headers);units.append(dict(source=str(source),target=target+':'+source.name,argv=[str(compiler),*argv],dependencies=list(map(str,sorted(headers,key=str))),outputs=[str(obj)]))
        recorded_objects={actual(v) for values in writes.values() for v in values if v.lower().endswith('.obj')}
        require(recorded_objects<=set(expected.values()),'stale/unmapped CL object writes')
        family='Lib' if target=='skimagexx' else 'link';role='archiver' if target=='skimagexx' else 'linker'
        commands=tracking(family+'.command.1.tlog')
        tracked_family='Lib-link' if family=='Lib' else family
        reads=tracking(tracked_family+'.read.1.tlog');writes=tracking(tracked_family+'.write.1.tlog')
        for key,lines in commands.items():
            argv=expand_responses(windows_argv(' '.join(lines)),native,capture.files,windows=True)
            require(argv and any(v.upper().startswith('/OUT:') for v in argv),'actual link output flag absent')
            outputs={actual(v) for values in writes.values() for v in values}
            require(outputs,'missing real link writes')
            direct={actual(v) for values in reads.values() for v in values}
            links.append(dict(target=target,tool=role,argv=[binaries[role],*argv],outputs=list(map(str,sorted(outputs,key=str))),inputs=list(map(str,sorted(direct,key=str)))))
        inputs.update(actual(v) for values in reads.values() for v in values)
    require(len(crt_flags)==1,'mixed/ambiguous actual CRT flags')
    require(any(p.name.upper() in ('MSVCRT.LIB','MSVCRTD.LIB','LIBCMT.LIB','LIBCMTD.LIB') for p in inputs),'actual CRT library reads absent')
    capture.files.update((native/'_core.vcxproj',native/'skimagexx.vcxproj'))
    buildlog=native.parent/'build-whole-product.log';capture.files.add(buildlog)
    versions=set(re.findall(r'MSBuild version ([\w.+-]+)',text(buildlog)))
    require(len(versions)==1,'actual MSBuild version absent/ambiguous in build log')
    extra=dict(generator_version=versions.pop(),target_arch=model['architecture'],toolset=model['toolset_version'],crt=next(iter(crt_flags)))
    return units,links,inputs,used,binaries,extra


def collect(source_root, build_root, revision, profile, output):
    require(re.fullmatch(r'[0-9a-f]{40}', revision) is not None,'exact source revision required')
    require(profile in ('windows','modern','minimum','debug'),'unknown qualification profile')
    source=Path(source_root).resolve(strict=True);build=Path(build_root).resolve(strict=True);native=build/'native-build'
    require(native.is_dir() and native.resolve().is_relative_to(build),'actual owned native build required')
    output=Path(output).absolute();require(not output.exists(),'never overwrite earlier facts')
    require(output.parent.resolve()==build,'facts must be sealed inside qualification output')
    capture=Capture(build/'build-identity-facts-evidence')
    cache_path=native/'CMakeCache.txt';compiler_path=single((native/'CMakeFiles').glob('*/CMakeCXXCompiler.cmake'),'missing/ambiguous CMake compiler evidence')
    capture.files.update((cache_path,compiler_path));cache=cmake_cache(cache_path);info=cmake_sets(compiler_path)
    require(Path(cache['CMAKE_HOME_DIRECTORY']).resolve(strict=True)==source,'actual build source root differs')
    require(info.get('CMAKE_CXX_COMPILER_ID') in ('GNU','MSVC'),'unsupported actual compiler')
    require((profile=='windows')==(info['CMAKE_CXX_COMPILER_ID']=='MSVC'),'actual compiler/profile differs')
    config='Debug' if profile=='debug' else 'Release'
    if info['CMAKE_CXX_COMPILER_ID']=='GNU':
        require(cache.get('CMAKE_BUILD_TYPE')==config,'actual configuration differs')
        units,links,inputs,used,binaries,extra=gnu(native,cache,info,capture)
    else:units,links,inputs,used,binaries,extra=msvc(native,config,cache,info,capture)
    require(all(Path(row['source']).is_relative_to(source) for row in units),'actual translation unit outside qualified source root')
    for key in ('SKIMAGEXX_OPENMP','SKIMAGEXX_AVX2'):require(cache.get(key) in ('ON','OFF'),'missing explicit actual feature configuration')
    roots=header_roots(used,source,native,info,windows=info['CMAKE_CXX_COMPILER_ID']=='MSVC')
    outputs={Path(p) for v in units+links for p in v['outputs']}
    require(all(p.is_relative_to(build) for p in outputs),'generated output outside owned build')
    link_inputs=[dict(path=str(p),generated=p in outputs) for p in sorted(inputs,key=str)]
    # Discarded LTO temporaries are never invented or silently omitted: actual()
    # above rejects missing files, keeping the attributable trial ineligible.
    wheel=single((build/'product-wheels').glob('*.whl'),'missing/ambiguous real product wheel')
    cores={Path(p) for row in links for p in row['outputs'] if Path(p).name.startswith('_core.') and Path(p).suffix in ('.so','.pyd')}
    core=single(cores,'missing/ambiguous actual linked production core')
    argv=[a for row in units+links for a in row['argv']]
    configuration=dict(compiler_id=info['CMAKE_CXX_COMPILER_ID'],compiler_version=info['CMAKE_CXX_COMPILER_VERSION'],generator=cache['CMAKE_GENERATOR'],**extra,build_type=config,openmp=cache['SKIMAGEXX_OPENMP']=='ON',avx2=cache['SKIMAGEXX_AVX2']=='ON',lto=any(a.startswith('-flto') or a.upper()=='/GL' or a.upper().startswith('/LTCG') for a in argv))
    groups=dict(configuration=[str(cache_path)],compiler=[str(compiler_path)]+[str(p) for p in capture.files if p.name.startswith(('resolve-','compiler-target','generator-version'))],commands=[str(p) for p in capture.files if 'command' in p.name.lower() or p.name in ('build.ninja','compile_commands.json','.ninja_log')],dependencies=[str(p) for p in capture.files if p.name in ('.ninja_deps','core-link.d') or 'deps' in p.name or '.read.' in p.name.lower()],responses=[str(p) for p in capture.files if p.suffix.lower()=='.rsp'])
    groups={k:sorted(set(v)) for k,v in groups.items() if v}
    # Include every raw provider input, including write/response/project logs.
    groups['raw_metadata']=sorted(map(str,capture.files))
    require(all(groups.get(k) for k in ('configuration','compiler','commands','dependencies')),'incomplete actual evidence groups')
    for path in capture.files:actual(path)
    facts=dict(schema=SCHEMA,revision=revision,profile=profile,source_root=str(source),build_root=str(build),configuration=configuration,binaries=binaries,header_roots=roots,translation_units=units,link_commands=links,link_inputs=link_inputs,evidence=groups,product_wheel=str(actual(wheel)),core=str(actual(core)))
    output.write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    return facts


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','build-root','revision','profile','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    facts=collect(args.source_root,args.build_root,args.revision,args.profile,args.output)
    print(json.dumps(dict(schema=facts['schema'],translation_units=len(facts['translation_units']),link_commands=len(facts['link_commands']),output=args.output)))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
