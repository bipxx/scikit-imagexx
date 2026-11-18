import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path.cwd()
SOURCE = ROOT / 'source'
PROOF = ROOT / 'proof'
PROOF.mkdir(exist_ok=False)
BASELINE = 'eb1f37e441b7d1262cb43f2205f39d3fec6c598c'
NATIVE = 'src/simd_color_avx2.cpp'
PREVIOUS = '1e9c8396f7bbf7c5ad7643ecfd23f3c066fa5724220cc640b64806f159d1c09f'
CANDIDATE = 'b98b89d70b057a43549f5e4fa075fb38c58cf91780d26df52f7e6d7a90e5153d'
ORACLE = 'b03ceda43574e3aeee174decd69eb3109f4bd53295cfd2e7854d310024811f9f'
ROWS = 1539
EXPECTED_BYTES = 280221993
CXX = '/usr/bin/g++'
COMMON = ['-std=c++17', '-O3', '-fno-fast-math', '-ffp-contract=off',
          '-ffunction-sections', '-fdata-sections', '-fopenmp', '-DSKIMAGEXX_AVX2',
          '-Iinclude', '-Isrc']

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def seal(name, data):
    with (PROOF / (name + '.json')).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2)
        stream.write(chr(10))

def git(*args):
    return subprocess.check_output(['git', '-C', str(SOURCE), *args])

def phase(label, args, *, env=None, output=None):
    with (PROOF / (label + '.stderr')).open('xb') as stderr, (PROOF / (label + '.stdout') if output is None else output).open('xb') as stdout:
        result = subprocess.run([str(a) for a in args], cwd=SOURCE, env=env, stdout=stdout, stderr=stderr)
    seal(label, dict(argv=list(map(str, args)), exit=result.returncode))
    return result.returncode

def compare(left, right, label):
    sums = [hashlib.sha256(), hashlib.sha256()]
    consumed = 0
    difference = None
    with left.open('rb') as before, right.open('rb') as after:
        while a := before.read(1024**2):
            b = after.read(len(a))
            if difference is None and a != b:
                n = next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), len(b))
                difference = dict(offset=consumed+n, prior=a[n:n+64].hex(), candidate=b[n:n+64].hex())
            for h, x in zip(sums, (a, b)):
                h.update(x)
            consumed += len(a)
        extra = after.read(64)
    lengths = [left.stat().st_size, right.stat().st_size]
    matches = not extra and lengths == [EXPECTED_BYTES, EXPECTED_BYTES] and difference is None and sums[0].digest() == sums[1].digest()
    result = dict(cases=ROWS, bytes=consumed, observed_lengths=lengths, raw_sha256=[h.hexdigest() for h in sums], matches=matches,
                  first_difference=difference)
    seal(label, result)
    assert matches
    return result

source = os.environ['PRODUCT_SOURCE']
assert git('rev-parse','HEAD').decode().strip() == source
assert git('rev-parse','HEAD^').decode().strip() == BASELINE
assert git('diff','--name-only','HEAD^','HEAD').decode().splitlines() == [NATIVE]
assert git('status','--porcelain').decode() == ''
previous = git('show', BASELINE + ':' + NATIVE)
assert sha(previous) == PREVIOUS and sha((SOURCE / NATIVE).read_bytes()) == CANDIDATE
assert sha((ROOT / 'diagnostic-cie-wide-public-oracle.cpp').read_bytes()) == ORACLE
assert all(shutil.disk_usage(p).free >= 30 * 1024**3 for p in (SOURCE, tempfile.gettempdir()))
assert not any(key in os.environ for key in ('SKIMAGEXX_DISABLE_SIMD','NPY_DISABLE_CPU_FEATURES','NPY_ENABLE_CPU_FEATURES'))
compiler = subprocess.check_output([CXX,'-dumpfullversion'], text=True).strip()
assert int(compiler.split('.')[0]) >= 13
seal('source', dict(source=source,parent=BASELINE,old_leaf=PREVIOUS,new_leaf=CANDIDATE,compiler=compiler,
                    git_version=git('--version').decode().strip(), arch=os.uname().machine))

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    tuner = p / 'cpu-preflight.cpp'
    tuner.write_text('''#include <cstdio>
#include <immintrin.h>
int main() {
    const bool v4 = __builtin_cpu_supports("x86-64-v4");
    std::printf("x86-64-v4=%d initial_mxcsr=0x%x\\n", v4, _mm_getcsr());
    return v4 && (_mm_getcsr() & _MM_MASK_MASK) == _MM_MASK_MASK ? 0 : 2;
}
''', encoding='utf-8')
    assert phase('host-compile', [CXX, '-std=c++17','-O2',tuner,'-o',p/'cpu']) == 0
    if phase('host-preflight', [p/'cpu']) != 0:
        seal('terminal', dict(source=source,status='no_eligible_host',cases_per_mode=0,timing_batches=0))
        raise SystemExit(2)
    previous_src = p/'previous.cpp'
    previous_src.write_bytes(previous)
    objs=[]
    for name in ('color','perceptual','convert','image'):
        out=p/(name+'.o')
        assert phase('compile-'+name,[CXX,*COMMON,'-c','src/'+name+'.cpp','-o',out])==0
        objs.append(out)
    oracle = p/'oracle.o'
    assert phase('compile-oracle',[CXX,*COMMON,'-c',ROOT/'diagnostic-cie-wide-public-oracle.cpp','-o',oracle])==0
    for name,path in [('previous',previous_src),('candidate',NATIVE)]:
        obj=p/(name+'.o')
        assert phase('compile-'+name,[CXX,*COMMON,'-mavx2','-c',path,'-o',obj])==0
        assert phase('link-'+name,[CXX,'-fopenmp','-Wl,--gc-sections',oracle,*objs,obj,'-o',p/name])==0
    assert phase('wide-assembly',['objdump','-drwC',p/'candidate.o'])==0
    pinned = ROOT/'diagnostic-cie-wide-local-oracle.cpp'
    assert sha(pinned.read_bytes()) == 'dd05c8b12136f9a6f1d51380f9baae05ff776b89706f5575c52706293d0ededf'
    assert phase('local-oracle-compile',[CXX,*COMMON,'-mavx2',pinned,*objs,'-Wl,--gc-sections','-o',p/'local'])==0
    assert phase('local-oracle',[p/'local'])==0
    assert (PROOF/'local-oracle.stdout').read_text(encoding='utf-8').strip() == 'f32 v4 direct power raw, neutral FE_INEXACT, both transforms directed and invalid-unmasked: pass'
    results={}
    for mode in ('normal','disabled'):
        env=dict(os.environ)
        if mode == 'disabled':
            env['SKIMAGEXX_DISABLE_SIMD']='1'
        for name in ('previous','candidate'):
            assert phase(mode+'-'+name,[p/name],env=env,output=p/(name+'.raw'))==0
        results[mode]=compare(p/'previous.raw',p/'candidate.raw',mode+'-comparison')
        for name in ('previous','candidate'):
            (p/(name+'.raw')).unlink()
    seal('terminal',dict(source=source,status='passed',timing_batches=0,comparisons=results,
                         local_oracle_status=0,case_bytes=EXPECTED_BYTES))
