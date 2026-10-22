"""Rebuild, verify parity, and benchmark every covered API against scikit-image."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,nargs='+',
                        help='Override worker counts (exposure stays serial); use 1 for serial-only verification')
    args=parser.parse_args()
    if args.threads is not None and min(args.threads)<1:
        parser.error('worker counts must be positive')
    if args.threads is not None and 1 not in args.threads:
        parser.error('include worker count 1 to verify single-thread performance')
    root=Path(__file__).resolve().parents[1]

    def run(*arguments):
        print('Running: '+subprocess.list2cmdline([sys.executable,*arguments]),flush=True)
        subprocess.run([sys.executable,*arguments],cwd=root,check=True)

    # Build before measuring so reports cannot accidentally time a stale wheel.
    # Install benchmarks/requirements.txt first to supply the build dependencies.
    run('-m','pip','install','.','--no-build-isolation')
    run('tools/inventory.py')
    for name in ('contrast','downscale','integrate','integral','gaussian','metadata','farid','edges','alpha','local_thresholds','thresholds','distance','stains','perceptual','color','morphology','coverage','conversions','exposure'):
        workers=[1] if name=='exposure' else args.threads
        options=[] if workers is None else ['--threads',*map(str,workers)]
        run(f'benchmarks/{name}.py',*options)
    # Reject missing APIs, mixed source snapshots, or absent serial comparisons.
    run('benchmarks/summarize.py')


if __name__=='__main__':
    main()
