"""Fixed prospective IGG public one-thread cases; no execution authority."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',action='store_true');a=p.parse_args()
 d=json.loads((ROOT/'coverage/contracts/inverse-gaussian-gradient-v1.json').read_bytes())
 if a.plan:print(json.dumps(dict(cases=d['benchmark_cases'],method=d['benchmark_method'],unbound=d['unbound']),indent=2));return
 raise SystemExit('IGG_UNBOUND: root-admitted source/runtime/core/SDK/cohorts and separate measurement GO required')
if __name__=='__main__':main()
