"""One-worker public ifrt2 comparison; genuine binding required."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from finite_radon_execution_v1 import execute

def main():return execute('ifrt2')
if __name__=='__main__':main()
