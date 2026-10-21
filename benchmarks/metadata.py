"""Every covered API with custom NumPy dtype metadata, versus scikit-image."""
import sys
from coverage import main

if __name__=='__main__':
    # Later explicit CLI options override these fixed report defaults.
    sys.argv[1:1]=['--dtype-metadata','--sizes','512','--output','benchmarks/metadata-parameters.json']
    main()
