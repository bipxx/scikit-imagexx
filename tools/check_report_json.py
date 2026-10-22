"""Check report JSON syntax without building native code or claiming parity."""
import json
from pathlib import Path


def main():
    files = sorted((Path(__file__).resolve().parents[1] / 'benchmarks').glob('*.json'))
    if not files:
        raise SystemExit('No benchmark reports found')
    errors = []
    for path in files:
        try:
            json.loads(path.read_bytes())
        except (OSError, ValueError) as error:
            errors.append(f'{path.name}: {error}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Parsed {len(files)} benchmark JSON reports. This is a syntax check only.')


if __name__ == '__main__':
    main()
