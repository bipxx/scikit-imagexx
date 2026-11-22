from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import importlib.abc
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time

SOURCE = '0b33a90c20c8e26912e4fb9e0fd427529f327c7a'
TREE = '44937060407ece134e268e5aa46fb675029d9d84'
MANIFEST_PIN = '5052cbfeff2f5b0096ab364b5221d221536128501ce175c2ac91296853ea25e6'
WORKFLOW = '.github/workflows/public-qualification.yml'
VERSIONS = {'windows311': '3.11.11', 'windows313': '3.13.14'}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')


def workflow_arguments(workflow_raw, selector_count):
    lines = [line.strip() for line in workflow_raw.decode('utf-8').splitlines()
             if line.strip().startswith('python -m pytest -q --capture=sys ')]
    require(len(lines) == 1, 'Expected one full workflow source command')
    command = lines[0].split(' 2>&1 | tee ')[0]
    argv = shlex.split(command)[3:]
    selectors = [value for value in argv if value.startswith('tests/')]
    require(len(selectors) == selector_count and argv[:2] == ['-q', '--capture=sys'], 'Full selector command changed')
    forbidden = ('-x', '--exitfirst', '--noconftest', '--confcutdir', '-c', '--collect-only', '--co', '--basetemp')
    require(not any(value.split('=', 1)[0] in forbidden or value.startswith('--maxfail') for value in argv),
            'Unexpected pytest override')
    return command, argv, selectors


def report_arguments(workflow_argv, out):
    args = list(workflow_argv)
    substitutions = []
    for idx, value in enumerate(args):
        if value.startswith('cache_dir='):
            args[idx] = 'cache_dir=' + str(out / 'cache')
        elif value.startswith('--junitxml='):
            args[idx] = '--junitxml=' + str(out / 'results.xml')
        else:
            continue
        substitutions.append({'index': idx, 'old': value, 'new': args[idx]})
    require(len(substitutions) == 2 and {x['old'].split('=', 1)[0] for x in substitutions} ==
            {'cache_dir', '--junitxml'}, 'Only report paths may change')
    return args, substitutions


def main():
    parser = ArgumentParser()
    parser.add_argument('--profile', choices=VERSIONS, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    cli = parser.parse_args()
    require(sys.platform == 'win32' and platform.python_version() == VERSIONS[cli.profile], 'Wrong Windows runtime')
    require(not sys.flags.optimize and not any(os.environ.get(k) for k in ('PYTEST_ADDOPTS', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD')),
            'Normal pytest startup environment required')
    source, out, fixture = cli.source.resolve(strict=True), cli.out.resolve(), cli.fixture.resolve()
    require(not fixture.is_relative_to(source) and not out.is_relative_to(source), 'Reports and fixtures must be external to source')
    require(not fixture.is_relative_to(out) and not out.is_relative_to(fixture), 'Fixtures and evidence must be separate')
    out.mkdir(exist_ok=False, parents=True)
    free = shutil.disk_usage(fixture.parent).free
    required = 44 * 1024**3
    write(out / 'capacity.json', {'observed_free_bytes': free, 'required_bytes': required, 'scope': 'one Windows profile',
                                  'pass': free >= required})
    require(free >= required, 'Hosted fixture destination has insufficient capacity')
    native_git = shutil.which('git.exe')
    require(native_git and Path(native_git).suffix.lower() == '.exe', 'Windows native Git is required')
    def git(*args):
        return subprocess.check_output([native_git, '-C', str(source), *args])

    manifest_path = Path(__file__).with_name('source-manifest-306-v1.json')
    require(digest(manifest_path) == MANIFEST_PIN, 'Frozen source manifest changed')
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest['source'] == SOURCE and manifest['tree'] == TREE, 'Wrong manifest provenance')
    require(git('rev-parse', 'HEAD', 'HEAD^{tree}').decode().splitlines() == [SOURCE, TREE] and not git('status', '--porcelain'),
            'Candidate commit or pristine status differs')
    index = git('ls-files', '-sz')
    (out / 'source-index-entry.bin').write_bytes(index)
    for path, pin in manifest['current_scope'].items():
        require(digest(source / path) == pin, f'Source bytes differ: {path}')
    workflow = (source / WORKFLOW).read_bytes()
    require(sha256(workflow).hexdigest() == manifest['workflow_sha256'], 'Raw qualification workflow changed')
    command, argv, selectors = workflow_arguments(workflow, 186)
    require(selectors == manifest['selectors'], 'Selector order differs')
    actual, substitutions = report_arguments(argv, out)
    fixture.mkdir(exist_ok=False)
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.dont_write_bytecode = True
    os.environ.update(PYTEST_DEBUG_TEMPROOT=str(fixture), TMP=str(fixture), TEMP=str(fixture), PYTHONDONTWRITEBYTECODE='1')
    attempted = []

    class ObserveImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'numpy', 'scipy', 'skimage', 'skimagexx'}:
                attempted.append(fullname)

    sys.meta_path.insert(0, ObserveImports())
    import pytest

    require(pytest.__version__ == '9.1.1', 'Wrong pytest version')

    class Inventory:
        def pytest_collection_finish(self, session):
            write(out / 'testIDs.json', [item.nodeid for item in session.items])

        def pytest_sessionfinish(self, session, exitstatus):
            write(out / 'plugins.json', [(name, getattr(plugin, '__file__', None))
                                        for name, plugin in session.config.pluginmanager.list_name_plugin()])

    write(out / 'command.json', {'source': SOURCE, 'source_tree': TREE, 'python': sys.version, 'executable': sys.executable,
                                 'native_git': native_git, 'native_git_version': git('--version').decode().strip(),
                                 'workflow_sha256': manifest['workflow_sha256'], 'actual_workflow_command': command,
                                 'workflow_argv': argv, 'argv': actual, 'selectors': selectors, 'selector_count': len(selectors),
                                 'report_path_substitutions_only': substitutions, 'normal_conftest': True,
                                 'plugin_autoload': True, 'no_failfast': True, 'fixture_root': str(fixture),
                                 'manifest_sha256': MANIFEST_PIN, 'profile': cli.profile})
    start = time.monotonic()
    code = int(pytest.main(actual, plugins=[Inventory()]))
    post = git('rev-parse', 'HEAD').decode().strip()
    clean = not git('status', '--porcelain')
    pins = all(digest(source / path) == pin for path, pin in manifest['current_scope'].items())
    write(out / 'terminal.json', {'exit_code': code, 'seconds': time.monotonic() - start, 'source_HEAD_after': post,
                                  'source_clean_after': clean, 'source_index_unchanged': git('ls-files', '-sz') == index,
                                  'raw_pins_unchanged': pins, 'scientific_import_attempts': attempted,
                                  'workflow_sha256': manifest['workflow_sha256']})
    require(post == SOURCE and clean and pins and git('ls-files', '-sz') == index, 'Source or index changed during tests')
    require(not attempted, 'Scientific imports attempted during source checks')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
