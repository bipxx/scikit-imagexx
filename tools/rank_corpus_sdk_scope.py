"""SDK scope of authenticated rank capsules; historical68 remains historical."""
import json
import re
from pathlib import Path
from tools.public_qualification import (CURRENT_PRODUCT_SCOPE, CURRENT_SCOPE_CONTRACT, PRODUCT_SCOPE,
    PROFILES, SDK_MEMBERS, current_contract, current_install_layout)

MARKER = 'qualification-sdk-scope.json'


def validate_sdk_scope(root, capsule, runtime):
    """Called after runtime/source authentication, before corpus fixtures."""
    source = runtime['source']; sdk = runtime['runtime']['sdk_sha256']
    current = CURRENT_SCOPE_CONTRACT in source['files_sha256']
    path = Path(capsule) / MARKER
    if not path.exists():
        assert not current, 'current corpus capsule requires an explicit SDK scope marker'
        assert len(sdk) == SDK_MEMBERS, 'historical corpus requires the original68-member SDK'
        return PRODUCT_SCOPE
    marker = json.loads(path.read_text(encoding='utf-8'))
    assert (type(marker) is dict and set(marker) == {'schema', 'scope', 'profile', 'source_revision'}
            and marker['schema'] == 'rank-corpus-sdk-scope-v1'
            and marker['scope'] == CURRENT_PRODUCT_SCOPE
            and type(marker['profile']) is str and marker['profile'] in PROFILES
            and type(marker['source_revision']) is str
            and re.fullmatch('[0-9a-f]{40}', marker['source_revision'])), 'invalid current SDK scope marker'
    assert current and marker['source_revision'] == source['revision'], 'SDK marker/source scope differs'
    root = Path(root)
    read = lambda name: (root / name).read_bytes()
    current_contract(read)
    layout = current_install_layout(read, source['files_sha256'], marker['profile'])
    assert set(sdk) == set(layout['SDK_members']), 'current SDK file closure differs from source install rules'
    return marker['scope']
