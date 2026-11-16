"""Exact admitted VI component bytes through current shared fixture additions."""
import importlib.util
from pathlib import Path


def historical_source(root,name):
    root=Path(root);path=root/'tools/join_source_context.py'
    spec=importlib.util.spec_from_file_location('_vi_shared_source_projection',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.enabled_predecessor_bytes(root,name,(root/name).read_bytes())


def provider_predecessor_source(root,name,raw=None,canonical=False):
    """Exact32e source only, for immutable VI history; not current acceptance."""
    root=Path(root)
    if raw is None:raw=(root/name).read_bytes()
    if (root/'coverage/contracts/vi-provider-sdk-source-v1.json').is_file():
        from tools.mark_lbp_fp_context import VI_PROVIDER_SDK_CONTEXT_SHA256
        from tools.vi_provider_sdk_context import predecessor_bytes as sdk_predecessor
        raw=sdk_predecessor(root,name,raw,VI_PROVIDER_SDK_CONTEXT_SHA256,canonical)
    if not(root/'coverage/contracts/vi-public-log2-provider-source-v1.json').is_file():return raw
    from tools.mark_lbp_fp_context import VI_PUBLIC_LOG2_CONTEXT_SHA256
    from tools.vi_public_log2_context import predecessor_bytes
    return predecessor_bytes(root,name,raw,VI_PUBLIC_LOG2_CONTEXT_SHA256,canonical)
