"""Planning regeneration preserves reviewed scope and detects stale artifacts."""
import hashlib
import json

from tools import generate_api_implementation_paths as generator


def test_regeneration_preserves_reviewed_igg_guidance_and_current_inventory():
    document = generator.make_document()
    route = next(row for row in document['entries']
                 if row['api'] == 'segmentation.inverse_gaussian_gradient')
    reviewed = {name: route[name] for name in (
        'easiest_credible_path', 'next_concrete_action', 'compatibility_hazards')}
    reviewed['one_worker_recipe'] = route['benchmark_plan']['one_worker_recipe']
    raw = json.dumps(reviewed, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False).encode('utf-8')
    # Updated for current408 where segmentation.inverse_gaussian_gradient moved
    # from upstream-only to native-partial in coverage/api.json.
    assert hashlib.sha256(raw).hexdigest() == (
        'dd7e443ad5ac919e8e37cf66725af045e7a4534161f655d2f3330471d2ad69b2')

    inventory = json.loads(generator.INVENTORY.read_text(encoding='utf-8'))
    expected = {row['api']: row for row in inventory['entries']}
    for row in document['entries']:
        assert row['current_implementation'] == expected[row['api']]['implementation']
    assert generator.validate(document, generator.render_markdown(document))[
        'api_entries'] == len(expected) == 477


def test_check_reports_stale_artifacts_without_rewriting(tmp_path, monkeypatch, capsys):
    document = generator.make_document()
    json_path = tmp_path / 'routes.json'
    markdown_path = tmp_path / 'routes.md'
    json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    markdown_path.write_text('stale route\n', encoding='utf-8')
    before = json_path.read_bytes(), markdown_path.read_bytes()
    monkeypatch.setattr(generator, 'ROOT', tmp_path)
    monkeypatch.setattr(generator, 'JSON_PATH', json_path)
    monkeypatch.setattr(generator, 'MD_PATH', markdown_path)
    monkeypatch.setattr(generator.sys, 'argv', ['generator', '--check'])

    assert generator.main() == 1

    assert 'Planning artifacts stale or missing: routes.md' in capsys.readouterr().err
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == before
