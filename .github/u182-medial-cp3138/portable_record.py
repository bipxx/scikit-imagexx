"""Compare a genuine hosted wheel install to retained Medial Axis package bytes."""
import csv
from configparser import ConfigParser
from hashlib import sha256
from base64 import urlsafe_b64encode
import io
import json
from pathlib import Path, PurePosixPath
import sys
from zipfile import ZipFile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_digest(members):
    raw = json.dumps(members, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return sha256(raw).hexdigest()


def rows(raw):
    result = {}
    for member, checksum, size in csv.reader(io.StringIO(raw.decode("utf-8"), newline="")):
        require(member and member not in result, f"duplicate or empty RECORD row: {member}")
        result[member] = [checksum, size]
    return result


def file_pin(path):
    digest = sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(2**20), b""):
            digest.update(chunk)
            size += len(chunk)
    return ["sha256=" + urlsafe_b64encode(digest.digest()).decode().rstrip("="), str(size)]


def site_path(root, name):
    parts = PurePosixPath(name).parts
    require(parts and parts[0] != "/" and ".." not in parts and "\\" not in name
            and ":" not in name and PurePosixPath(name).as_posix() == name, f"unsafe member: {name}")
    path = root.joinpath(*parts)
    require(path.resolve(strict=True) == path and path.is_file(), f"not a regular in-place file: {name}")
    return path


def immutable_members(record, record_name, *, installed):
    directory = record_name.removesuffix("/RECORD")
    generated = {record_name, *(f"{directory}/{name}"
                                for name in ("INSTALLER", "REQUESTED", "direct_url.json"))}
    result = {}
    for name, pin in record.items():
        if installed and name in generated:
            continue
        if not installed and name == record_name:
            continue
        parts = PurePosixPath(name).parts
        if not installed and len(parts) >= 3 and parts[0].endswith(".data"):
            if parts[1] == "scripts":
                continue
            require(parts[1] in {"purelib", "platlib"}, f"unmapped wheel scheme: {name}")
            name = PurePosixPath(*parts[2:]).as_posix()
        require(name not in result and pin[0].startswith("sha256=") and len(pin[0]) == 50
                and pin[1].isdigit(), f"invalid immutable RECORD member: {name}")
        result[name] = pin
    return result


def script_names(raw):
    config = ConfigParser(interpolation=None)
    config.optionxform = str
    if raw is not None:
        config.read_string(raw.decode("utf-8"))
    names = sorted(name for section in ("console_scripts", "gui_scripts")
                   if config.has_section(section) for name in config[section])
    require(len(names) == len(set(names)) and all(name.isascii() and name not in {".", ".."}
            and name and all(ch.isalnum() or ch in "._-" for ch in name) for name in names),
            "unsafe or duplicate wheel entry point")
    return names


def check_target_wrappers(host_record, names, site, distribution):
    """Observe pip's posix_home RECORD aliases and hash their relocated wrappers."""
    prefix = "../../bin/"
    expected = {prefix + name for name in names}
    seen = {member for member in host_record if member.startswith(prefix)}
    require(seen == expected, f"generated target wrappers differ: {distribution}; "
            f"missing={sorted(expected - seen)[:8]}; unexpected={sorted(seen - expected)[:8]}")
    for name in names:
        wrapper = "bin/" + name
        require(host_record[prefix + name] == file_pin(site_path(site, wrapper)),
                f"generated target wrapper bytes differ: {distribution}/{name}")
    return {member: host_record[member] for member in sorted(seen)}


def record_delta(actual, expected):
    return {"missing": sorted(expected.keys() - actual.keys())[:8],
            "unexpected": sorted(actual.keys() - expected.keys())[:8],
            "changed": sorted(name for name in actual.keys() & expected.keys()
                              if actual[name] != expected[name])[:8]}


def check_install(manifest, wheelhouse, site):
    """Return actual hosted receipt and live-member pins. Called before scientific imports."""
    wheelhouse, site = Path(wheelhouse).resolve(strict=True), Path(site).resolve(strict=True)
    require(not any(site.rglob("*.pyc")), "bytecode present in hosted stock site")
    all_members, report = {}, {}
    for name, spec in manifest["stock_distributions"].items():
        wheel = site_path(wheelhouse, spec["wheel"])
        native_digest, size = file_pin(wheel)
        expected = urlsafe_b64encode(bytes.fromhex(spec["wheel_sha256"])).decode().rstrip("=")
        require(native_digest == "sha256=" + expected and int(size) == spec["wheel_bytes"],
                f"wheel identity changed: {name}")
        record_name = spec["record"]
        with ZipFile(wheel) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            require(len(entries) == len(names), f"duplicate wheel members: {name}")
            raw_record = archive.read(record_name)
            packed = rows(raw_record)
            require(set(packed) == {entry.filename for entry in entries if not entry.is_dir()}
                    and packed[record_name] == ["", ""],
                    f"wheel RECORD coverage changed: {name}")
            require(all(int(pin[1]) == archive.getinfo(member).file_size
                        for member, pin in packed.items() if member != record_name),
                    f"wheel member sizes changed: {name}")
            expected_members = immutable_members(packed, record_name, installed=False)
            entry_point_file = record_name.removesuffix("RECORD") + "entry_points.txt"
            names = script_names(archive.read(entry_point_file) if entry_point_file in packed else None)
            require(names == spec["generated_entry_points"], f"wheel entry points changed: {name}")
        require(len(expected_members) == spec["immutable_members"]
                and canonical_digest(expected_members) == spec["retained_immutable_digest"],
                f"wheel differs from retained immutable members: {name}")

        live_record = site_path(site, record_name).read_bytes()
        host_record = rows(live_record)
        wrappers = check_target_wrappers(host_record, names, site, name)
        installed = {member: pin for member, pin in host_record.items() if member not in wrappers}
        actual_members = immutable_members(installed, record_name, installed=True)
        require(actual_members == expected_members, f"host immutable RECORD differs: {name}; "
                f"{record_delta(actual_members, expected_members)}")
        for member, pin in actual_members.items():
            require(file_pin(site_path(site, member)) == pin, f"host file differs: {member}")
            require(member not in all_members, f"overlapping distribution members: {member}")
            all_members[member] = pin
        report[name] = {
            "wheel_sha256": spec["wheel_sha256"],
            "wheel_record_sha256": sha256(raw_record).hexdigest(),
            "retained_uv_record_sha256": spec["retained_uv_record_sha256"],
            "host_record_sha256": sha256(live_record).hexdigest(),
            "immutable_members": len(actual_members),
            "immutable_digest": spec["retained_immutable_digest"],
            "generated_wrappers": wrappers,
        }
    for member, sha in manifest["selected_retained_live_files_sha256"].items():
        checksum = "sha256=" + urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip("=")
        require(member in all_members and all_members[member][0] == checksum,
                f"selected retained byte pin changed: {member}")
    return report, all_members


def check_loaded(site, members, required, module_registry=None):
    """Check all loaded stock modules and the required runtime computation providers."""
    module_registry = sys.modules if module_registry is None else module_registry
    site = Path(site).resolve(strict=True)
    namespaces = {PurePosixPath(member).parts[0].removesuffix(".py") for member in members
                  if not PurePosixPath(member).parts[0].endswith((".dist-info", ".libs"))}
    witnessed = {}
    for name, module in tuple(module_registry.items()):
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        path = Path(origin).resolve(strict=True)
        stock_namespace = name.partition(".")[0] in namespaces
        if not path.is_relative_to(site):
            require(not stock_namespace, f"foreign loaded provider: {name}")
            continue
        member = path.relative_to(site).as_posix()
        require(member in members and file_pin(site_path(site, member)) == members[member],
                f"unverified loaded module: {name}")
        witnessed[member] = members[member][0]
    require(set(required) <= witnessed.keys(), "missing loaded Medial Axis provider")
    return dict(sorted(witnessed.items()))
