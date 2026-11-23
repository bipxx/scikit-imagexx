"""Recognize competing activity at the quiet timing boundary."""
import re

COMPETING = re.compile(r"(?:^|[/\s])(?:g\+\+|gcc|cc1plus|cmake|ninja|pytest|ctest|curl|wget|git|gh|speed-stage\.py|quiet-benchmark\.py|pip)(?:[\s]|$)")

def check_processes(raw, own_pid, staged_pid):
    processes = {}
    for line in raw.splitlines():
        pid, parent, status, name, *detail = line.split(maxsplit=4)
        processes[int(pid)] = (int(parent), status, name, detail[0] if detail else "")
    ancestor = own_pid
    allowed = set()
    while ancestor in processes and ancestor not in allowed:
        allowed.add(ancestor)
        ancestor = processes[ancestor][0]
    active = []
    for pid, (_, status, name, arguments) in processes.items():
        if pid in allowed or status.startswith("Z") or name == "ps":
            continue
        if pid == staged_pid or COMPETING.search(arguments) or COMPETING.search(name) or " -m pytest" in arguments:
            active.append({"pid": pid, "name": name})
    return active
