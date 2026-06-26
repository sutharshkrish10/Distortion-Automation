"""
phase0_match.py  --  discover files and match them into parts by ID.

Filenames are inconsistent across the three source folders ("2mm Dense point
cloud.ply", "Mesh from 2PR.stl", "2PR.stl"), so we match purely by the part
size embedded in the name via a regex. New parts drop in with no code change.

Public API:
    match_parts() -> dict[part_id] -> {"nominal": Path|None, "ct": Path|None,
                                       "zephyr": Path|None}
"""

from __future__ import annotations

import re
from pathlib import Path

import config as C
from common import LOG

_SIZE_RE = re.compile(C.PART_SIZE_RE, re.IGNORECASE)


def _part_id_from_name(name: str) -> str | None:
    for size, pid in C.SIZE_TO_PARTID.items():
        if pid.lower() in name.lower():
            return pid
    m = _SIZE_RE.search(name)
    if m:
        return C.SIZE_TO_PARTID.get(m.group(1))
    return None


def _scan(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not folder.is_dir():
        LOG.warning("source folder missing: %s", folder)
        return found
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        pid = _part_id_from_name(p.name)
        if pid is None:
            LOG.warning("could not match part id for file: %s", p.name)
            continue
        if pid in found:
            LOG.warning("duplicate %s file for %s (keeping first): %s",
                        folder.name, pid, p.name)
            continue
        found[pid] = p
    return found


def match_parts() -> dict[str, dict[str, Path | None]]:
    nominal = _scan(C.SOURCE_DIRS["nominal"], (".stl",))
    ct      = _scan(C.SOURCE_DIRS["ct"],      (".stl",))
    zephyr  = _scan(C.SOURCE_DIRS["zephyr"],  (".stl",))

    part_ids = sorted(set(nominal) | set(ct) | set(zephyr))
    parts: dict[str, dict[str, Path | None]] = {}
    for pid in part_ids:
        parts[pid] = {
            "nominal": nominal.get(pid),
            "ct":      ct.get(pid),
            "zephyr":  zephyr.get(pid),
        }
    return parts


if __name__ == "__main__":
    for pid, srcs in match_parts().items():
        LOG.info("Part %s", pid)
        for s, p in srcs.items():
            LOG.info("    %-8s : %s", s, p.name if p else "** MISSING **")
