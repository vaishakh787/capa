# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
capa/features/extractors/vmray/noise.py

Sandbox noise filtering for VMRay traces.

VMRay records all processes observed during a sandbox run, including OS
background processes (explorer.exe, svchost.exe, etc.) unrelated to the
analysed sample.  This module identifies the *relevant* process subtree —
the submission process and all children it spawned — so VMRayExtractor can
optionally skip irrelevant processes during feature extraction.

Design
------
VMRay assigns every process a ``monitor_id`` and an ``origin_monitor_id``.
For the submission (entry-point) process, ``origin_monitor_id == monitor_id``
(the process originated from itself — no VMRay-tracked parent).

We use this invariant to find the root of the submission process tree,
then walk parent->child relationships to collect all descendants.
No hardcoded process-name lists are required.

Usage
-----
    from capa.features.extractors.vmray import VMRayAnalysis
    from capa.features.extractors.vmray.noise import get_relevant_monitor_ids

    analysis = VMRayAnalysis(zipfile_path)
    relevant = get_relevant_monitor_ids(analysis)   # frozenset[int]
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from capa.features.extractors.vmray import VMRayAnalysis

logger = logging.getLogger(__name__)


def _find_submission_monitor_id(analysis: "VMRayAnalysis") -> int | None:
    """
    Return the monitor_id of the single submission (root) process.

    A process is a submission root when origin_monitor_id == monitor_id.
    When multiple such processes exist (e.g. compound archives), we prefer
    the one whose filename matches analysis.submission_name.
    Falls back to the first candidate if no filename match is found.
    Returns None only when no monitor_processes exist at all.
    """
    if not analysis.monitor_processes:
        return None

    submission_name = (getattr(analysis, "submission_name", "") or "").lower()

    roots: list[int] = []
    for monitor_id, proc in analysis.monitor_processes.items():
        if proc.origin_monitor_id == proc.monitor_id:
            roots.append(monitor_id)
            logger.debug(
                "submission root candidate: monitor_id=%d  image_name=%s  pid=%d",
                monitor_id, proc.image_name, proc.pid,
            )

    if not roots:
        logger.warning(
            "no submission root found via origin_monitor_id heuristic; "
            "noise filtering disabled — all %d processes included",
            len(analysis.monitor_processes),
        )
        return None

    if len(roots) == 1:
        return roots[0]

    # Multiple roots — prefer filename match against submission_name.
    if submission_name:
        for r in roots:
            proc = analysis.monitor_processes.get(r)
            if proc and submission_name in (proc.filename or "").lower():
                logger.debug(
                    "selected submission root by filename match: monitor_id=%d", r
                )
                return r

    # No filename match — use first candidate and warn.
    logger.warning(
        "multiple root processes found (%s), using first: monitor_id=%d",
        roots, roots[0],
    )
    return roots[0]


def _build_children_index(analysis: "VMRayAnalysis") -> dict[int, list[int]]:
    """
    Build a parent_monitor_id -> [child_monitor_id, ...] mapping.

    Uses ppid (OS-level parent PID) bridged to VMRay monitor_ids via a
    pid->monitor_id reverse lookup.
    """
    pid_to_monitor_id: dict[int, int] = {}
    for monitor_id, proc in analysis.monitor_processes.items():
        pid_to_monitor_id[proc.pid] = monitor_id

    children: dict[int, list[int]] = defaultdict(list)

    for monitor_id, proc in analysis.monitor_processes.items():
        if proc.ppid == 0:
            continue  # root process — no parent to link

        parent_monitor_id = pid_to_monitor_id.get(proc.ppid)
        if parent_monitor_id is None:
            logger.debug(
                "process monitor_id=%d (pid=%d) has parent pid=%d "
                "with no matching monitor_id; treating as parentless",
                monitor_id, proc.pid, proc.ppid,
            )
            continue

        if parent_monitor_id == monitor_id:
            continue  # self-referential submission root

        children[parent_monitor_id].append(monitor_id)

    return dict(children)


def get_relevant_monitor_ids(analysis: "VMRayAnalysis") -> frozenset[int]:
    """
    Return the frozenset of monitor_ids belonging to the submission process
    and all processes it spawned (directly or transitively).

    Processes outside this set are sandbox infrastructure noise and can be
    skipped during VMRay feature extraction.

    Algorithm
    ---------
    1. Find the single submission root: origin_monitor_id == monitor_id,
       with filename-based disambiguation when multiple roots exist.
    2. Build parent->children index from ppid relationships.
    3. BFS from the submission root to collect the full subtree.

    If no root is found (empty or malformed archive), returns all monitor_ids
    so no data is silently lost.
    """
    if not analysis.monitor_processes:
        return frozenset()

    root = _find_submission_monitor_id(analysis)

    if root is None:
        # Safe fallback: include everything rather than silently lose data.
        logger.warning("could not identify submission root; retaining all processes")
        return frozenset(analysis.monitor_processes.keys())

    children_idx = _build_children_index(analysis)

    relevant: set[int] = set()
    queue: list[int] = [root]

    while queue:
        current = queue.pop()
        if current in relevant:
            continue
        relevant.add(current)

        proc = analysis.monitor_processes.get(current)
        if proc:
            logger.debug(
                "keeping process: monitor_id=%d  image_name=%s  pid=%d",
                current, proc.image_name, proc.pid,
            )

        for child_id in children_idx.get(current, []):
            queue.append(child_id)

    excluded = len(analysis.monitor_processes) - len(relevant)
    logger.info(
        "noise filter: %d/%d processes retained, %d excluded",
        len(relevant), len(analysis.monitor_processes), excluded,
    )

    return frozenset(relevant)


def log_noise_filter_summary(
    analysis: "VMRayAnalysis",
    relevant: frozenset[int],
) -> None:
    """
    Log which processes were kept and excluded.
    Call after get_relevant_monitor_ids() when verbose output is desired.
    """
    kept     = []
    excluded = []
    for monitor_id, proc in sorted(analysis.monitor_processes.items()):
        entry = f"  monitor_id={monitor_id:>4}  pid={proc.pid:<6}  {proc.image_name}"
        (kept if monitor_id in relevant else excluded).append(entry)

    logger.debug("=== VMRay Noise Filter Summary ===")
    logger.debug("Retained (%d):", len(kept))
    for line in kept:
        logger.debug(line)
    logger.debug("Excluded (%d):", len(excluded))
    for line in excluded:
        logger.debug(line)
    logger.debug("==================================")