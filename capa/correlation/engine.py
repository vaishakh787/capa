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
capa/correlation/engine.py

Static-dynamic correlation engine (post-processing layer).

Correlates two capa ResultDocument objects — one from static analysis and one
from a VMRay sandbox run — and produces confidence-scored capability results.

Coverage model
--------------
coverage = |static_rule_names ∩ dynamic_rule_names| / |static_rule_names|

Set-based, so repeated dynamic matches for the same rule count once.

Integration example
-------------------
    from pathlib import Path
    from capa.render.result_document import ResultDocument
    from capa.correlation.engine import correlate_results, print_correlation_report

    static_doc  = ResultDocument.from_file(Path("static_result.json"))
    dynamic_doc = ResultDocument.from_file(Path("dynamic_result.json"))

    report = correlate_results(static_doc, dynamic_doc)
    print_correlation_report(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

# Lazy import: avoid pulling in the full capa chain at module load time.
# The concrete types are only needed inside correlate_results().
if TYPE_CHECKING:
    from capa.render.result_document import ResultDocument, RuleMatches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence tiers
# ---------------------------------------------------------------------------

class EvidenceTier(str, Enum):
    """Confidence classification for a correlated capability."""

    CONFIRMED_RUNTIME         = "CONFIRMED_RUNTIME"
    STATIC_ONLY_UNEXECUTED    = "STATIC_ONLY_UNEXECUTED"
    INCONCLUSIVE_LOW_COVERAGE = "INCONCLUSIVE_LOW_COVERAGE"
    DYNAMIC_ONLY              = "DYNAMIC_ONLY"


# Coverage below this value → INCONCLUSIVE rather than UNEXECUTED.
COVERAGE_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CorrelatedRule:
    rule_name:           str
    namespace:           Optional[str]
    tier:                EvidenceTier
    confidence:          float
    coverage:            float
    static_match_count:  int
    dynamic_match_count: int
    explanation:         str
    static_rule_meta:    Optional[object] = field(default=None, repr=False)
    dynamic_rule_meta:   Optional[object] = field(default=None, repr=False)


@dataclass
class CorrelationReport:
    results:             list[CorrelatedRule]
    coverage:            float
    total_static_rules:  int
    total_dynamic_rules: int
    confirmed_count:     int
    unexecuted_count:    int
    inconclusive_count:  int
    dynamic_only_count:  int


# ---------------------------------------------------------------------------
# Pure logic helpers (no capa imports — testable in isolation)
# ---------------------------------------------------------------------------

def _compute_rule_coverage(
    static_rule_names:  set[str],
    dynamic_rule_names: set[str],
) -> float:
    """
    coverage = |static ∩ dynamic| / |static|

    Returns 0.0 when static_rule_names is empty.
    """
    if not static_rule_names:
        return 0.0
    return len(static_rule_names & dynamic_rule_names) / len(static_rule_names)


def _assign_tier(
    in_static:  bool,
    in_dynamic: bool,
    coverage:   float,
) -> tuple[EvidenceTier, float, str]:
    """Return (tier, confidence, explanation) for a rule's correlation status."""

    if in_static and in_dynamic:
        return (
            EvidenceTier.CONFIRMED_RUNTIME, 0.9,
            "Matched in both static analysis and dynamic execution. "
            "Behaviour was observed at runtime.",
        )

    if not in_static and in_dynamic:
        return (
            EvidenceTier.DYNAMIC_ONLY, 0.7,
            "Matched only in the dynamic trace. "
            "Likely a dynamic-scope-only rule with no static equivalent.",
        )

    # Static only — use coverage to distinguish unexecuted from inconclusive.
    if coverage >= COVERAGE_THRESHOLD:
        return (
            EvidenceTier.STATIC_ONLY_UNEXECUTED, 0.4,
            f"Static match not observed at runtime. "
            f"Coverage {coverage:.0%} is sufficient — code path likely not "
            f"triggered during sandbox run, not a false positive.",
        )

    return (
        EvidenceTier.INCONCLUSIVE_LOW_COVERAGE, 0.6,
        f"Static match not observed at runtime. "
        f"Coverage is low ({coverage:.0%}) — the sandbox may have exited "
        f"early or hit an anti-sandbox check.",
    )


# ---------------------------------------------------------------------------
# Core correlation function
# ---------------------------------------------------------------------------

def correlate_results(
    static_doc:  "ResultDocument",
    dynamic_doc: "ResultDocument",
) -> CorrelationReport:
    """
    Correlate a static and a dynamic capa ResultDocument.

    Operates at the rule-name level (semantic correlation).
    No address-based matching — virtual addresses are ASLR-dependent.

    Args:
        static_doc:  ResultDocument from capa's static analysis pass.
        dynamic_doc: ResultDocument from capa's dynamic (VMRay) analysis pass.

    Returns:
        CorrelationReport with per-rule CorrelatedRule entries and statistics.
    """
    static_rules:  dict[str, "RuleMatches"] = dict(static_doc.rules)
    dynamic_rules: dict[str, "RuleMatches"] = dict(dynamic_doc.rules)

    static_names:  set[str] = set(static_rules)
    dynamic_names: set[str] = set(dynamic_rules)
    all_names:     set[str] = static_names | dynamic_names

    coverage = _compute_rule_coverage(static_names, dynamic_names)

    logger.info(
        "correlating %d static + %d dynamic rules (coverage: %.1f%%)",
        len(static_names), len(dynamic_names), coverage * 100,
    )

    correlated: list[CorrelatedRule] = []

    for rule_name in sorted(all_names):
        in_static  = rule_name in static_names
        in_dynamic = rule_name in dynamic_names

        static_meta  = static_rules.get(rule_name)
        dynamic_meta = dynamic_rules.get(rule_name)

        namespace: Optional[str] = None
        if static_meta is not None:
            namespace = static_meta.meta.namespace
        elif dynamic_meta is not None:
            namespace = dynamic_meta.meta.namespace

        static_match_count  = len(static_meta.matches)  if static_meta  else 0
        dynamic_match_count = len(dynamic_meta.matches) if dynamic_meta else 0

        tier, confidence, explanation = _assign_tier(in_static, in_dynamic, coverage)

        correlated.append(CorrelatedRule(
            rule_name           = rule_name,
            namespace           = namespace,
            tier                = tier,
            confidence          = confidence,
            coverage            = coverage,
            static_match_count  = static_match_count,
            dynamic_match_count = dynamic_match_count,
            explanation         = explanation,
            static_rule_meta    = static_meta,
            dynamic_rule_meta   = dynamic_meta,
        ))

    _TIER_ORDER = {
        EvidenceTier.CONFIRMED_RUNTIME:         0,
        EvidenceTier.STATIC_ONLY_UNEXECUTED:    1,
        EvidenceTier.INCONCLUSIVE_LOW_COVERAGE: 2,
        EvidenceTier.DYNAMIC_ONLY:              3,
    }
    correlated.sort(key=lambda r: (_TIER_ORDER[r.tier], r.rule_name))

    return CorrelationReport(
        results             = correlated,
        coverage            = coverage,
        total_static_rules  = len(static_names),
        total_dynamic_rules = len(dynamic_names),
        confirmed_count     = sum(1 for r in correlated if r.tier == EvidenceTier.CONFIRMED_RUNTIME),
        unexecuted_count    = sum(1 for r in correlated if r.tier == EvidenceTier.STATIC_ONLY_UNEXECUTED),
        inconclusive_count  = sum(1 for r in correlated if r.tier == EvidenceTier.INCONCLUSIVE_LOW_COVERAGE),
        dynamic_only_count  = sum(1 for r in correlated if r.tier == EvidenceTier.DYNAMIC_ONLY),
    )


# ---------------------------------------------------------------------------
# CLI reporting helper
# ---------------------------------------------------------------------------

def print_correlation_report(report: CorrelationReport) -> None:
    """Print a human-readable correlation report grouped by evidence tier."""

    _LABELS = {
        EvidenceTier.CONFIRMED_RUNTIME:         "Confirmed at Runtime",
        EvidenceTier.STATIC_ONLY_UNEXECUTED:    "Static Match — Code Path Unexecuted",
        EvidenceTier.INCONCLUSIVE_LOW_COVERAGE: "Inconclusive — Low Sandbox Coverage",
        EvidenceTier.DYNAMIC_ONLY:              "Dynamic Only (no static equivalent)",
    }
    _ORDER = list(_LABELS)

    by_tier: dict[EvidenceTier, list[CorrelatedRule]] = {t: [] for t in _ORDER}
    for r in report.results:
        by_tier[r.tier].append(r)

    print("=" * 70)
    print("  Capa Static-Dynamic Correlation Report")
    print("=" * 70)

    for tier in _ORDER:
        rules = by_tier[tier]
        if not rules:
            continue
        print(f"\n  [{_LABELS[tier]}]")
        print(f"  {'─' * 66}")
        for r in rules:
            print(f"  • {r.rule_name}")
            if r.namespace:
                print(f"    namespace  : {r.namespace}")
            print(
                f"    matches    : static={r.static_match_count}  "
                f"dynamic={r.dynamic_match_count}  "
                f"confidence={r.confidence:.1f}"
            )

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Static rules          : {report.total_static_rules}")
    print(f"  Dynamic rules         : {report.total_dynamic_rules}")
    print(f"  Rule-overlap coverage : {report.coverage:.1%}")
    print()
    print(f"  CONFIRMED_RUNTIME         : {report.confirmed_count}")
    print(f"  STATIC_ONLY_UNEXECUTED    : {report.unexecuted_count}")
    print(f"  INCONCLUSIVE_LOW_COVERAGE : {report.inconclusive_count}")
    print(f"  DYNAMIC_ONLY              : {report.dynamic_only_count}")
    print()
    print("  STATIC_ONLY_UNEXECUTED = code path not triggered during sandbox run,")
    print("  not a false positive.")
    print("=" * 70)