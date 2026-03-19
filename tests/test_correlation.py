from __future__ import annotations
import datetime
from dataclasses import dataclass
from unittest.mock import MagicMock
import pytest
import capa.rules, capa.version
import capa.features.freeze as frz
import capa.render.result_document as rdoc
from capa.features.address import AbsoluteVirtualAddress
from capa.correlation.engine import (
    EvidenceTier, CorrelationReport, CorrelatedRule,
    correlate_results, _compute_rule_coverage,
    print_correlation_report, COVERAGE_THRESHOLD,
)

def _frozen_addr(va=0):
    import capa.features.freeze as frz
    from capa.features.address import AbsoluteVirtualAddress
    return frz.Address.from_capa(AbsoluteVirtualAddress(va))

def _api_match(api_name, va=0):
    import capa.features.freeze.features as frzf
    leaf = rdoc.Match(success=True, node=rdoc.FeatureNode(feature=frzf.APIFeature(api=api_name)), children=(), locations=(_frozen_addr(va),), captures={})
    return rdoc.Match(success=True, node=rdoc.StatementNode(statement=rdoc.CompoundStatement(type=rdoc.CompoundStatementType.OR)), children=(leaf,), locations=(), captures={})

def _make_rule_matches(rule_name, apis, namespace=''):
    meta = rdoc.RuleMetadata(name=rule_name, namespace=namespace, authors=(), scopes=capa.rules.Scopes(static=capa.rules.Scope.FILE, dynamic=capa.rules.Scope.PROCESS), attack=(), mbc=(), references=(), examples=(), description='', lib=False, is_subscope_rule=False, maec=rdoc.MaecMetadata())
    matches = tuple((_frozen_addr(i*0x1000), _api_match(api, i*0x1000)) for i, api in enumerate(apis))
    return rdoc.RuleMatches(meta=meta, source='', matches=matches)

def _make_metadata(flavor=rdoc.Flavor.STATIC):
    if flavor == rdoc.Flavor.STATIC:
        analysis = rdoc.StaticAnalysis(format='pe', arch='x86', os='windows', extractor='test', rules=(), base_address=_frozen_addr(0x400000), layout=rdoc.StaticLayout(functions=()), feature_counts=rdoc.StaticFeatureCounts(file=0, functions=()), library_functions=())
    else:
        analysis = rdoc.DynamicAnalysis(format='pe', arch='x86', os='windows', extractor='test', rules=(), layout=rdoc.DynamicLayout(processes=()), feature_counts=rdoc.DynamicFeatureCounts(file=0, processes=()))
    return rdoc.Metadata(timestamp=datetime.datetime(2025,1,1), version=capa.version.__version__, argv=None, sample=rdoc.Sample(md5='a'*32, sha1='b'*40, sha256='c'*64, path='test.exe'), flavor=flavor, analysis=analysis)

def _make_doc(rules, namespaces=None, flavor=rdoc.Flavor.STATIC):
    ns = namespaces or {}
    return rdoc.ResultDocument(meta=_make_metadata(flavor), rules={name: _make_rule_matches(name, apis, ns.get(name,'')) for name, apis in rules.items()})

class TestComputeRuleCoverage:
    def test_full_overlap(self):
        assert _compute_rule_coverage({'a','b','c'}, {'a','b','c','d'}) == pytest.approx(1.0)
    def test_partial_overlap(self):
        assert _compute_rule_coverage({'a','b','c'}, {'a'}) == pytest.approx(1/3)
    def test_no_overlap(self):
        assert _compute_rule_coverage({'a','b'}, {'c','d'}) == pytest.approx(0.0)
    def test_empty_static(self):
        assert _compute_rule_coverage(set(), {'a'}) == pytest.approx(0.0)
    def test_empty_both(self):
        assert _compute_rule_coverage(set(), set()) == pytest.approx(0.0)

class TestCorrelateResults:
    def test_confirmed_runtime(self):
        static = _make_doc({'read file': ['CreateFileA','ReadFile']})
        dynamic = _make_doc({'read file': ['CreateFileA','ReadFile']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        r = next(x for x in report.results if x.rule_name == 'read file')
        assert r.tier == EvidenceTier.CONFIRMED_RUNTIME
        assert r.confidence == pytest.approx(0.9)
    def test_static_only_high_coverage(self):
        static = _make_doc({'read file': ['CreateFileA'], 'write file': ['WriteFile'], 'screenshot': ['GetDC','BitBlt']})
        dynamic = _make_doc({'read file': ['CreateFileA'], 'write file': ['WriteFile']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        r = next(x for x in report.results if x.rule_name == 'screenshot')
        assert r.tier == EvidenceTier.STATIC_ONLY_UNEXECUTED
    def test_inconclusive_low_coverage(self):
        static = _make_doc({'inject': ['OpenProcess','VirtualAllocEx','WriteProcessMemory','CreateRemoteThread']})
        dynamic = _make_doc({'unrelated': ['NtQuerySystemInformation']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        r = next(x for x in report.results if x.rule_name == 'inject')
        assert r.tier == EvidenceTier.INCONCLUSIVE_LOW_COVERAGE
        assert r.coverage < COVERAGE_THRESHOLD
    def test_dynamic_only_rule_included(self):
        static = _make_doc({'read file': ['CreateFileA']})
        dynamic = _make_doc({'read file': ['CreateFileA'], 'dynamic only': ['SomeAPI']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        r = next(x for x in report.results if x.rule_name == 'dynamic only')
        assert r.tier == EvidenceTier.DYNAMIC_ONLY
    def test_namespace_preserved(self):
        static = _make_doc({'inject': ['OpenProcess']}, namespaces={'inject': 'host-interaction/process/inject'})
        dynamic = _make_doc({'inject': ['OpenProcess']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        r = next(x for x in report.results if x.rule_name == 'inject')
        assert r.namespace == 'host-interaction/process/inject'
    def test_empty_static_doc(self):
        static = _make_doc({})
        dynamic = _make_doc({'read file': ['CreateFileA']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        assert report.total_static_rules == 0
        assert any(r.tier == EvidenceTier.DYNAMIC_ONLY for r in report.results)
    def test_counts_consistent(self):
        static = _make_doc({'read file': ['CreateFileA'], 'write file': ['WriteFile'], 'inject': ['OpenProcess','VirtualAllocEx']})
        dynamic = _make_doc({'read file': ['CreateFileA']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        total = report.confirmed_count + report.unexecuted_count + report.inconclusive_count + report.dynamic_only_count
        assert total == len(report.results)
    def test_render_smoke(self, capsys):
        static = _make_doc({'read file': ['CreateFileA']})
        dynamic = _make_doc({'read file': ['CreateFileA']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        print_correlation_report(report)
        captured = capsys.readouterr()
        assert 'CONFIRMED_RUNTIME' in captured.out
        assert 'read file' in captured.out
    def test_static_match_count_correct(self):
        static = _make_doc({'read file': ['CreateFileA','ReadFile','CloseHandle']})
        dynamic = _make_doc({}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        r = next(x for x in report.results if x.rule_name == 'read file')
        assert r.static_match_count == 3
    def test_coverage_stored_on_results(self):
        static = _make_doc({'read file': ['CreateFileA'], 'write file': ['WriteFile']})
        dynamic = _make_doc({'read file': ['CreateFileA']}, flavor=rdoc.Flavor.DYNAMIC)
        report = correlate_results(static, dynamic)
        assert report.coverage == pytest.approx(0.5)
        for r in report.results:
            assert r.coverage == pytest.approx(report.coverage)

class TestNoiseFilter:
    def _make_analysis(self, processes, submission_name='malware.exe'):
        @dataclass
        class FakeMP:
            pid: int; ppid: int; monitor_id: int; origin_monitor_id: int; image_name: str; filename: str
        analysis = MagicMock()
        analysis.submission_name = submission_name
        analysis.monitor_processes = {mid: FakeMP(pid, ppid, mid, origin, img, fname) for mid, (pid, ppid, origin, img, fname) in processes.items()}
        return analysis
    def test_single_root(self):
        from capa.features.extractors.vmray.noise import get_relevant_monitor_ids
        analysis = self._make_analysis({1: (1234, 0, 1, 'malware.exe', 'malware.exe')})
        assert get_relevant_monitor_ids(analysis) == frozenset({1})
    def test_root_and_child(self):
        from capa.features.extractors.vmray.noise import get_relevant_monitor_ids
        analysis = self._make_analysis({1: (1234, 0, 1, 'malware.exe', 'malware.exe'), 2: (5678, 1234, 1, 'cmd.exe', 'cmd.exe'), 3: (9999, 0, 3, 'explorer.exe', 'explorer.exe')})
        result = get_relevant_monitor_ids(analysis)
        assert 1 in result and 2 in result and 3 not in result
    def test_transitive_grandchild(self):
        from capa.features.extractors.vmray.noise import get_relevant_monitor_ids
        analysis = self._make_analysis({1: (100, 0, 1, 'malware.exe', 'malware.exe'), 2: (200, 100, 1, 'cmd.exe', 'cmd.exe'), 3: (300, 200, 2, 'ps.exe', 'ps.exe'), 4: (400, 0, 4, 'svchost.exe', 'svchost.exe')})
        assert get_relevant_monitor_ids(analysis) == frozenset({1, 2, 3})
    def test_empty_returns_empty(self):
        from capa.features.extractors.vmray.noise import get_relevant_monitor_ids
        assert get_relevant_monitor_ids(self._make_analysis({})) == frozenset()
    def test_noise_processes_excluded(self):
        from capa.features.extractors.vmray.noise import get_relevant_monitor_ids
        analysis = self._make_analysis({1: (100, 0, 1, 'malware.exe', 'malware.exe'), 2: (200, 0, 2, 'explorer.exe', 'explorer.exe'), 3: (300, 0, 3, 'svchost.exe', 'svchost.exe')})
        result = get_relevant_monitor_ids(analysis)
        assert 1 in result and 2 not in result and 3 not in result
