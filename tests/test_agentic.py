import threading

from PIL import Image

from imgsearch.config import AppConfig
from imgsearch.core.agentic import AgenticSearch, Plan, rule_based_plan
from imgsearch.core.registry import build_backends
from imgsearch.core.services import SearchService
from imgsearch.graph.builder import build_records
from imgsearch.graph.memory_graph import MemoryGraphStore
from imgsearch.index.indexer import Indexer
from imgsearch.store.chroma_store import ChromaStore


# ---------------------------------------------------------------- planner
def test_rule_based_plan_detects_required_objects():
    names = {"0": "사람", "1": "오토바이", "2": "휴대폰"}
    p = rule_based_plan("사람과 오토바이가 함께 있는 이미지", names)
    assert set(p.required_objects) == {"사람", "오토바이"}
    assert p.excluded_objects == []
    assert p.semantic  # expanded search text


def test_rule_based_plan_detects_exclusion():
    names = {"0": "사람", "1": "오토바이"}
    p = rule_based_plan("오토바이가 있고 사람은 없는 이미지", names)
    assert "오토바이" in p.required_objects
    assert "사람" in p.excluded_objects
    assert "사람" not in p.required_objects


def test_rule_based_plan_deterministic():
    names = {"0": "사람", "1": "오토바이"}
    a = rule_based_plan("사람 오토바이", names)
    b = rule_based_plan("사람 오토바이", names)
    assert (a.semantic, a.required_objects, a.excluded_objects) == (
        b.semantic, b.required_objects, b.excluded_objects
    )


# ---------------------------------------------------------------- pipeline
def _setup(tmp_path):
    ds = tmp_path / "ds" / "f"
    ds.mkdir(parents=True)
    specs = [("0",), ("0", "1"), ("1",), ("0", "1"), ("2",)]
    for i, classes in enumerate(specs):
        Image.new("RGB", (48, 48), (i * 40 % 255, i * 25 % 255, 60)).save(ds / f"im{i}.jpg")
        (ds / f"im{i}.txt").write_text("".join(f"{c} 0.5 0.5 0.3 0.3\n" for c in classes))
    cfg = AppConfig(
        dataset_root=str(tmp_path / "ds"),
        index_granularity="image",
        class_names={"0": "사람", "1": "오토바이", "2": "휴대폰"},
        search_mode="hybrid",
    )
    emb, cap, chat = build_backends(cfg)
    store = ChromaStore(tmp_path / "chroma")
    Indexer(emb, cap, store, cfg).build(str(tmp_path / "ds"), full_rebuild=True)
    svc = SearchService(emb, store, chat, cfg)
    graph = MemoryGraphStore()
    graph.build(build_records(str(tmp_path / "ds"), cfg))
    return cfg, svc, graph, chat


def test_agentic_graph_filter_requires_all_objects(tmp_path):
    cfg, svc, graph, chat = _setup(tmp_path)
    agent = AgenticSearch(svc, graph, chat, cfg)
    steps: list[str] = []
    res = agent.run("사람과 오토바이가 함께 있는 이미지", step_cb=steps.append, k=10)

    # only im1 and im3 have BOTH 사람 and 오토바이
    assert res.hits
    assert all("오토바이" in h.caption and "사람" in h.caption for h in res.hits)
    assert len(res.hits) == 2
    assert any("Graph filter" in s for s in steps)  # A2A trace emitted
    assert any("Plan" in s for s in steps)


def test_agentic_exclusion(tmp_path):
    cfg, svc, graph, chat = _setup(tmp_path)
    agent = AgenticSearch(svc, graph, chat, cfg)
    res = agent.run("오토바이가 있고 휴대폰은 없는 이미지", step_cb=lambda s: None, k=10)
    assert res.hits
    assert all("오토바이" in h.caption for h in res.hits)
    assert all("휴대폰" not in h.caption for h in res.hits)


def test_agentic_step_cb_called_on_caller_thread(tmp_path):
    # the callback must be invoked synchronously on the thread that calls run()
    # (the UI relies on this so it can hand in a Signal.emit that marshals to GUI)
    cfg, svc, graph, chat = _setup(tmp_path)
    agent = AgenticSearch(svc, graph, chat, cfg)
    seen: list[int] = []
    agent.run("사람", step_cb=lambda s: seen.append(threading.get_ident()), k=5)
    assert seen and all(t == threading.get_ident() for t in seen)


def test_agentic_graph_mismatch_falls_back(tmp_path):
    # graph built from a DIFFERENT root than the index -> filter must be skipped,
    # returning the unfiltered hybrid hits rather than an empty result
    cfg, svc, graph, chat = _setup(tmp_path)
    other = MemoryGraphStore()
    other.build([])  # empty graph stands in for a mismatched/missing one
    from imgsearch.graph.base import GraphRecord

    other.build([GraphRecord("Z:/elsewhere/x.jpg", "Z:/elsewhere", ["사람", "오토바이"])])
    agent = AgenticSearch(svc, other, chat, cfg)
    steps: list[str] = []
    res = agent.run("사람과 오토바이", step_cb=steps.append, k=10)
    assert res.hits  # not empty — filter was skipped
    assert any("skipped" in s for s in steps)


def test_agentic_uses_chat_plan_when_available(tmp_path):
    cfg, svc, graph, chat = _setup(tmp_path)

    class _PlanningChat:
        name = "fake"

        def refine_query(self, t):
            return t

        def summarize(self, q, h):
            return ""

        def plan(self, query, class_names=None):
            return {"semantic": "오토바이", "required_objects": ["오토바이"], "excluded_objects": []}

    agent = AgenticSearch(svc, graph, _PlanningChat(), cfg)
    res = agent.run("아무거나", step_cb=lambda s: None, k=10)
    assert res.hits
    assert all("오토바이" in h.caption for h in res.hits)


def test_agentic_bad_plan_falls_back_to_rules(tmp_path):
    cfg, svc, graph, chat = _setup(tmp_path)

    class _BrokenChat:
        name = "broken"

        def refine_query(self, t):
            return t

        def summarize(self, q, h):
            return ""

        def plan(self, query, class_names=None):
            raise ValueError("bad json")

    agent = AgenticSearch(svc, graph, _BrokenChat(), cfg)
    res = agent.run("사람과 오토바이", step_cb=lambda s: None, k=10)
    assert res.hits  # rule-based fallback kept it working
