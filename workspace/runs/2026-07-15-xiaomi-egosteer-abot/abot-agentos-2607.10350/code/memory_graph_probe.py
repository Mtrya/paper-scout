"""
ABot-AgentOS memory graph probe.

Implements a miniature version of the typed, source-grounded multi-modal memory
described in ABot-AgentOS (arXiv:2607.10350). The goal is to make the method
concrete enough that the mechanism can be inspected, modified, and rerun.

What this demonstrates:
1. Typed graph schema (entity, event, place, session, evidence nodes).
2. Multi-modal memory writing from a synthetic egocentric observation.
3. Hybrid seed selection (semantic + lexical + metadata + type).
4. Evidence-subgraph expansion along typed edges.
5. Retrieval trace for audit/self-evolution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class Node:
    id: str
    type: str  # e.g. source, evidence, entity, place, session, semantic_event
    content: str
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    type: str  # e.g. temporal_order, containment, observation, participation, location, identity, spatial, provenance
    info: Dict[str, Any] = field(default_factory=dict)


class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adj: Dict[str, List[Edge]] = {}

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        if node.id not in self.adj:
            self.adj[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        self.adj.setdefault(edge.src, []).append(edge)

    def neighbors(self, node_id: str, edge_type: Optional[str] = None) -> List[Node]:
        out = []
        for e in self.adj.get(node_id, []):
            if edge_type is None or e.type == edge_type:
                out.append(self.nodes[e.dst])
        return out


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def semantic_similarity(q: str, v: Node) -> float:
    """Tiny embedding-free semantic proxy: word overlap over meaningful words."""
    q_tokens = set(normalize(q).split())
    v_tokens = set(normalize(v.content + " " + json.dumps(v.info)).split())
    if not q_tokens:
        return 0.0
    return len(q_tokens & v_tokens) / len(q_tokens)


def lexical_overlap(q: str, v: Node) -> float:
    """Exact token overlap."""
    q_tokens = set(normalize(q).split())
    v_text = normalize(v.content + " " + json.dumps(v.info))
    v_tokens = set(v_text.split())
    if not q_tokens or not v_tokens:
        return 0.0
    return len(q_tokens & v_tokens) / len(q_tokens | v_tokens)


def metadata_compat(q: str, v: Node) -> float:
    """Boost nodes whose metadata mentions query entities/time/place."""
    score = 0.0
    qn = normalize(q)
    info_text = normalize(json.dumps(v.info))
    for key in ["time_ref", "place", "source_id", "modality"]:
        if key in v.info and str(v.info[key]).lower() in qn:
            score += 0.25
    if v.type in qn:
        score += 0.1
    if any(tok in info_text for tok in qn.split()):
        score += 0.1
    return min(score, 1.0)


def type_preference(q: str, v: Node) -> float:
    """Favor node types expected by query keywords."""
    qn = normalize(q)
    type_map = {
        "person": ["entity"],
        "object": ["entity"],
        "where": ["place"],
        "when": ["semantic_event"],
        "what happened": ["semantic_event"],
        "image": ["evidence"],
        "frame": ["evidence"],
    }
    for key, types in type_map.items():
        if key in qn and v.type in types:
            return 1.0
    return 0.0


def hybrid_seed_score(
    q: str,
    v: Node,
    lambdas: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1),
) -> float:
    """
    Implements Equation (2) from the paper:
    s(q,v) = λ_sem s_sem + λ_lex s_lex + λ_meta s_meta + λ_type s_type
    """
    sem = semantic_similarity(q, v)
    lex = lexical_overlap(q, v)
    meta = metadata_compat(q, v)
    typ = type_preference(q, v)
    return lambdas[0] * sem + lambdas[1] * lex + lambdas[2] * meta + lambdas[3] * typ


def expand_evidence_subgraph(
    graph: MemoryGraph,
    seed_ids: List[str],
    depth: int = 2,
    edge_budget: int = 20,
    preferred_edges: Optional[Set[str]] = None,
) -> Tuple[Set[str], List[Edge]]:
    """
    Expand seed nodes along typed edges under a fixed depth/evidence budget.
    Returns the visited node ids and the collected edges.
    """
    visited = set(seed_ids)
    collected: List[Edge] = []
    frontier = list(seed_ids)
    for _ in range(depth):
        next_frontier = []
        for node_id in frontier:
            for e in graph.adj.get(node_id, []):
                if preferred_edges and e.type not in preferred_edges:
                    continue
                if len(collected) >= edge_budget:
                    break
                collected.append(e)
                if e.dst not in visited:
                    visited.add(e.dst)
                    next_frontier.append(e.dst)
        frontier = next_frontier
        if not frontier:
            break
    return visited, collected


def write_observation_to_memory(
    graph: MemoryGraph,
    session_id: str,
    timestamp: str,
    place_name: str,
    utterance: Optional[str],
    image_path: Optional[str],
    visible_objects: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Converts a raw observation into typed, source-grounded graph records.
    Mirrors the paper's example of writing 'I adopted a Maltese dog yesterday'
    as a time-grounded semantic event with identity, evidence, and provenance.
    """
    visible_objects = visible_objects or []

    # Source/session container
    session = Node(
        id=f"session:{session_id}",
        type="session",
        content=f"Session {session_id}",
        info={"session_id": session_id, "start_time": timestamp},
    )
    graph.add_node(session)

    # Place node
    place = Node(
        id=f"place:{place_name}",
        type="place",
        content=place_name,
        info={"name": place_name},
    )
    graph.add_node(place)

    event_id = f"event:{session_id}:{timestamp}"
    event = Node(
        id=event_id,
        type="semantic_event",
        content=utterance or "observation",
        info={
            "time_ref": timestamp,
            "session_id": session_id,
            "place": place_name,
            "confidence": 0.95,
            "adapter_version": "probe-v1",
            "extractor_model": "probe-llm",
        },
    )
    graph.add_node(event)
    graph.add_edge(Edge(src=session.id, dst=event.id, type="provenance"))
    graph.add_edge(Edge(src=event.id, dst=place.id, type="location"))

    # Image evidence node
    if image_path:
        evidence = Node(
            id=f"evidence:{image_path}",
            type="evidence",
            content=f"Image evidence: {image_path}",
            info={
                "modality": "image",
                "source_id": image_path,
                "time_ref": timestamp,
                "place": place_name,
            },
        )
        graph.add_node(evidence)
        graph.add_edge(Edge(src=event.id, dst=evidence.id, type="provenance"))

    # Visible entities
    for obj in visible_objects:
        entity = Node(
            id=f"entity:{obj['name']}:{session_id}",
            type="entity",
            content=obj.get("description", obj["name"]),
            info={
                "name": obj["name"],
                "category": obj.get("category"),
                "state": obj.get("state"),
                "last_seen": timestamp,
                "place": place_name,
            },
        )
        graph.add_node(entity)
        graph.add_edge(Edge(src=event.id, dst=entity.id, type="observation"))
        graph.add_edge(Edge(src=entity.id, dst=place.id, type="location"))

    return event_id


def retrieve_memory_answer(
    graph: MemoryGraph,
    query: str,
    top_k_seeds: int = 3,
    depth: int = 2,
    edge_budget: int = 12,
) -> Dict[str, Any]:
    """
    Run the hybrid retriever and return an evidence subgraph plus trace.
    """
    scores = [
        (node_id, hybrid_seed_score(query, node))
        for node_id, node in graph.nodes.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    seed_ids = [nid for nid, _ in scores[:top_k_seeds]]

    visited, edges = expand_evidence_subgraph(
        graph,
        seed_ids,
        depth=depth,
        edge_budget=edge_budget,
        preferred_edges={"location", "observation", "provenance", "temporal_order"},
    )

    evidence_nodes = [graph.nodes[nid] for nid in visited]
    trace = {
        "query": query,
        "seed_nodes": seed_ids,
        "seed_scores": {nid: sc for nid, sc in scores[:top_k_seeds]},
        "retrieved_node_ids": list(visited),
        "retrieved_edges": [(e.src, e.type, e.dst) for e in edges],
        "evidence": [
            {"id": n.id, "type": n.type, "content": n.content, "info": n.info}
            for n in evidence_nodes
        ],
    }
    return trace


def main():
    graph = MemoryGraph()

    # Write a synthetic multi-modal observation.
    write_observation_to_memory(
        graph,
        session_id="home-001",
        timestamp="2026-07-14T10:00:00Z",
        place_name="living_room",
        utterance="I adopted a Maltese dog yesterday.",
        image_path="frames/home-001/000100.jpg",
        visible_objects=[
            {"name": "Maltese_dog", "category": "animal", "state": "sitting_on_sofa"},
            {"name": "red_cushion", "category": "object", "state": "on_sofa"},
        ],
    )

    # Write a later observation that updates state.
    write_observation_to_memory(
        graph,
        session_id="home-002",
        timestamp="2026-07-15T08:00:00Z",
        place_name="backyard",
        utterance="The dog is playing in the backyard.",
        image_path="frames/home-002/000020.jpg",
        visible_objects=[
            {"name": "Maltese_dog", "category": "animal", "state": "running"},
        ],
    )

    # Add temporal continuity between the two dog observations.
    dog1 = "entity:Maltese_dog:home-001"
    dog2 = "entity:Maltese_dog:home-002"
    if dog1 in graph.nodes and dog2 in graph.nodes:
        graph.add_edge(
            Edge(
                src=dog1,
                dst=dog2,
                type="identity",
                info={"reason": "same_name_and_category"},
            )
        )
        graph.add_edge(
            Edge(
                src=dog1,
                dst=dog2,
                type="temporal_order",
                info={"before": "2026-07-14T10:00:00Z", "after": "2026-07-15T08:00:00Z"},
            )
        )

    # Query the memory.
    queries = [
        "What dog did I adopt?",
        "Where is the dog now?",
        "Show me the image evidence from the living room.",
    ]

    for q in queries:
        trace = retrieve_memory_answer(graph, q)
        print(f"\n=== QUERY: {q} ===")
        print(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
