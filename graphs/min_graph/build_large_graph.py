import json
import os
import random
from typing import Dict, List

import jsonlines

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

random.seed(22)

FREQUENCY_LABELS = [
    "very_rare",
    "occasional",
    "frequent",
    "very_frequent",
]

PREDICATES = [
    "biolink:related_to",
    "biolink:affects",
    "biolink:interacts_with",
]


def _generate_nodes(num_nodes: int) -> Dict[str, List[Dict[str, str]]]:
    """Create node identifiers grouped by type."""
    num_genes = int(num_nodes * 0.1)
    num_diseases = int(num_nodes * 0.2)
    num_symptoms = int(num_nodes * 0.3)
    num_processes = num_nodes - num_genes - num_diseases - num_symptoms

    genes = [f"G{i}" for i in range(num_genes)]
    diseases = [f"D{i}" for i in range(num_diseases)]
    symptoms = [f"S{i}" for i in range(num_symptoms)]
    processes = [f"P{i}" for i in range(num_processes)]

    nodes = (
        [{"id": g, "name": g, "category": "biolink:Gene"} for g in genes]
        + [{"id": d, "name": d, "category": "biolink:Disease"} for d in diseases]
        + [
            {"id": s, "name": s, "category": "biolink:PhenotypicFeature"}
            for s in symptoms
        ]
        + [
            {"id": p, "name": p, "category": "biolink:BiologicalProcess"}
            for p in processes
        ]
    )

    return {
        "genes": genes,
        "diseases": diseases,
        "symptoms": symptoms,
        "processes": processes,
        "nodes": nodes,
    }


def _generate_edges(node_dict: Dict[str, List[str]]) -> (List[Dict[str, str]], Dict[str, List[str]], Dict[str, Dict[str, str]]):
    """Create edges along with ground truth maps."""
    edge_id = 0
    edges = []
    genes = node_dict["genes"]
    diseases = node_dict["diseases"]
    symptoms = node_dict["symptoms"]
    processes = node_dict["processes"]
    all_nodes = genes + diseases + symptoms + processes

    gene_to_diseases: Dict[str, List[str]] = {}
    for gene in genes:
        num_links = random.randint(1, min(3, len(diseases)))
        assoc = random.sample(diseases, num_links)
        gene_to_diseases[gene] = assoc
        for disease in assoc:
            edges.append(
                {
                    "id": edge_id,
                    "subject": gene,
                    "object": disease,
                    "predicate": "gene_associated_with_disease",
                }
            )
            edge_id += 1

    disease_to_symptom_freq: Dict[str, Dict[str, str]] = {}
    for disease in diseases:
        num_links = random.randint(1, min(5, len(symptoms)))
        assoc = random.sample(symptoms, num_links)
        disease_to_symptom_freq[disease] = {
            symptom: random.choice(FREQUENCY_LABELS) for symptom in assoc
        }
        for symptom in assoc:
            edges.append(
                {
                    "id": edge_id,
                    "subject": disease,
                    "object": symptom,
                    "predicate": "disease_has_symptom",
                }
            )
            edge_id += 1

    # Additional random edges to make the graph denser
    num_random_edges = len(all_nodes) * 2
    for _ in range(num_random_edges):
        subj, obj = random.sample(all_nodes, 2)
        edges.append(
            {
                "id": edge_id,
                "subject": subj,
                "object": obj,
                "predicate": random.choice(PREDICATES),
            }
        )
        edge_id += 1

    return edges, gene_to_diseases, disease_to_symptom_freq


def _write_graph(out_dir: str, nodes: List[Dict[str, str]], edges: List[Dict[str, str]], gene_to_diseases: Dict[str, List[str]], disease_to_symptom_freq: Dict[str, Dict[str, str]]):
    os.makedirs(out_dir, exist_ok=True)
    with jsonlines.open(os.path.join(out_dir, "nodes.jsonl"), mode="w") as writer:
        writer.write_all(nodes)
    with jsonlines.open(os.path.join(out_dir, "edges.jsonl"), mode="w") as writer:
        writer.write_all(edges)
    with open(os.path.join(out_dir, "gene_to_diseases.json"), "w") as f:
        json.dump(gene_to_diseases, f, indent=2)
    with open(os.path.join(out_dir, "disease_symptom_frequencies.json"), "w") as f:
        json.dump(disease_to_symptom_freq, f, indent=2)


def build_graph(num_nodes: int, output_dir_name: str):
    node_dict = _generate_nodes(num_nodes)
    edges, gene_to_diseases, disease_to_symptom_freq = _generate_edges(node_dict)
    _write_graph(
        os.path.join(GRAPHS_DIR, output_dir_name),
        node_dict["nodes"],
        edges,
        gene_to_diseases,
        disease_to_symptom_freq,
    )
    print(
        f"Generated graph '{output_dir_name}' with {len(node_dict['nodes'])} nodes and {len(edges)} edges"
    )


def main():
    graphs_to_build = {
        "large_graph_1k": 1000,
        "large_graph_10k": 10000,
        "large_graph_100k": 100000,
    }
    for name, size in graphs_to_build.items():
        build_graph(size, name)


if __name__ == "__main__":
    main()
