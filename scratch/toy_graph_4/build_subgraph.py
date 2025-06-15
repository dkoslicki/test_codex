import json
import os
from typing import List, Dict, Set, Tuple

import jsonlines
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOVER_URL = "https://kg2cplover.rtx.ai:9990"


def get_neighbor_ids(node_ids: Set[str], category_constraints: List[str] = ["biolink:NamedThing"]) -> Set[str]:
    print(f"Getting neighbors for {len(node_ids)} nodes")
    query = {"node_ids": list(node_ids), "categories": category_constraints}
    response = requests.post(f"{PLOVER_URL}/get_neighbors", json=query)
    response_json = response.json()
    return {neighbor_id for node_id, neighbor_ids in response_json.items()
            for neighbor_id in neighbor_ids}


def do_doubly_pinned_query(node_ids_a: Set[str], node_ids_b: Set[str],
                           exclude_sources: Set[str] = set()) -> Tuple[dict, dict]:
    print(f"Doing doubly pinned query between {len(node_ids_a)} and {len(node_ids_b)} nodes")
    attribute_constraints = [{"id": "primary_knowledge_source",
                              "name": f"exclude {exclude_source} edges",
                              "value": exclude_source,
                              "operator": "==",
                              "not": True
                              } for exclude_source in exclude_sources]
    query = {"nodes": {"n00": {"ids": list(node_ids_a)}, "n01": {"ids": list(node_ids_b)}},
             "edges": {"e01": {"subject": "n00", "object": "n01", "attribute_constraints": attribute_constraints}}}
    response = requests.post(f"{PLOVER_URL}/query", json=query)
    response_json = response.json()
    return create_jsonl_nodes(response_json), create_jsonl_edges(response_json)


def create_jsonl_nodes(trapi_response: dict) -> Dict[str, dict]:
    jsonl_nodes_map = dict()
    for node_id, node in trapi_response["message"]["knowledge_graph"]["nodes"].items():
        jsonl_node = {"id": node_id, "name": node["name"], "category": node["categories"][0]}
        jsonl_nodes_map[node_id] = jsonl_node
    return jsonl_nodes_map


def create_jsonl_edges(trapi_response: dict) -> Dict[str, dict]:
    jsonl_edges_map = dict()
    for edge_id, edge in trapi_response["message"]["knowledge_graph"]["edges"].items():
        jsonl_edge = {"id": edge_id, "subject": edge["subject"], "predicate": edge["predicate"],
                      "object": edge["object"]}
        jsonl_edges_map[edge_id] = jsonl_edge
    return jsonl_edges_map


def main():
    # Load our disease, gene, and symptom frequency info
    with open(f"{SCRIPT_DIR}/gene_to_diseases.json") as gene_to_diseases_file:
        gene_to_diseases = json.load(gene_to_diseases_file)
    with open(f"{SCRIPT_DIR}/disease_symptom_frequencies.json") as frequencies_file:
        disease_symptom_frequencies = json.load(frequencies_file)
    diseases = set(disease_symptom_frequencies)
    genes = set(gene_to_diseases)
    symptoms = {symptom_id for disease, symptoms in disease_symptom_frequencies.items() for symptom_id in symptoms}
    print(diseases)
    print(genes)
    print(symptoms)

    nodes_map = dict()
    edges_map = dict()
    intermediate_node_categories = ["biolink:Protein", "biolink:Pathway",
                                    "biolink:BiologicalProcess", "biolink:PhysiologicalProcess"]

    # Get one-hop paths from Plover
    nodes_map_1, edges_map_1 = do_doubly_pinned_query(genes, symptoms, exclude_sources={"infores:ordo"})
    nodes_map.update(nodes_map_1)
    edges_map.update(edges_map_1)
    print(f"Have {len(nodes_map)} nodes, {len(edges_map)} edges after getting one-hop paths.")

    # Get two-hop paths from Plover
    neighbors_1 = get_neighbor_ids(genes, category_constraints=intermediate_node_categories)
    nodes_map_2, edges_map_2 = do_doubly_pinned_query(neighbors_1, symptoms)
    nodes_map.update(nodes_map_2)
    edges_map.update(edges_map_2)
    nodes_map_1, edges_map_1 = do_doubly_pinned_query(genes, set(nodes_map_2))
    nodes_map.update(nodes_map_1)
    edges_map.update(edges_map_1)
    print(f"Have {len(nodes_map)} nodes, {len(edges_map)} edges after getting two-hop paths.")

    # Get three-hop paths from Plover
    neighbors_2 = get_neighbor_ids(neighbors_1, category_constraints=intermediate_node_categories)
    nodes_map_3, edges_map_3 = do_doubly_pinned_query(neighbors_2, symptoms)
    nodes_map.update(nodes_map_3)
    edges_map.update(edges_map_3)
    nodes_map_2, edges_map_2 = do_doubly_pinned_query(neighbors_1, set(nodes_map_3))
    nodes_map.update(nodes_map_2)
    edges_map.update(edges_map_2)
    nodes_map_1, edges_map_1 = do_doubly_pinned_query(genes, set(nodes_map_2))
    nodes_map.update(nodes_map_1)
    edges_map.update(edges_map_1)

    print(f"Saving data..")
    with jsonlines.open(f"{SCRIPT_DIR}/nodes.jsonl", mode='w') as nodes_writer:
        nodes_writer.write_all(list(nodes_map.values()))
    with jsonlines.open(f"{SCRIPT_DIR}/edges.jsonl", mode='w') as edges_writer:
        edges_writer.write_all(list(edges_map.values()))


if __name__ == "__main__":
    main()
