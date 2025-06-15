import copy
import json
import os
from collections import defaultdict
import random

import jsonlines
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

random.seed(22)


def serialize_with_sets(obj: any) -> any:
    return list(obj) if isinstance(obj, set) else obj


# Find all gene -- symptom pairs, and load other lookup maps
with open(f"{SCRIPT_DIR}/../../orphanet/orphanet_disease_to_symptoms.json") as symptoms_file:
    disease_to_symptom_frequencies = json.load(symptoms_file)
with open(f"{SCRIPT_DIR}/../../orphanet/orphanet_disease_to_genes.json") as genes_file:
    disease_to_genes = json.load(genes_file)
with open(f"{SCRIPT_DIR}/../../orphanet/orphanet_gene_to_diseases.json") as gene_to_diseases_file:
    gene_to_diseases = json.load(gene_to_diseases_file)
with open(f"{SCRIPT_DIR}/../../orphanet/equivalent_genes.json") as gene_equivalencies_file:
    gene_equivalencies = json.load(gene_equivalencies_file)
gene_symptom_pairs = set()
for disease, symptom_frequencies in disease_to_symptom_frequencies.items():
    associated_genes = disease_to_genes.get(disease, [])
    for symptom_id in symptom_frequencies:
        for gene_id in associated_genes:
            gene_symptom_pairs.add((gene_id, symptom_id))
print(f"Have {len(gene_symptom_pairs)} gene--symptom pairs!")

# TODO: Convert gene/symptom identifiers to KG2/ARAX-preferred IDs? (i.e., send to synonymizer API?)

allowed_categories = ["biolink:Protein", "biolink:Pathway", "biolink:BiologicalProcess", "biolink:PhysiologicalProcess"]
# Get neighbors of all genes
query = {"node_ids": list(gene_to_diseases), "categories": allowed_categories}
response = requests.post(url="https://kg2cploverdb.ci.transltr.io/get_neighbors",
                         json=query,
                         headers={'accept': 'application/json'})
gene_to_neighbors = {gene_id: set(neighbors) for gene_id, neighbors in response.json().items() if neighbors}
print(f"{len(gene_to_neighbors)} of {len(gene_to_diseases)} genes have neighbors of specified categories")

# Get neighbors of all symptoms
all_symptoms = {symptom_id for disease, symptom_frequencies in disease_to_symptom_frequencies.items()
                for symptom_id in symptom_frequencies}
query = {"node_ids": list(all_symptoms), "categories": allowed_categories}
response = requests.post(url="https://kg2cploverdb.ci.transltr.io/get_neighbors",
                         json=query,
                         headers={'accept': 'application/json'})
symptom_to_neighbors = {symptom_id: set(neighbors) for symptom_id, neighbors in response.json().items() if neighbors}
print(f"{len(symptom_to_neighbors)} of {len(all_symptoms)} symptoms have neighbors of specified categories")

# Calculate overlap for each of the gene--symptom pairs
connected_gene_symptom_pairs = dict()
for gene_id, symptom_id in gene_symptom_pairs:
    gene_neighbors = gene_to_neighbors.get(gene_id, set())
    symptom_neighbors = symptom_to_neighbors.get(symptom_id, set())
    overlap = gene_neighbors.intersection(symptom_neighbors)
    if overlap:
        connected_gene_symptom_pairs[(gene_id, symptom_id)] = overlap
print(f"{len(connected_gene_symptom_pairs)} of {len(gene_symptom_pairs)} gene--symptom pairs are connected "
      f"(using this two-hop method)")

# Map all intermediate nodes to their related genes and symptoms (for quick lookup)
intermediate_nodes_to_genes = defaultdict(set)
intermediate_nodes_to_symptoms = defaultdict(set)
for (gene_id, symptom_id), intermediate_nodes in connected_gene_symptom_pairs.items():
    for intermediate_node in intermediate_nodes:
        intermediate_nodes_to_genes[intermediate_node].add(gene_id)
        intermediate_nodes_to_symptoms[intermediate_node].add(symptom_id)

# Build the graph (in terms of node IDs)
iteration_results = dict()
best_chosen_pairs = None
best_intermediate_nodes = None
best_ratio = 0
for iteration_num in range(20):  # Pick the best graph out of these
    start_pair = random.choice(list(connected_gene_symptom_pairs))
    chosen_pairs = {start_pair}
    intermediate_nodes = copy.deepcopy(connected_gene_symptom_pairs[start_pair])
    print(f"Start pair is: {start_pair}, with {len(intermediate_nodes)} intermediate nodes")
    while len(intermediate_nodes) < 50:
        num_intermediate_nodes_start = len(intermediate_nodes)

        # Look up other connected subgraphs
        connected_genes = {gene_id for intermediate_node in intermediate_nodes
                           for gene_id in intermediate_nodes_to_genes[intermediate_node]}
        connected_symptoms = {symptom_id for intermediate_node in intermediate_nodes
                              for symptom_id in intermediate_nodes_to_symptoms[intermediate_node]}
        # See if any pairs exist between these
        for connected_gene in connected_genes:
            for connected_symptom in connected_symptoms:
                new_pair = (connected_gene, connected_symptom)
                if new_pair in connected_gene_symptom_pairs and len(intermediate_nodes) < 50:
                    chosen_pairs.add(new_pair)
                    intermediate_nodes = intermediate_nodes.union(connected_gene_symptom_pairs[new_pair])

        # Otherwise just add another random pair
        if num_intermediate_nodes_start == len(intermediate_nodes):
            print(f"Couldn't find connected pairs; randomly selecting one to add")
            random_pair = random.choice(list(connected_gene_symptom_pairs))
            chosen_pairs.add(random_pair)
            intermediate_nodes = intermediate_nodes.union(connected_gene_symptom_pairs[random_pair])

        print(f"Have chosen {len(chosen_pairs)} pairs, with {len(intermediate_nodes)} intermediate nodes")
    ratio = len(chosen_pairs) / len(intermediate_nodes)
    print(f"At end of iteration, ratio of pairs to intermediate nodes is: {ratio}")
    if ratio > best_ratio:
        best_chosen_pairs = chosen_pairs
        best_intermediate_nodes = intermediate_nodes
        best_ratio = ratio

print(f"The winning graph had ratio {best_ratio} ({len(best_chosen_pairs)} pairs, "
      f"{len(best_intermediate_nodes)} intermediate nodes)")

# Then get the actual edges/knowledge graph
chosen_gene_ids = {pair[0] for pair in best_chosen_pairs}
chosen_symptom_ids = {pair[1] for pair in best_chosen_pairs}
connected_node_pairs = set()
for intermediate_node in best_intermediate_nodes:
    connected_genes = intermediate_nodes_to_genes[intermediate_node]
    connected_symptoms = intermediate_nodes_to_symptoms[intermediate_node]
    retained_connected_genes = connected_genes.intersection(chosen_gene_ids)
    retained_connected_symptoms = connected_symptoms.intersection(chosen_symptom_ids)
    for retained_connected_gene in retained_connected_genes:
        connected_node_pairs.add((retained_connected_gene, intermediate_node))
    for retained_connected_symptom in retained_connected_symptoms:
        connected_node_pairs.add((retained_connected_symptom, intermediate_node))

print(f"Querying for edges between {len(connected_node_pairs)} final pairs of nodes")
response = requests.post(url=f"https://kg2cploverdb.ci.transltr.io/get_edges",
                         json={"pairs": list(connected_node_pairs)},
                         headers={'accept': 'application/json'})
kg = response.json()["knowledge_graph"]
print(f"Returned KG has {len(kg['edges'])} edges, {len(kg['nodes'])} nodes")

with open(f"{SCRIPT_DIR}/sample_graph.json", "w+") as output_file:
    json.dump(kg, output_file, indent=2)

# Convert graph to preferred identifiers (HGNC, HP)
gene_preferred_id_map = dict()
for preferred_gene_id, equiv_gene_ids in gene_equivalencies.items():
    for equiv_gene_id in equiv_gene_ids:
        gene_preferred_id_map[equiv_gene_id] = preferred_gene_id
query_body = {
    "curies": list(kg["nodes"]),
    "conflate": True,
    "drug_chemical_conflate": True,
    "individual_types": True
}
response = requests.post(url="https://nodenormalization-sri.renci.org/get_normalized_nodes",
                         json=query_body)
preferred_id_map = dict()
for input_curie, node_info in response.json().items():
    preferred_id = input_curie
    if input_curie in gene_preferred_id_map:
        preferred_id = gene_preferred_id_map[input_curie]
    else:
        if node_info:
            equivalent_ids = {item["identifier"] for item in node_info["equivalent_identifiers"]}
            for equiv_id in equivalent_ids:
                if equiv_id.startswith("HGNC:") or equiv_id.startswith("HP:"):
                    preferred_id = equiv_id
                elif equiv_id.startswith("OMIM:"):
                    preferred_id = equiv_id
        else:
            print(f"Input curie {input_curie} didn't return any normalized info")
        # Final attempt to convert to proper gene identifier.. (may have converted to OMIM, which now we can convert..)
        if not preferred_id.startswith("HGNC:") and not preferred_id.startswith("HP:"):
            preferred_id = gene_preferred_id_map.get(preferred_id, preferred_id)

    preferred_id_map[input_curie] = preferred_id

# Convert to a minimal jsonlines format for the graph (nodes and edges files), like model expects
jsonl_edges = [{"id": edge_key,
                "subject": preferred_id_map.get(edge["subject"], edge["subject"]),
                "object": preferred_id_map.get(edge["object"], edge["object"]),
                "predicate": edge["predicate"]}
               for edge_key, edge in kg["edges"].items()]
jsonl_nodes = [{"id": preferred_id_map.get(node_key, node_key), "name": node["name"], "categories": node["categories"]}
               for node_key, node in kg["nodes"].items()]
print(f"Saving {len(jsonl_edges)} jsonl edges, {len(jsonl_nodes)} jsonl nodes")
with jsonlines.open(f"{SCRIPT_DIR}/nodes.jsonl", mode="w") as writer:
    writer.write_all(jsonl_nodes)
with jsonlines.open(f"{SCRIPT_DIR}/edges.jsonl", mode="w") as writer:
    writer.write_all(jsonl_edges)

# Reverse map so we have symptom --> diseases
symptom_to_diseases = defaultdict(set)
for disease, symptom_frequencies in disease_to_symptom_frequencies.items():
    for symptom_id in symptom_frequencies:
        symptom_to_diseases[symptom_id].add(disease)
# Then filter down our symptom / gene labels, and save them
disease_to_symptom_frequencies_filtered = defaultdict(lambda: dict())
gene_to_diseases_filtered = defaultdict(set)
for chosen_gene_id, chosen_symptom_id in best_chosen_pairs:
    gene_diseases = set(gene_to_diseases[chosen_gene_id])
    symptom_diseases = symptom_to_diseases[chosen_symptom_id]
    shared_diseases = gene_diseases.intersection(symptom_diseases)
    gene_to_diseases_filtered[chosen_gene_id] = gene_to_diseases_filtered[chosen_gene_id].union(shared_diseases)
    for disease_id in shared_diseases:
        symptom_frequency = disease_to_symptom_frequencies[disease_id][chosen_symptom_id]
        disease_to_symptom_frequencies_filtered[disease_id][chosen_symptom_id] = symptom_frequency
with open(f"{SCRIPT_DIR}/disease_symptom_frequencies.json", "w+") as output_file:
    json.dump(disease_to_symptom_frequencies_filtered, output_file, indent=2, default=serialize_with_sets)
with open(f"{SCRIPT_DIR}/gene_to_diseases.json", "w+") as output_file:
    json.dump(gene_to_diseases_filtered, output_file, indent=2, default=serialize_with_sets)




