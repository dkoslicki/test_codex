"""
This script largely came from David Koslicki's original toy implementation; it's been adjusted to load a graph
from a specified directory, which must contain nodes.jsonl, edges.jsonl, gene_to_diseases.json, and
disease_symptom_frequencies.json files.
Usage: python toy_implementation_jsonlines.py <path_to_graph_dir e.g., graphs/toy_graph_1>
"""
import argparse
import json
import time
from collections import defaultdict

import jsonlines
import numpy as np
from scipy.special import expit  # Logistic function
from scipy.optimize import minimize
import random
random.seed(42)

# ----------------------------
# 1. Load the toy data
# ----------------------------

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("graph_dir",
                        help="Path to the directory containing files for the KG you want to use. (Must include "
                             "nodes.jsonl, edges.jsonl, gene_to_diseases.json, and "
                             "disease_symptom_frequencies.json.)")
args = arg_parser.parse_args()

# Load the knowledge graph
with jsonlines.open(f"{args.graph_dir}/nodes.jsonl") as reader:
    nodes_dict = {row["id"]: row for row in reader}
with jsonlines.open(f"{args.graph_dir}/edges.jsonl") as reader:
    edges_dict = {row["id"]: row for row in reader}

# Load the Orphanet info
with open(f"{args.graph_dir}/gene_to_diseases.json", "r") as gene_to_diseases_file:
    gene_to_diseases_raw = json.load(gene_to_diseases_file)
gene_to_diseases = {gene_id: set(disease_ids) for gene_id, disease_ids in gene_to_diseases_raw.items()}
with open(f"{args.graph_dir}/disease_symptom_frequencies.json", "r") as disease_symp_file:
    disease_symptom_frequencies_raw = json.load(disease_symp_file)

disease_symptom_frequencies = dict()
for disease, symptoms_freq_dict in disease_symptom_frequencies_raw.items():
    for symptom, frequency in symptoms_freq_dict.items():
        # TODO: Extend frequencies to descendant symptoms!?
        disease_symptom_frequencies[(disease, symptom)] = frequency


genes = list(gene_to_diseases)
diseases = list(disease_symptom_frequencies)
symptoms = list({symptom_id for disease, symptoms in disease_symptom_frequencies_raw.items()
                 for symptom_id in symptoms})
print(genes, diseases, symptoms)

# Frequency categories and their midpoints
frequency_midpoints = {
    'obligate': 1.00,
    'very_frequent': 0.90,
    'frequent': 0.55,
    'occasional': 0.17,
    'very_rare': 0.02
}

# Build X: set of (disease, symptom, frequency_category)
disease_symptom_data = set()
for (disease, symptom), frequency in disease_symptom_frequencies.items():
    disease_symptom_data.add((disease, symptom, frequency))

# ----------------------------
# 2. Implement Functions and Mappings
# ----------------------------

# Map frequency categories to midpoint fractions
# Already defined as frequency_midpoints

# Function to get frequency categories for a gene-symptom pair
def get_gene_symptom_frequencies(gene, symptom):
    frequencies = set()
    for disease in gene_to_diseases[gene]:
        if (disease, symptom) in disease_symptom_frequencies:
            frequencies.add(disease_symptom_frequencies[(disease, symptom)])
    return frequencies

# Build gene_symptom_pairs: set of (gene, symptom) pairs
gene_symptom_pairs = set()
for gene in genes:
    for disease in gene_to_diseases[gene]:
        for symptom in symptoms:
            if (disease, symptom) in disease_symptom_frequencies:
                gene_symptom_pairs.add((gene, symptom))

# Build shuffled_gene_symptom_pairs: shuffled (gene, symptom) pairs
shuffled_gene_symptom_pairs = set()
symptoms_shuffled = symptoms.copy()
random.shuffle(symptoms_shuffled)
for (gene, symptom), symptom_shuffled in zip(gene_symptom_pairs, symptoms_shuffled):
    if (gene, symptom_shuffled) not in gene_symptom_pairs:
        shuffled_gene_symptom_pairs.add((gene, symptom_shuffled))

# Ensure shuffled_gene_symptom_pairs is disjoint from gene_symptom_pairs and same size
shuffled_gene_symptom_pairs = set(list(shuffled_gene_symptom_pairs)[:len(gene_symptom_pairs)])

# Indicator function: checks if a symptom is associated with a gene
def is_symptom_associated_with_gene(gene, symptom):
    return 1 if (gene, symptom) in gene_symptom_pairs else 0

# Calculate average frequency midpoint for use in maximum_symptom_frequency
average_frequency_midpoint = np.mean(list(frequency_midpoints.values()))

# Function to get the maximum symptom frequency for a gene-symptom pair
def maximum_symptom_frequency(gene, symptom):
    if (gene, symptom) in gene_symptom_pairs:
        frequencies = get_gene_symptom_frequencies(gene, symptom)
        if frequencies:
            return max([frequency_midpoints[freq] for freq in frequencies])
        else:
            return average_frequency_midpoint  # If no frequency found, use average
    else:
        return average_frequency_midpoint  # For (gene, symptom) not in gene_symptom_pairs

# Function to get the set of symptoms associated with a gene
def get_gene_symptom_set(gene):
    return {symptom for symptom in symptoms if (gene, symptom) in gene_symptom_pairs.union(shuffled_gene_symptom_pairs)}


def gene_weight_normalization(gene):
    return sum([maximum_symptom_frequency(gene, symptom) for symptom in get_gene_symptom_set(gene)])


# Define predicates
predicates = list({edge["predicate"] for edge in edges_dict.values()})
predicate_indices = {predicate: i for i, predicate in enumerate(predicates)}

# ----------------------------
# 4. Model Implementation
# ----------------------------

# Number of nodes
num_nodes = len(nodes_dict)
nodes = list(nodes_dict)
node_indices = {node: i for i, node in enumerate(nodes)}

# Number of predicates
num_predicates = len(predicates)

# Adjacency matrices for each predicate
adjacency_matrices = np.zeros((num_predicates, num_nodes, num_nodes))
for edge in edges_dict.values():
    predicate = edge["predicate"]
    predicate_idx = predicate_indices[predicate]
    source_idx = node_indices[edge["subject"]]
    target_idx = node_indices[edge["object"]]
    adjacency_matrices[predicate_idx, source_idx, target_idx] = 1

# Initial weights for predicates
predicate_weights = np.random.rand(num_predicates)

# Baseline offset
baseline_offset = -10.0  # Initialized to a negative value

# Hyperparameters
l1_regularization = 0  # Corresponds to 'a' in the original code
l2_regularization = 0  # Corresponds to 'b' in the original code
predicate_l2_regularization = 0  # Corresponds to 'c' in the original code
max_path_length = 4  # Maximum path length

# List of all genes with associated symptoms
gene_list = [gene for gene in genes if get_gene_symptom_set(gene)]

# Initialize node weights for each gene
node_weights = {gene: np.random.rand(num_nodes) for gene in gene_list}

# Functions to compute raw scores and predicted probabilities
def compute_raw_score(node_weights_gene, predicate_weights, baseline_offset, symptom_idx, gene_idx):
    diag_node_weights = np.diag(node_weights_gene)
    weighted_adjacency = sum(predicate_weights[p] * adjacency_matrices[p] for p in range(num_predicates))
    M = diag_node_weights @ weighted_adjacency
    total_matrix_power = np.zeros_like(M)
    for l in range(2, max_path_length + 1):
        total_matrix_power += np.linalg.matrix_power(M, l)
    #raw_score = total_matrix_power[symptom_idx, gene_idx] + baseline_offset
    raw_score = total_matrix_power[gene_idx, symptom_idx] + baseline_offset
    return raw_score

def compute_predicted_probability(raw_score):
    return expit(raw_score)  # Logistic function (inverse logit)

# ----------------------------
# 5. Model Training
# ----------------------------

# Build mapping from genes and symptoms to indices
gene_indices = {gene: node_indices[gene] for gene in genes}
symptom_indices = {symptom: node_indices[symptom] for symptom in symptoms}

# Flatten node_weights into a vector for optimization
def flatten_node_weights(node_weights_dict):
    return np.concatenate([node_weights_dict[gene] for gene in gene_list])

def unflatten_node_weights(flat_weights):
    node_weights_dict = {}
    n = num_nodes
    for i, gene in enumerate(gene_list):
        node_weights_dict[gene] = flat_weights[i*n:(i+1)*n]
    return node_weights_dict

# Objective function to minimize
def objective_function(params):
    # Unpack parameters
    num_q = num_nodes * len(gene_list)
    flat_node_weights = params[:num_q]
    predicate_weights = params[num_q:num_q+num_predicates]
    baseline_offset = params[-1]
    node_weights_dict = unflatten_node_weights(flat_node_weights)
    total_loss = 0
    for gene in gene_list:
        node_weights_gene = node_weights_dict[gene]
        gene_idx = gene_indices[gene]
        normalization_factor = gene_weight_normalization(gene)
        sum_loss = 0
        for symptom in get_gene_symptom_set(gene):
            symptom_idx = symptom_indices[symptom]
            raw_score = compute_raw_score(node_weights_gene, predicate_weights, baseline_offset, symptom_idx, gene_idx)
            predicted_probability = compute_predicted_probability(raw_score)
            true_label = is_symptom_associated_with_gene(gene, symptom)
            symptom_frequency = maximum_symptom_frequency(gene, symptom)
            cross_entropy_loss = - (true_label * np.log(predicted_probability + 1e-15) + (1 - true_label) * np.log(1 - predicted_probability + 1e-15))
            sum_loss += symptom_frequency * cross_entropy_loss
        # Regularization terms for node weights
        node_weights_l1 = (l1_regularization / num_nodes) * np.sum(np.abs(node_weights_gene))
        node_weights_l2 = (l2_regularization / num_nodes) * np.sqrt(np.sum(node_weights_gene ** 2))
        total_loss += (sum_loss / normalization_factor) + node_weights_l1 + node_weights_l2
    # Regularization term for predicate weights
    predicate_weights_l2 = (predicate_l2_regularization / num_predicates) * np.sum(predicate_weights ** 2)
    average_loss = (total_loss / len(gene_list)) + predicate_weights_l2
    return average_loss

# Initial parameters
initial_node_weights = flatten_node_weights(node_weights)
initial_params = np.concatenate([initial_node_weights, predicate_weights, [baseline_offset]])

# Enforce parameters are non-negative
# Bounds for node weights (non-negative)
bounds_node_weights = [(0, None) for _ in range(len(initial_node_weights))]
# Bounds for predicate_weights (non-negative)
bounds_predicate_weights = [(0, None) for _ in range(num_predicates)]
# Bounds for baseline_offset (fixed value for simplicity)
bounds_baseline_offset = [(-4, -4)]
# Combine all bounds
bounds = bounds_node_weights + bounds_predicate_weights + bounds_baseline_offset

# Optimization
print(f"Calling minimize... graph has {len(nodes_dict)} nodes and {len(edges_dict)} edges")
start = time.time()
result = minimize(objective_function, initial_params, method='L-BFGS-B', bounds=bounds, options={'disp': True})
print(f"Done running minimize. Took {round((time.time() - start) / 60)} minutes.")

# Extract optimized parameters
optimized_params = result.x
optimized_node_weights_flat = optimized_params[:len(initial_node_weights)]
optimized_predicate_weights = optimized_params[len(initial_node_weights):len(initial_node_weights) + num_predicates]
optimized_baseline_offset = optimized_params[-1]
optimized_node_weights = unflatten_node_weights(optimized_node_weights_flat)

# ----------------------------
# 6. View Intermediate Node Weights
# ----------------------------

def get_top_nodes_for_gene(gene, node_weights_dict, top_k=3):
    node_weights_gene = node_weights_dict[gene]
    # Get indices of top_k nodes
    top_indices = np.argsort(node_weights_gene)[::-1][:top_k]
    top_nodes = [nodes[i] for i in top_indices]
    return top_nodes

# View top nodes for genes
for gene in genes:
    top_nodes = get_top_nodes_for_gene(gene, optimized_node_weights)
    top_node_names = [nodes_dict[top_node_id].get("name", top_node_id) for top_node_id in top_nodes]
    print(f"Top nodes for gene {nodes_dict[gene].get('name', gene)}: {top_node_names}")

# Let's compare how well the true labels match the predicted probabilities for all gene-symptom pairs
for gene in genes:
    symptoms_in_graph = set(nodes).intersection(symptoms)
    for symptom in symptoms_in_graph:
        true_label = is_symptom_associated_with_gene(gene, symptom)
        predicted_probability = compute_predicted_probability(compute_raw_score(optimized_node_weights[gene],
                                                                               optimized_predicate_weights, optimized_baseline_offset, symptom_indices[symptom], gene_indices[gene]))
        print(f"Gene: {nodes_dict[gene].get('name', gene)}, Symptom: {nodes_dict[symptom].get('name', symptom)}, "
              f"True Label: {true_label}, Predicted Probability: {predicted_probability:.4f}")


# ----------------------------
# 7. Make Predictions
# ----------------------------

# Function to re-optimize node weights for prediction
def predict_node_weights(gene, optimized_predicate_weights, optimized_baseline_offset):
    # Initialize node weights for the gene with random values
    initial_node_weights_gene = np.random.rand(num_nodes)
    # Bounds for node weights (non-negative)
    bounds_node_weights_gene = [(0, None) for _ in range(num_nodes)]

    # Symptoms associated with the gene
    associated_symptoms = {symptom for symptom in symptoms if (gene, symptom) in gene_symptom_pairs}

    # Indices mapping
    gene_idx = gene_indices[gene]
    symptom_indices_list = [symptom_indices[symptom] for symptom in associated_symptoms]

    # Normalization factor
    normalization_factor = sum([maximum_symptom_frequency(gene, symptom) for symptom in associated_symptoms])

    # Objective function to minimize for node weights
    def objective_function_node_weights(node_weights_gene):
        loss = 0
        for symptom_idx in symptom_indices_list:
            raw_score = compute_raw_score(node_weights_gene, optimized_predicate_weights, optimized_baseline_offset, symptom_idx, gene_idx)
            predicted_probability = compute_predicted_probability(raw_score)
            true_label = 1  # Since we are setting y_{g,s} = 1 for prediction
            symptom_frequency = maximum_symptom_frequency(gene, nodes[symptom_idx])
            cross_entropy_loss = - np.log(predicted_probability + 1e-15)  # Since true_label = 1
            loss += symptom_frequency * cross_entropy_loss
        # Regularization terms for node weights
        node_weights_l1 = (l1_regularization / num_nodes) * np.sum(np.abs(node_weights_gene))
        node_weights_l2 = (l2_regularization / num_nodes) * np.sqrt(np.sum(node_weights_gene ** 2))
        total_loss = (loss / normalization_factor) + node_weights_l1 + node_weights_l2
        return total_loss

    # Optimize node weights with bounds
    result_node_weights = minimize(objective_function_node_weights, initial_node_weights_gene, method='L-BFGS-B', bounds=bounds_node_weights_gene)
    optimized_node_weights_gene = result_node_weights.x
    return optimized_node_weights_gene

# Function to predict intermediate nodes for a gene
def predict_intermediate_nodes(gene, optimized_predicate_weights, optimized_baseline_offset, top_k=6):
    optimized_node_weights_gene = predict_node_weights(gene, optimized_predicate_weights, optimized_baseline_offset)
    # Get indices of top_k nodes
    top_indices = np.argsort(optimized_node_weights_gene)[-top_k:][::-1]
    top_nodes = [nodes[i] for i in top_indices]
    return top_nodes


# View predictions for genes
for gene in genes:
    predicted_nodes = predict_intermediate_nodes(gene, optimized_predicate_weights, optimized_baseline_offset)
    predicted_node_names = [nodes_dict[node_id].get('name', node_id) for node_id in predicted_nodes]
    print(f"Predicted intermediate nodes for gene {nodes_dict[gene].get('name', gene)}: {predicted_node_names}")
