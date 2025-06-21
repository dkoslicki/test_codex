"""
Usage: python pytorch_implementation.py <path to graph directory - e.g., graphs/toy_graph_2>
"""

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Tuple, Set, List, Dict

import jsonlines
import torch
import torchmetrics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# These are the labels used in the Oprhanet data download (NOTE: slightly different than those in the Orphanet UI)
SYMPTOM_FREQUENCY_MIDPOINTS = {
    'obligate': 1.00,
    'very_frequent': 0.90,
    'frequent': 0.55,
    'occasional': 0.17,
    'very_rare': 0.02
}
AVERAGE_FREQUENCY_MIDPOINT = torch.tensor(list(SYMPTOM_FREQUENCY_MIDPOINTS.values())).mean()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(f"{SCRIPT_DIR}/log_raretarget.txt")])

random.seed(22)


class GroundtruthHelper:

    def __init__(self, gene_to_diseases: Dict[str, Set[str]],
                 disease_to_symptom_frequencies: Dict[str, Dict[str, float]],
                 gene_to_symptoms: Dict[str, Set[str]],
                 gene_symptom_pairs: Set[Tuple[str, str]],
                 gene_to_symptom_frequency_sets: Dict[str, Dict[str, Set[float]]],
                 gene_to_max_symptom_frequencies: Dict[str, Dict[str, float]],
                 genes: Set[str], diseases: Set[str], symptoms: Set[str]):
        self.gene_to_diseases = gene_to_diseases
        self.disease_to_symptom_frequencies = disease_to_symptom_frequencies
        self.gene_to_symptoms = gene_to_symptoms
        self.gene_symptom_pairs = gene_symptom_pairs
        self.gene_to_symptom_frequency_sets = gene_to_symptom_frequency_sets
        self.gene_to_max_symptom_frequencies = gene_to_max_symptom_frequencies
        self.genes = genes
        self.diseases = diseases
        self.symptoms = symptoms
        self.genes_list = list(self.genes)
        self.symptoms_list = list(self.symptoms)
        self.gene_indices_map = {gene: index for index, gene in enumerate(self.genes_list)}

        logging.debug(f"Groundtruth data contains {len(self.genes)} genes, {len(self.diseases)} diseases, "
                      f"and {len(self.symptoms)} symptoms.")
        logging.debug(f"Groundtruth data contains {len(self.gene_symptom_pairs)} gene--symptom pairs.")

    def __str__(self):
        return (f"GroundtruthHelper(gene_to_diseases={self.gene_to_diseases}, "
                f"gene_to_symptom_frequency_sets={self.gene_to_symptom_frequency_sets},"
                f"gene_to_max_symptom_frequencies={self.gene_to_max_symptom_frequencies})")

    def gene_and_symptom_are_associated(self, gene, symptom) -> int:
        return 1 if (gene, symptom) in self.gene_symptom_pairs else 0


class GraphHelper:

    def __init__(self, nodes_map: Dict[str, any], edges_map: Dict[str, any], device: torch.device):
        self.nodes_map = nodes_map
        self.edges_map = edges_map

        # Create some useful data structures
        self.predicates_list = list({edge["predicate"] for edge in self.edges_map.values()})
        self.num_predicates = len(self.predicates_list)
        self.predicate_indices_map = {predicate: index for index, predicate in enumerate(self.predicates_list)}

        self.nodes_list = list(self.nodes_map.keys())
        self.num_nodes = len(self.nodes_list)
        self.node_indices_map = {node_key: index for index, node_key in enumerate(self.nodes_list)}

        self.adjacency_tensor, self.adjacency_list = self.create_adjacency_maps(self.edges_map, device)

        logging.debug(f"Graph contains {self.num_nodes} nodes and {len(self.edges_map)} edges.")
        logging.debug(f"Graph contains {len(self.predicates_list)} distinct predicates: {self.predicates_list} ")

    def create_adjacency_maps(self, edges_map: dict, device: torch.device) -> Tuple[torch.tensor, dict]:
        # Create an adjacency matrix for each predicate
        # TODO: For symmetric predicates, record edges in both directions?
        adjacency_list = defaultdict(set)
        predicate_adj_tensor = torch.zeros(self.num_predicates, self.num_nodes, self.num_nodes, device=device)
        for edge in self.edges_map.values():
            predicate = edge["predicate"]
            predicate_index = self.predicate_indices_map[predicate]
            subject_index = self.node_indices_map[edge["subject"]]
            object_index = self.node_indices_map[edge["object"]]
            predicate_adj_tensor[predicate_index, subject_index, object_index] = 1
            adjacency_list[edge["subject"]].add(edge["object"])
        assert len(predicate_adj_tensor) == self.num_predicates
        triples = {(edge["subject"], edge["predicate"], edge["object"]) for edge in edges_map.values()}
        assert torch.sum(predicate_adj_tensor) == len(triples)
        return predicate_adj_tensor, adjacency_list

    def get_all_intermediate_nodes(self, start_node: str, target_node: str) -> Set[str]:
        intermediate_nodes_map = defaultdict(set)
        return self.get_all_intermediate_nodes_recursive(start_node, target_node,
                                                         intermediate_nodes_map).difference({start_node, target_node})

    def get_all_intermediate_nodes_recursive(self, current_node: str, target_node: str,
                                             intermediate_nodes_map: Dict[str, Set[str]]) -> Set[str]:
        if current_node not in intermediate_nodes_map:
            # Initialize entry for this node
            neighbors = self.adjacency_list.get(current_node, set())
            if target_node in neighbors:
                intermediate_nodes_map[current_node].add(current_node)
            else:
                intermediate_nodes_map[current_node] = set()

            # Find neighbor solutions to update this node's solution with
            neighbor_solutions = [self.get_all_intermediate_nodes_recursive(neighbor, target_node,
                                                                            intermediate_nodes_map)
                                  for neighbor in neighbors]
            neighbor_solutions_union = set().union(*neighbor_solutions)
            if neighbor_solutions_union:
                intermediate_nodes_map[current_node].add(current_node)
                intermediate_nodes_map[current_node] |= neighbor_solutions_union

        return intermediate_nodes_map[current_node]

    def check_if_nodes_are_connected(self, start_node: str, target_node: str, num_hops: int) -> bool:
        encountered_nodes = {start_node}
        counter = 0
        while target_node not in encountered_nodes and counter < num_hops:
            neighbors = [self.adjacency_list[node] for node in encountered_nodes]
            encountered_nodes |= set().union(*neighbors)
            if target_node in encountered_nodes:
                print(f"\nNode {self.get_node_name(start_node)} IS connected to "
                      f"{self.get_node_name(target_node)} within {num_hops} hops!")
            counter += 1
        if target_node not in encountered_nodes:
            print(f"\nNO CONNECTION between {self.get_node_name(start_node)} and {self.get_node_name(target_node)} "
                  f"within {num_hops} hops")
        return target_node in encountered_nodes

    def get_node_name(self, node_id) -> str:
        return self.nodes_map.get(node_id, {}).get("name", node_id)


class GeneSymptomClassifier:

    def __init__(self, graph_dir: str, device: torch.device):
        self.graph, self.groundtruth = load_data(graph_dir, device)

        self.num_flat_weights = self.graph.num_nodes * len(self.groundtruth.genes_list)

        # Generate negative training examples
        self.shuffled_gene_symptom_pairs = self.generate_shuffled_gene_symptom_pairs()
        self.shuffled_gene_to_symptoms = defaultdict(set)
        for shuffled_gene, shuffled_symptom in self.shuffled_gene_symptom_pairs:
            self.shuffled_gene_to_symptoms[shuffled_gene].add(shuffled_symptom)

        # Properties that will be learned (initiated to random)
        self.node_weights_tensor = torch.tensor([torch.rand(self.graph.num_nodes).tolist()
                                                for gene in self.groundtruth.genes_list],
                                                requires_grad=True,
                                                device=device)
        self.predicate_weights = torch.rand(self.graph.num_predicates, requires_grad=True, device=device)
        self.baseline_offset = torch.tensor(-10.0, requires_grad=True, device=device)

        # Hyperparameters:
        self.L1_REGULARIZATION = 0.00000001  # Corresponds to 'a' in equations
        self.L2_REGULARIZATION = 0.00000001  # Corresponds to 'b' in equations
        self.PREDICATE_L2_REGULARIZATION = 0.0000001  # Corresponds to 'c' in equations
        self.MAX_PATH_LENGTH = 4
        self.LEARNING_RATE = 0.1
        self.MIN_DELTA = 1e-4
        self.STABLE_ROUNDS_REQUIRED = 20

        # Pre-compute some handy index tensors for batched operations
        self.symptom_indices_map = {
            symptom: idx for idx, symptom in enumerate(self.groundtruth.symptoms_list)
        }
        self.gene_node_indices = torch.tensor(
            [self.graph.node_indices_map[g] for g in self.groundtruth.genes_list],
            device=device,
        )
        self.symptom_node_indices = torch.tensor(
            [self.graph.node_indices_map.get(s, 0) for s in self.groundtruth.symptoms_list],
            device=device,
        )
        self.symptom_valid_mask = torch.tensor(
            [s in self.graph.node_indices_map for s in self.groundtruth.symptoms_list],
            dtype=torch.bool,
            device=device,
        )

        # Prepare label and frequency matrices used during training
        num_genes = len(self.groundtruth.genes_list)
        num_symptoms = len(self.groundtruth.symptoms_list)
        self.label_matrix = torch.zeros(num_genes, num_symptoms, device=device)
        self.frequency_matrix = torch.zeros(num_genes, num_symptoms, device=device)
        for g_idx, gene in enumerate(self.groundtruth.genes_list):
            true_set = self.groundtruth.gene_to_symptoms.get(gene, set())
            neg_set = self.shuffled_gene_to_symptoms.get(gene, set())
            for symptom in true_set.union(neg_set):
                s_idx = self.symptom_indices_map[symptom]
                self.label_matrix[g_idx, s_idx] = (
                    1 if symptom in true_set else 0
                )
                freq = self.groundtruth.gene_to_max_symptom_frequencies.get(gene, {}).get(
                    symptom,
                    AVERAGE_FREQUENCY_MIDPOINT.item(),
                )
                self.frequency_matrix[g_idx, s_idx] = freq

        self.sum_max_frequencies_vec = torch.tensor(
            [self.sum_max_frequencies(g) for g in self.groundtruth.genes_list],
            device=device,
        )

    def train_model(self):
        logging.info(f"TRAINING MODEL..")

        # NOTE: in initial implementation, node weights dict only included genes that had associated symptoms?...
        logging.debug(f"Predicate weights tensor is: {self.predicate_weights}")
        logging.debug(f"Baseline offset is: {self.baseline_offset}")
        logging.debug(f"Shape of node weights tensor is: {self.node_weights_tensor.shape}")

        # TODO: we don't define bounds like initial scipy implementation... that ok?
        logging.info(f"Starting first optimization..")
        start = time.time()
        params = [self.node_weights_tensor, self.predicate_weights, self.baseline_offset]
        optimizer = torch.optim.Adam(params, lr=self.LEARNING_RATE)
        best_loss = float("inf")
        stable_rounds = 0
        for iteration_num in range(1000):
            iter_start = time.time()
            optimizer.zero_grad()
            fwd_start = time.time()
            loss = self.objective_function(params)
            fwd_end = time.time()
            loss.backward()
            bwd_end = time.time()
            optimizer.step()
            step_end = time.time()

            if loss < best_loss - self.MIN_DELTA:
                best_loss = loss
            else:
                stable_rounds += 1
            if iteration_num % 100 == 0:
                logging.info(
                    f"Iter {iteration_num} - loss: {loss:.4f} | fwd: {fwd_end - fwd_start:.2f}s "
                    f"bwd: {bwd_end - fwd_end:.2f}s step: {step_end - bwd_end:.2f}s "
                    f"total: {step_end - iter_start:.2f}s"
                )
            if stable_rounds >= self.STABLE_ROUNDS_REQUIRED:
                logging.info(f"Reached stability after {iteration_num} iterations. Loss is {loss:.4f}")
                break
        logging.info(f"Done with first optimization. Took {round((time.time() - start) / 60)} minutes.")

        logging.info(f"Predicate weights tensor is now: {self.predicate_weights}")
        logging.info(f"Baseline offset tensor is: {self.baseline_offset}")
        self.show_predicted_labels()
        first_metrics = self.evaluate_classification(threshold=0.5, log_output=False)
        self.show_top_intermediate_nodes()

        logging.info(
            "Starting second optimization (focused on gene-specific node weights)"
        )
        start = time.time()

        # Freeze predicate weights and baseline offset learned in the first step
        self.predicate_weights.requires_grad = False
        self.baseline_offset.requires_grad = False
        self.node_weights_tensor.requires_grad = False
        for gene_index, gene in enumerate(self.groundtruth.genes_list):
            gene_node_weights = self.node_weights_tensor[gene_index].detach().clone()
            gene_node_weights.requires_grad = True
            params = [gene_node_weights]
            optimizer_2 = torch.optim.Adam(params, lr=self.LEARNING_RATE)
            best_loss = float("inf")
            stable_rounds = 0
            for iteration_num in range(1000):
                optimizer_2.zero_grad()
                loss = self.objective_function_gene(params, gene=gene)
                loss.backward()
                optimizer_2.step()

                if loss < best_loss - self.MIN_DELTA:
                    best_loss = loss
                else:
                    stable_rounds += 1

                if iteration_num % 100 == 0:
                    logging.debug(
                        f"On iteration {iteration_num} for {gene} - loss is: {loss:.4f}"
                    )
                if stable_rounds >= self.STABLE_ROUNDS_REQUIRED:
                    break

            # Update the main tensor with learned values for this gene
            with torch.no_grad():
                self.node_weights_tensor[gene_index].copy_(gene_node_weights.detach())

        logging.info(
            f"Done with second optimization. Took {round((time.time() - start) / 60)} minutes."
        )
        logging.info(f"Node weights tensor is: {self.node_weights_tensor}")

        logging.info(f"Predicate weights tensor is now: {self.predicate_weights}")
        logging.info(f"Baseline offset tensor is: {self.baseline_offset}")
        self.show_predicted_labels()

        second_metrics = self.evaluate_classification(threshold=0.5, log_output=False)
        self.summarize_metrics(first_metrics, second_metrics)

    def objective_function(self, params):
        node_weights_tensor, predicate_weights, baseline_offset = params

        predicted_probs = self.compute_predicted_probability_matrix(
            node_weights_tensor=node_weights_tensor,
            predicate_weights=predicate_weights,
            baseline_offset=baseline_offset,
        )

        first_term = self.label_matrix * torch.log(predicted_probs + 1e-15)
        second_term = (1 - self.label_matrix) * torch.log(1 - predicted_probs + 1e-15)
        cross_entropy = -(first_term + second_term)
        weighted = self.frequency_matrix * cross_entropy
        sum_loss_per_gene = weighted.sum(dim=1) / self.sum_max_frequencies_vec

        l1_penalty = (
            self.L1_REGULARIZATION * torch.sum(torch.abs(node_weights_tensor), dim=1)
        ) / self.graph.num_nodes
        l2_penalty = (
            self.L2_REGULARIZATION * torch.sqrt(torch.sum(node_weights_tensor ** 2, dim=1))
        ) / self.graph.num_nodes

        total_loss = sum_loss_per_gene + l1_penalty + l2_penalty

        l2_penalty_predicates = (
            self.PREDICATE_L2_REGULARIZATION * torch.sum(predicate_weights ** 2)
        ) / self.graph.num_predicates

        average_loss = total_loss.mean() + l2_penalty_predicates
        return average_loss

    def objective_function_gene(self, params, gene: str):
        # Thank you to David for the core of this function
        # TODO: in initial implementation, node weights were initialized to random here... should we do that?
        gene_node_weights = params[0]
        sum_loss = 0
        true_symptom_set = self.groundtruth.gene_to_symptoms.get(gene, set())
        negative_symptom_set = self.shuffled_gene_to_symptoms.get(gene, set())
        for symptom in true_symptom_set.union(negative_symptom_set):
            predicted_probability = self.compute_predicted_probability(
                gene, symptom, gene_node_weights_override=gene_node_weights
            )
            true_label = self.groundtruth.gene_and_symptom_are_associated(gene, symptom)
            symptom_frequency = self.get_frequency(gene, symptom)
            first_term = true_label * torch.log(predicted_probability + 1e-15)
            second_term = (1 - true_label) * torch.log(1 - predicted_probability + 1e-15)
            cross_entropy_loss = -(first_term + second_term)
            sum_loss += symptom_frequency * cross_entropy_loss
        # Apply node weight regularization penalties
        l1_penalty = (self.L1_REGULARIZATION * torch.sum(torch.abs(gene_node_weights))) / self.graph.num_nodes
        l2_penalty = (self.L2_REGULARIZATION * torch.sqrt(torch.sum(gene_node_weights ** 2))) / self.graph.num_nodes
        total_loss = (sum_loss / self.sum_max_frequencies(gene)) + l1_penalty + l2_penalty
        return total_loss

    def compute_predicted_probability_matrix(
        self,
        node_weights_tensor: torch.Tensor | None = None,
        predicate_weights: torch.Tensor | None = None,
        baseline_offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return predicted probabilities for all genes and symptoms."""
        start = time.time()
        if node_weights_tensor is None:
            node_weights_tensor = self.node_weights_tensor
        if predicate_weights is None:
            predicate_weights = self.predicate_weights
        if baseline_offset is None:
            baseline_offset = self.baseline_offset

        num_genes = node_weights_tensor.shape[0]
        diag_node_weights = torch.diag_embed(node_weights_tensor)
        weighted_adjacency = torch.sum(
            predicate_weights.view(-1, 1, 1) * self.graph.adjacency_tensor, dim=0
        )
        M = torch.bmm(diag_node_weights, weighted_adjacency.unsqueeze(0).expand(num_genes, -1, -1))

        M_power = torch.bmm(M, M)
        matrix_power_sum = M_power.clone()
        for _ in range(3, self.MAX_PATH_LENGTH + 1):
            M_power = torch.bmm(M_power, M)
            matrix_power_sum += M_power

        gene_rows = matrix_power_sum[torch.arange(num_genes), self.gene_node_indices]
        selected = gene_rows[:, self.symptom_node_indices]
        predicted = torch.sigmoid(selected + baseline_offset)
        predicted = predicted * self.symptom_valid_mask.float()
        logging.debug(
            f"Batched probability computation took {time.time() - start:.4f}s"
        )
        return predicted

    def compute_predicted_probability(
        self, gene: str, symptom: str, gene_node_weights_override: torch.Tensor | None = None
    ) -> torch.Tensor:
        if symptom not in self.symptom_indices_map:
            return torch.tensor(0.0, device=self.node_weights_tensor.device)

        if gene_node_weights_override is None:
            node_weights = self.node_weights_tensor
        else:
            node_weights = self.node_weights_tensor.clone()
            gene_idx = self.groundtruth.gene_indices_map[gene]
            node_weights[gene_idx] = gene_node_weights_override

        probs = self.compute_predicted_probability_matrix(node_weights_tensor=node_weights)
        gene_idx = self.groundtruth.gene_indices_map[gene]
        symptom_idx = self.symptom_indices_map[symptom]
        return probs[gene_idx, symptom_idx]

    def get_frequency(self, gene, symptom):
        if gene in self.groundtruth.gene_to_max_symptom_frequencies:
            if symptom in self.groundtruth.gene_to_max_symptom_frequencies[gene]:
                return self.groundtruth.gene_to_max_symptom_frequencies[gene][symptom]
        return AVERAGE_FREQUENCY_MIDPOINT  # Default for negative examples

    def show_predicted_labels(self):
        for gene in self.groundtruth.genes:
            logging.info(f"For gene {self.graph.get_node_name(gene)}:")
            # TODO: Note only looking at trained examples here... later add performance on validation set
            true_symptom_set = self.groundtruth.gene_to_symptoms.get(gene, set())
            negative_symptom_set = self.shuffled_gene_to_symptoms.get(gene, set())
            for symptom in true_symptom_set.union(negative_symptom_set):
                true_label = self.groundtruth.gene_and_symptom_are_associated(gene, symptom)
                predicted_prob = self.compute_predicted_probability(gene, symptom)
                logging.info(f"    true label of {true_label} vs. {predicted_prob:.4f} predicted for symptom "
                             f"'{self.graph.get_node_name(symptom)}'")

    def show_top_intermediate_nodes(self):
        logging.info(f"Extracting top intermediate nodes for each positive gene--symptom pair..")
        # For each positive gene--symptom pair, grab all intermediate nodes
        for gene_index, gene in enumerate(self.groundtruth.genes_list):
            for symptom in self.groundtruth.gene_to_symptoms[gene]:
                intermediate_nodes = self.graph.get_all_intermediate_nodes(gene, symptom)

                # Report those intermediate nodes with the highest weights
                gene_node_weights = self.node_weights_tensor[gene_index]
                intermediate_nodes_with_indices = intermediate_nodes.intersection(self.graph.node_indices_map)
                intermediate_node_weight_map = {node: gene_node_weights[self.graph.node_indices_map[node]].item()
                                                for node in intermediate_nodes_with_indices}
                top_6_int_nodes = sorted(intermediate_node_weight_map.items(), key=lambda x: x[1], reverse=True)[:6]
                if intermediate_node_weight_map:
                    node_weight_str = "\n   ".join([f"{index + 1}. {self.graph.get_node_name(node)} ({weight:.4f}) ({node})"
                                                    for index, (node, weight) in enumerate(top_6_int_nodes)])
                else:
                    node_weight_str = "No intermediate nodes."
                logging.info(f"For gene '{self.graph.get_node_name(gene)}' and symptom "
                             f"'{self.graph.get_node_name(symptom)}', top intermediate nodes are:\n   {node_weight_str}")

    def evaluate_classification(self, threshold, log_output=True):
        true_labels = []
        predicted_probs = []
        for gene in self.groundtruth.genes:
            # TODO: Note only looking at trained examples here... later add performance on validation set
            true_symptom_set = self.groundtruth.gene_to_symptoms.get(gene, set())
            negative_symptom_set = self.shuffled_gene_to_symptoms.get(gene, set())
            for symptom in true_symptom_set.union(negative_symptom_set):
                true_label = self.groundtruth.gene_and_symptom_are_associated(gene, symptom)
                predicted_prob = self.compute_predicted_probability(gene, symptom)
                true_labels.append(true_label)
                predicted_probs.append(predicted_prob)
        true_labels = torch.tensor(true_labels, dtype=torch.float32)
        predicted_probs = torch.tensor(predicted_probs, dtype=torch.float32)

        predicted_labels = (predicted_probs >= threshold).float()

        tp = ((predicted_labels == 1) & (true_labels == 1)).sum().item()
        fp = ((predicted_labels == 1) & (true_labels == 0)).sum().item()
        fn = ((predicted_labels == 0) & (true_labels == 1)).sum().item()
        tn = ((predicted_labels == 0) & (true_labels == 0)).sum().item()

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        auroc = torchmetrics.AUROC(task="binary")(predicted_labels, true_labels)

        confusion_matrix = torch.zeros(2, 2)
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            confusion_matrix[int(true_label), int(predicted_label)] += 1

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "confusion_matrix": confusion_matrix,
        }
        if log_output:
            self.log_metrics(metrics)
        return metrics

    def log_metrics(self, metrics):
        logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"F1 Score: {metrics['f1']:.4f}")
        logging.info(f"AUROC: {metrics['auroc']:.4f}")
        logging.info(f"Confusion matrix:\n {metrics['confusion_matrix']}")

    def summarize_metrics(self, joint_metrics, per_gene_metrics):
        """Print metrics from both optimization stages."""
        logging.info("Metrics after first optimization (joint):")
        self.log_metrics(joint_metrics)
        logging.info("Metrics after second optimization (per-gene):")
        self.log_metrics(per_gene_metrics)

    def sum_max_frequencies(self, gene: str):
        # Sum the max frequencies for all symptoms associated with this gene
        return sum(list(self.groundtruth.gene_to_max_symptom_frequencies[gene].values()))

    def generate_shuffled_gene_symptom_pairs(self) -> Set[Tuple[str, str]]:
        # Create shuffled gene symptom pairs (negative examples) that are disjoint from groundtruth pairs
        num_groundtruth_pairs = len(self.groundtruth.gene_symptom_pairs)
        logging.info(f"Generating shuffled (disjoint) gene--symptom pairs..")
        max_possible_distinct_pairs = len(self.groundtruth.genes) * len(self.groundtruth.symptoms_list)
        target_num_shuffled_pairs = min(num_groundtruth_pairs, max_possible_distinct_pairs - num_groundtruth_pairs)
        shuffled_pairs_disjoint = set()
        logging.debug(f"Max possible distinct gene--symptom pairs is: {max_possible_distinct_pairs}; "
                      f"groundtruth contains {num_groundtruth_pairs} of those pairs. Thus we want "
                      f"{target_num_shuffled_pairs} shuffled pairs.")
        if target_num_shuffled_pairs < num_groundtruth_pairs:
            logging.warning(f"Not possible to generate as many shuffled pairs as there are groundtruth pairs!")
        while len(shuffled_pairs_disjoint) < target_num_shuffled_pairs:
            symptom_half_shuffled = random.sample(self.groundtruth.symptoms_list, len(self.groundtruth.symptoms_list))
            shuffled_pairs = set(list(zip(self.groundtruth.genes_list, symptom_half_shuffled)))
            shuffled_pairs_disjoint |= shuffled_pairs.difference(self.groundtruth.gene_symptom_pairs)

        # Trim down shuffled pairs to make sure we don't have MORE than we do groundtruth examples
        shuffled_pairs_disjoint = set(list(shuffled_pairs_disjoint)[:target_num_shuffled_pairs])
        assert len(shuffled_pairs_disjoint) == target_num_shuffled_pairs
        logging.info(f"Created {len(shuffled_pairs_disjoint)} shuffled gene--symptom pairs.")
        return shuffled_pairs_disjoint


def load_data(graph_dir_path: str, device: torch.device) -> Tuple[GraphHelper, GroundtruthHelper]:
    logging.info(f"LOADING DATA..")
    # Load the knowledge graph
    with jsonlines.open(f"{graph_dir_path}/nodes.jsonl") as reader:
        nodes = {row["id"]: row for row in reader}
    with jsonlines.open(f"{graph_dir_path}/edges.jsonl") as reader:
        edges = {row["id"]: row for row in reader}

    # Load orphanet data (provides ground truth gene--symptom labels)
    with open(f"{graph_dir_path}/gene_to_diseases.json", "r") as gene_to_diseases_file:
        gene_to_diseases = json.load(gene_to_diseases_file)
    gene_to_diseases = {gene_id: set(disease_ids) for gene_id, disease_ids in gene_to_diseases.items()}
    with open(f"{graph_dir_path}/disease_symptom_frequencies.json", "r") as disease_symp_file:
        disease_to_symptom_frequency_labels = json.load(disease_symp_file)

    # Do some data validation
    for disease, symptom_frequency_labels in disease_to_symptom_frequency_labels.items():
        for symptom, frequency_label in symptom_frequency_labels.items():
            if frequency_label not in SYMPTOM_FREQUENCY_MIDPOINTS:
                logging.error(f"Disease {disease} has symptom '{symptom}' whose frequency label of "
                              f"'{frequency_label}' doesn't appear in SYMPTOM_FREQUENCY_MIDPOINTS!")
            assert frequency_label in SYMPTOM_FREQUENCY_MIDPOINTS

    # Convert symptom frequency labels to actual floats
    disease_to_symptom_frequencies = {disease: {symptom: SYMPTOM_FREQUENCY_MIDPOINTS[frequency_label]
                                                for symptom, frequency_label in symptom_frequency_dict.items()}
                                      for disease, symptom_frequency_dict in disease_to_symptom_frequency_labels.items()}

    # Create some other handy data structures/maps
    gene_to_symptoms = defaultdict(set)
    gene_to_symptom_frequency_sets = defaultdict(lambda: defaultdict(set))
    for gene, diseases in gene_to_diseases.items():
        for disease in diseases:
            symptom_frequency_dict = disease_to_symptom_frequencies[disease]
            for symptom, frequency in symptom_frequency_dict.items():
                if frequency > 0.0:
                    gene_to_symptoms[gene].add(symptom)
                    gene_to_symptom_frequency_sets[gene][symptom].add(frequency)
    gene_symptom_pairs = {(gene, symptom) for gene, symptoms in gene_to_symptoms.items() for symptom in symptoms}
    gene_to_max_symptom_frequencies = {gene: {symptom: max(frequency_set)
                                              for symptom, frequency_set in symptom_frequency_sets.items()}
                                       for gene, symptom_frequency_sets in gene_to_symptom_frequency_sets.items()}

    # Wrap up all our groundtruth maps into one object
    groundtruth = GroundtruthHelper(gene_to_diseases=gene_to_diseases,
                                    disease_to_symptom_frequencies=disease_to_symptom_frequencies,
                                    gene_to_symptoms=gene_to_symptoms,
                                    genes=set(gene_to_diseases),
                                    diseases=set(disease_to_symptom_frequencies),
                                    symptoms={symptom_id
                                              for disease, symptoms in disease_to_symptom_frequencies.items()
                                              for symptom_id in symptoms},
                                    gene_symptom_pairs=gene_symptom_pairs,
                                    gene_to_symptom_frequency_sets=gene_to_symptom_frequency_sets,
                                    gene_to_max_symptom_frequencies=gene_to_max_symptom_frequencies)

    # And wrap up all our graph data into one object
    graph = GraphHelper(nodes_map=nodes, edges_map=edges, device=device)

    return graph, groundtruth


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("graph_dir",
                            help="Path to the directory containing files for the KG you want to use. (Must include "
                                 "nodes.jsonl, edges.jsonl, gene_to_diseases.json, and "
                                 "disease_symptom_frequencies.json.)")
    args = arg_parser.parse_args()

    # Use GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    classifier = GeneSymptomClassifier(args.graph_dir, device=device)

    classifier.train_model()


if __name__ == "__main__":
    main()
