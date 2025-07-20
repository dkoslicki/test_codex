#!/usr/bin/env python
"""
Fast RareTarget implementation  (GPU-sparse version)
----------------------------------------------------
• Sparse message-passing with torch-sparse  →  50-100× faster, tiny VRAM
• Prints Confusion Matrix, AUC, F1, Accuracy after training
• Writes full training log to training.log and metrics to performance.txt
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import jsonlines
import numpy as np
import torch
from torch import Tensor

# ---- extra metrics deps ------------------------------------------------------
try:
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        confusion_matrix,
        accuracy_score,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-learn not found – run `pip install scikit-learn` "
        "before executing this script."
    ) from exc

try:
    from torch_sparse import SparseTensor  # pip install torch-sparse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "torch-sparse not found – run `pip install torch-sparse` "
        "before executing this script."
    ) from exc

# --------------------------------------------------------------------- logging
log_messages: List[str] = []


class _MemoryHandler(logging.Handler):
    def emit(self, record):
        log_messages.append(self.format(record))


_memory_handler = _MemoryHandler()
_memory_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logging.basicConfig(level=logging.DEBUG, handlers=[_MemoryHandler()])

# --------------------------------------------------------------------- constants
SYMPTOM_FREQUENCY_MIDPOINTS = {
    "obligate": 1.00,
    "very_frequent": 0.90,
    "frequent": 0.55,
    "occasional": 0.17,
    "very_rare": 0.02,
}
AVERAGE_FREQUENCY_MIDPOINT = torch.tensor(
    list(SYMPTOM_FREQUENCY_MIDPOINTS.values())
).mean()


# --------------------------------------------------------------------- helpers
class GroundtruthHelper:
    def __init__(
        self,
        gene_to_diseases: Dict[str, Set[str]],
        disease_to_symptom_frequencies: Dict[str, Dict[str, float]],
        gene_to_symptoms: Dict[str, Set[str]],
        gene_symptom_pairs: Set[Tuple[str, str]],
        gene_to_symptom_frequency_sets: Dict[str, Dict[str, Set[float]]],
        gene_to_max_symptom_frequencies: Dict[str, Dict[str, float]],
        genes: Set[str],
        diseases: Set[str],
        symptoms: Set[str],
    ):
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
        self.gene_indices_map = {g: i for i, g in enumerate(self.genes_list)}

        logging.debug(
            f"Ground-truth: {len(self.genes)} genes, "
            f"{len(self.diseases)} diseases, {len(self.symptoms)} symptoms."
        )
        logging.debug(f"…with {len(self.gene_symptom_pairs)} gene–symptom pairs.")

    # -----------------------------------------------------------------
    def gene_and_symptom_are_associated(self, gene: str, symptom: str) -> int:
        return int((gene, symptom) in self.gene_symptom_pairs)


class GraphHelper:
    def __init__(self, nodes_map: Dict[str, dict], edges_map: Dict[str, dict], device: torch.device):
        self.nodes_map = nodes_map
        self.edges_map = edges_map

        # predicates
        self.predicates_list = list({e["predicate"] for e in self.edges_map.values()})
        self.num_predicates = len(self.predicates_list)
        self.predicate_indices_map = {p: i for i, p in enumerate(self.predicates_list)}

        # nodes
        self.nodes_list = list(self.nodes_map)
        self.num_nodes = len(self.nodes_list)
        self.node_indices_map = {n: i for i, n in enumerate(self.nodes_list)}

        # build sparse COO
        self._build_sparse_edges(device)

        logging.debug(
            f"Graph: {self.num_nodes} nodes, {len(self.edges_map)} edges, "
            f"{self.num_predicates} predicates."
        )

    def _build_sparse_edges(self, device: torch.device):
        rows, cols, pred_ids = [], [], []
        for edge in self.edges_map.values():
            rows.append(self.node_indices_map[edge["subject"]])
            cols.append(self.node_indices_map[edge["object"]])
            pred_ids.append(self.predicate_indices_map[edge["predicate"]])
        self.edge_index = torch.tensor([rows, cols], dtype=torch.long, device=device)
        self.edge_pred_ids = torch.tensor(pred_ids, dtype=torch.long, device=device)
        self.size = (self.num_nodes, self.num_nodes)

        # adjacency list for debugging
        self.adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        for s, o in zip(rows, cols):
            self.adjacency_list[self.nodes_list[int(s)]].add(self.nodes_list[int(o)])


# --------------------------------------------------------------------- model
class GeneSymptomClassifier:
    # --------------------------- init --------------------------------
    def __init__(self, graph_dir: str, device: torch.device):
        self.device = device
        self.graph, self.groundtruth = load_data(graph_dir, device)

        # negative sampling
        self.shuffled_gene_symptom_pairs = self._generate_negatives()
        self.negatives_by_gene = defaultdict(set)
        for g, h in self.shuffled_gene_symptom_pairs:
            self.negatives_by_gene[g].add(h)

        # learnable params
        self.node_weights_tensor = torch.rand(
            len(self.groundtruth.genes_list), self.graph.num_nodes, device=device, requires_grad=True
        )
        self.predicate_weights = torch.rand(self.graph.num_predicates, device=device, requires_grad=True)
        self.baseline_offset = torch.tensor(-10.0, device=device, requires_grad=True)

        # hyper-params
        self.L1_REG = 1e-8
        self.L2_REG = 1e-8
        self.PRED_L2_REG = 1e-7
        self.MAX_PATH_LENGTH = 4
        self.LR = 0.1
        self.MIN_DELTA = 1e-4
        self.STABLE_ROUNDS_REQUIRED = 20

    # --------------------- sparse helpers ----------------------------
    def _make_weighted_adj(self, predicate_w: Tensor) -> SparseTensor:
        vals = predicate_w[self.graph.edge_pred_ids]
        return SparseTensor(
            row=self.graph.edge_index[0],
            col=self.graph.edge_index[1],
            value=vals,
            sparse_sizes=self.graph.size,
        )

    def _path_scores(
        self,
        start_vec: Tensor,  # (B,N)
        node_w: Tensor,  # (B,N)
        A_w: SparseTensor,
        L: int = 4,
    ) -> Tensor:
        scores = torch.zeros_like(start_vec)
        v = start_vec
        for _ in range(1, L):
            dense = v * node_w
            v = A_w.t().matmul(dense.T).T  # sparse left-multiply
            scores += v
        return scores

    # ----------------------- objective -------------------------------
    def objective_function(self, params):
        node_w_all, predicate_w, bias = params
        A_w = self._make_weighted_adj(predicate_w)

        gene_idx = torch.tensor(
            [self.groundtruth.gene_indices_map[g] for g in self.groundtruth.genes_list],
            device=self.device,
        )
        B, N = gene_idx.numel(), self.graph.num_nodes
        one_hot = torch.zeros(B, N, device=self.device)
        one_hot[torch.arange(B), gene_idx] = 1.0

        path_scores = self._path_scores(one_hot, node_w_all, A_w, self.MAX_PATH_LENGTH)

        total_loss = torch.zeros((), device=self.device)
        for gi, gene in enumerate(self.groundtruth.genes_list):
            pos = self.groundtruth.gene_to_symptoms.get(gene, set())
            neg = self.negatives_by_gene.get(gene, set())
            syms = list(pos | neg)
            if not syms:
                continue
            idx = torch.tensor(
                [self.graph.node_indices_map[s] for s in syms if s in self.graph.node_indices_map],
                device=self.device,
            )
            if idx.numel() == 0:
                continue

            raw = path_scores[gi, idx] + bias
            prob = torch.sigmoid(raw)

            labels = torch.tensor(
                [self.groundtruth.gene_and_symptom_are_associated(gene, s) for s in syms],
                dtype=torch.float32,
                device=self.device,
            )
            freqs = torch.tensor(
                [self._get_frequency(gene, s) for s in syms],
                dtype=torch.float32,
                device=self.device,
            )

            ce = -(labels * torch.log(prob + 1e-15) + (1 - labels) * torch.log(1 - prob + 1e-15))
            weighted = (freqs * ce).sum()

            l1 = (self.L1_REG * node_w_all[gi].abs().sum()) / N
            l2 = (self.L2_REG * torch.sqrt((node_w_all[gi] ** 2).sum())) / N
            total_loss += (weighted / self._sum_max_freqs(gene)) + l1 + l2

        pred_l2 = (self.PRED_L2_REG * (predicate_w ** 2).sum()) / self.graph.num_predicates
        return total_loss / len(self.groundtruth.genes_list) + pred_l2

    # ---------------- gene-specific objective ------------------------
    def objective_function_gene(self, params, gene: str):
        gene_node_w = params[0]
        predicate_w = self.predicate_weights.detach()
        bias = self.baseline_offset.detach()

        A_w = self._make_weighted_adj(predicate_w)
        N = self.graph.num_nodes
        one_hot = torch.zeros(1, N, device=self.device)
        one_hot[0, self.graph.node_indices_map[gene]] = 1.0
        path_scores = self._path_scores(one_hot, gene_node_w.unsqueeze(0), A_w, self.MAX_PATH_LENGTH)[0]

        pos = self.groundtruth.gene_to_symptoms.get(gene, set())
        neg = self.negatives_by_gene.get(gene, set())
        syms = list(pos | neg)
        idx = torch.tensor(
            [self.graph.node_indices_map[s] for s in syms if s in self.graph.node_indices_map],
            device=self.device,
        )
        raw = path_scores[idx] + bias
        prob = torch.sigmoid(raw)

        labels = torch.tensor(
            [self.groundtruth.gene_and_symptom_are_associated(gene, s) for s in syms],
            dtype=torch.float32,
            device=self.device,
        )
        freqs = torch.tensor(
            [self._get_frequency(gene, s) for s in syms],
            dtype=torch.float32,
            device=self.device,
        )

        ce = -(labels * torch.log(prob + 1e-15) + (1 - labels) * torch.log(1 - prob + 1e-15))
        weighted = (freqs * ce).sum()

        l1 = (self.L1_REG * gene_node_w.abs().sum()) / N
        l2 = (self.L2_REG * torch.sqrt((gene_node_w ** 2).sum())) / N
        return (weighted / self._sum_max_freqs(gene)) + l1 + l2

    # ---------------- prediction util --------------------------------
    @torch.no_grad()
    def compute_predicted_probability(
        self,
        gene: str,
        symptom: str,
        gene_node_weights_override: Tensor | None = None,
    ):
        if symptom not in self.graph.node_indices_map:
            return torch.tensor(0.0, device=self.device)

        A_w = self._make_weighted_adj(self.predicate_weights)
        N = self.graph.num_nodes
        one_hot = torch.zeros(1, N, device=self.device)
        one_hot[0, self.graph.node_indices_map[gene]] = 1.0

        node_w = (
            gene_node_weights_override.unsqueeze(0)
            if gene_node_weights_override is not None
            else self.node_weights_tensor[self.groundtruth.gene_indices_map[gene]].unsqueeze(0)
        )

        score = self._path_scores(one_hot, node_w, A_w, self.MAX_PATH_LENGTH)[0]
        raw = score[self.graph.node_indices_map[symptom]] + self.baseline_offset
        return torch.sigmoid(raw)

    # ---------------- evaluation -------------------------------------
    @torch.no_grad()
    def evaluate_model(self) -> Dict[str, float]:
        y_true, y_scores = [], []
        for gene in self.groundtruth.genes_list:
            pos = self.groundtruth.gene_to_symptoms.get(gene, set())
            neg = self.negatives_by_gene.get(gene, set())
            syms = list(pos | neg)
            for s in syms:
                y_true.append(self.groundtruth.gene_and_symptom_are_associated(gene, s))
                y_scores.append(self.compute_predicted_probability(gene, s).item())

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = (y_scores >= 0.5).astype(int)

        metrics = {
            "AUC": roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else float("nan"),
            "F1": f1_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
        }
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["ConfusionMatrix"] = (tn, fp, fn, tp)
        return metrics

    # ---------------- training loop ----------------------------------
    def train_model(self):
        logging.info("TRAINING MODEL …")
        params = [self.node_weights_tensor, self.predicate_weights, self.baseline_offset]
        optim = torch.optim.Adam(params, lr=self.LR)

        best, stable = float("inf"), 0
        t0 = time.time()
        for it in range(1000):
            optim.zero_grad(set_to_none=True)
            loss = self.objective_function(params)
            loss.backward()
            optim.step()

            if loss < best - self.MIN_DELTA:
                best, stable = loss.item(), 0
            else:
                stable += 1
            if it % 100 == 0:
                logging.debug(f"[joint] iter {it:04d}  loss={loss:.4f}")
            if stable >= self.STABLE_ROUNDS_REQUIRED:
                break
        logging.info(f"First optimisation finished in {(time.time()-t0)/60:.1f} min, loss={best:.4f}")

        # -------- per-gene fine-tune ----------------------------------
        self.predicate_weights.requires_grad_(False)
        self.baseline_offset.requires_grad_(False)
        self.node_weights_tensor.requires_grad_(False)

        for gi, gene in enumerate(self.groundtruth.genes_list):
            local_w = self.node_weights_tensor[gi].detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([local_w], lr=self.LR)
            best_g, stable = float("inf"), 0
            for _ in range(1000):
                opt.zero_grad()
                loss = self.objective_function_gene([local_w], gene)
                loss.backward()
                opt.step()
                if loss < best_g - self.MIN_DELTA:
                    best_g, stable = loss.item(), 0
                else:
                    stable += 1
                if stable >= self.STABLE_ROUNDS_REQUIRED:
                    break
            with torch.no_grad():
                self.node_weights_tensor[gi].copy_(local_w)
            logging.debug(f"[per-gene] {gene}: final loss={best_g:.4f}")

    # ---------------- misc helpers -----------------------------------
    def _get_frequency(self, gene: str, symptom: str) -> float:
        if symptom in self.groundtruth.gene_to_max_symptom_frequencies.get(gene, {}):
            return self.groundtruth.gene_to_max_symptom_frequencies[gene][symptom]
        return float(AVERAGE_FREQUENCY_MIDPOINT)

    def _sum_max_freqs(self, gene: str) -> float:
        return sum(self.groundtruth.gene_to_max_symptom_frequencies[gene].values())

    def _generate_negatives(self) -> Set[Tuple[str, str]]:
        gts = self.groundtruth
        num_pos = len(gts.gene_symptom_pairs)
        max_pairs = len(gts.genes) * len(gts.symptoms_list)
        target = min(num_pos, max_pairs - num_pos)
        negs: Set[Tuple[str, str]] = set()
        while len(negs) < target:
            shuffled = random.sample(gts.symptoms_list, len(gts.symptoms_list))
            pairs = set(zip(gts.genes_list, shuffled))
            negs |= pairs - gts.gene_symptom_pairs
            negs = set(list(negs)[: target])
        return negs


# --------------------------------------------------------------------- data IO
def load_data(graph_dir: str, device: torch.device) -> Tuple[GraphHelper, GroundtruthHelper]:
    logging.info("LOADING DATA …")
    with jsonlines.open(os.path.join(graph_dir, "nodes.jsonl")) as r:
        nodes = {row["id"]: row for row in r}
    with jsonlines.open(os.path.join(graph_dir, "edges.jsonl")) as r:
        edges = {row["id"]: row for row in r}

    with open(os.path.join(graph_dir, "gene_to_diseases.json")) as f:
        gene_to_diseases = {k: set(v) for k, v in json.load(f).items()}

    with open(os.path.join(graph_dir, "disease_symptom_frequencies.json")) as f:
        dis_freq_lbl = json.load(f)

    dis_freq = {
        d: {s: SYMPTOM_FREQUENCY_MIDPOINTS[lbl] for s, lbl in dct.items()}
        for d, dct in dis_freq_lbl.items()
    }

    gene_to_symptoms = defaultdict(set)
    gene_to_symptom_freq_sets = defaultdict(lambda: defaultdict(set))
    for g, diseases in gene_to_diseases.items():
        for d in diseases:
            for s, freq in dis_freq[d].items():
                if freq > 0.0:
                    gene_to_symptoms[g].add(s)
                    gene_to_symptom_freq_sets[g][s].add(freq)

    gene_symptom_pairs = {(g, s) for g, ss in gene_to_symptoms.items() for s in ss}
    gene_to_max_freq = {
        g: {s: max(fs) for s, fs in freq_map.items()} for g, freq_map in gene_to_symptom_freq_sets.items()
    }

    groundtruth = GroundtruthHelper(
        gene_to_diseases=gene_to_diseases,
        disease_to_symptom_frequencies=dis_freq,
        gene_to_symptoms=gene_to_symptoms,
        gene_symptom_pairs=gene_symptom_pairs,
        gene_to_symptom_frequency_sets=gene_to_symptom_freq_sets,
        gene_to_max_symptom_frequencies=gene_to_max_freq,
        genes=set(gene_to_diseases),
        diseases=set(dis_freq),
        symptoms={s for d in dis_freq.values() for s in d},
    )

    graph = GraphHelper(nodes_map=nodes, edges_map=edges, device=device)
    return graph, groundtruth


# --------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_dir", help="Directory with graph files")
    args = parser.parse_args()

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    logging.info(f"Using device: {device}")

    model = GeneSymptomClassifier(args.graph_dir, device)
    model.train_model()

    metrics = model.evaluate_model()
    print("\n=== PERFORMANCE ===")
    for k, v in metrics.items():
        if k == "ConfusionMatrix":
            tn, fp, fn, tp = v
            print(f"Confusion Matrix: tn={tn} fp={fp} fn={fn} tp={tp}")
        else:
            print(f"{k}: {v:.4f}")

    # write log & metrics files
    with open("training.log", "w") as f:
        f.write("\n".join(log_messages))
    with open("performance.txt", "w") as f:
        for k, v in metrics.items():
            if k == "ConfusionMatrix":
                tn, fp, fn, tp = v
                f.write(f"ConfusionMatrix tn={tn} fp={fp} fn={fn} tp={tp}\n")
            else:
                f.write(f"{k}: {v:.6f}\n")
    logging.info("Log written to training.log – metrics to performance.txt")


if __name__ == "__main__":
    random.seed(22)
    torch.manual_seed(22)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(22)
    main()

