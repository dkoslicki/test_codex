#!/usr/bin/env python
"""
Fast RareTarget implementation – v2
-----------------------------------
Changes vs. the previous version
• **Fully‑vectorised loss:**  all gene‑symptom pairs are processed in one
  big gather, eliminating a long Python loop that was CPU‑bound.
• Same sparse message‑passing core → still tiny VRAM.
• Same logging + metrics output (training.log & performance.txt).
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
from torch_sparse import SparseTensor             # pip install torch-sparse
from sklearn.metrics import (                     # pip install scikit-learn
    roc_auc_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
)

# ────────────────────────── logging ──────────────────────────────────────────
_log: List[str] = []


class _MemHandler(logging.Handler):
    def emit(self, record):
        _log.append(self.format(record))


logging.basicConfig(level=logging.DEBUG, handlers=[_MemHandler()],
                    format="%(asctime)s %(levelname)s: %(message)s")

# ────────────────────────── constants ────────────────────────────────────────
SYMPTOM_FREQ = {
    "obligate": 1.00,
    "very_frequent": 0.90,
    "frequent": 0.55,
    "occasional": 0.17,
    "very_rare": 0.02,
}
AVG_FREQ = torch.tensor(list(SYMPTOM_FREQ.values())).mean()


# ────────────────────────── helpers ──────────────────────────────────────────
class GroundtruthHelper:
    def __init__(
        self,
        gene_to_diseases,
        disease_to_symptom_frequencies,
        gene_to_symptoms,
        gene_symptom_pairs,
        gene_to_symptom_frequency_sets,
        gene_to_max_symptom_frequencies,
        genes,
        diseases,
        symptoms,
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

        self.genes_list = list(genes)
        self.symptoms_list = list(symptoms)
        self.gene_idx = {g: i for i, g in enumerate(self.genes_list)}

        logging.debug(
            f"Ground‑truth: {len(genes)} genes, {len(diseases)} diseases, "
            f"{len(symptoms)} symptoms – {len(gene_symptom_pairs)} pairs."
        )

    def assoc(self, g: str, s: str) -> int:
        return int((g, s) in self.gene_symptom_pairs)


class GraphHelper:
    def __init__(self, nodes_map, edges_map, device):
        self.nodes_map = nodes_map
        self.edges_map = edges_map

        self.predicates_list = list({e["predicate"] for e in edges_map.values()})
        self.num_predicates = len(self.predicates_list)
        self.pred_idx = {p: i for i, p in enumerate(self.predicates_list)}

        self.nodes_list = list(nodes_map)
        self.num_nodes = len(self.nodes_list)
        self.node_idx = {n: i for i, n in enumerate(self.nodes_list)}

        self._build_sparse_edges(device)

        logging.debug(
            f"Graph: {self.num_nodes} nodes, {len(edges_map)} edges, "
            f"{self.num_predicates} predicates."
        )

    def _build_sparse_edges(self, device):
        rows, cols, preds = [], [], []
        for e in self.edges_map.values():
            rows.append(self.node_idx[e["subject"]])
            cols.append(self.node_idx[e["object"]])
            preds.append(self.pred_idx[e["predicate"]])
        self.edge_index = torch.tensor([rows, cols], device=device)
        self.edge_preds = torch.tensor(preds, device=device)
        self.size = (self.num_nodes, self.num_nodes)


# ────────────────────────── model ────────────────────────────────────────────
class GeneSymptomClassifier:
    # ---------------- init ----------------------------------------------------
    def __init__(self, graph_dir: str, device: torch.device):
        self.device = device
        self.graph, self.gt = load_data(graph_dir, device)

        self.neg_pairs = self._make_negatives()
        self.neg_by_gene = defaultdict(set)
        for g, s in self.neg_pairs:
            self.neg_by_gene[g].add(s)

        # ►► pre‑compute *all* training pairs and weights (vectorised loss)
        self._prepare_pair_tensors()

        # params
        self.node_w = torch.rand(len(self.gt.genes), self.graph.num_nodes,
                                 device=device, requires_grad=True)
        self.pred_w = torch.rand(self.graph.num_predicates,
                                 device=device, requires_grad=True)
        self.bias = torch.tensor(-10.0, device=device, requires_grad=True)

        # hyper
        self.L1_REG, self.L2_REG = 1e-8, 1e-8
        self.PRED_L2_REG = 1e-7
        self.MAX_PATH = 4
        self.LR = 0.1
        self.MIN_DELTA, self.PATIENCE = 1e-4, 20

    # ---------------- negatives ----------------------------------------------
    def _make_negatives(self) -> Set[Tuple[str, str]]:
        gp = self.gt.gene_symptom_pairs
        n_pos = len(gp)
        max_pairs = len(self.gt.genes) * len(self.gt.symptoms)
        tgt = min(n_pos, max_pairs - n_pos)
        neg: Set[Tuple[str, str]] = set()
        while len(neg) < tgt:
            shuffled = random.sample(self.gt.symptoms_list, len(self.gt.symptoms))
            neg |= set(zip(self.gt.genes_list, shuffled)) - gp
            neg = set(list(neg)[:tgt])
        return neg

    # ---------------- pair tensors (vectorised loss) --------------------------
    def _prepare_pair_tensors(self):
        g_idx, s_idx, labels, freqs = [], [], [], []
        for gi, gene in enumerate(self.gt.genes_list):
            pos = self.gt.gene_to_symptoms.get(gene, set())
            neg = self.neg_by_gene.get(gene, set())
            for s in pos | neg:
                if s not in self.graph.node_idx:
                    continue
                g_idx.append(gi)
                s_idx.append(self.graph.node_idx[s])
                labels.append(1 if s in pos else 0)
                freqs.append(self._freq(gene, s))
        self.pair_g = torch.tensor(g_idx, dtype=torch.long, device=self.device)
        self.pair_s = torch.tensor(s_idx, dtype=torch.long, device=self.device)
        self.pair_lbl = torch.tensor(labels, dtype=torch.float32, device=self.device)
        self.pair_freq = torch.tensor(freqs, dtype=torch.float32, device=self.device)

        denom = [max(self._sum_freqs(g), 1e-6) for g in self.gt.genes_list]
        self.gene_denom = torch.tensor(denom, dtype=torch.float32, device=self.device)

    # ---------------- sparse helpers -----------------------------------------
    def _make_A(self, pred_w: Tensor) -> SparseTensor:
        vals = pred_w[self.graph.edge_preds]
        return SparseTensor(row=self.graph.edge_index[0],
                            col=self.graph.edge_index[1],
                            value=vals,
                            sparse_sizes=self.graph.size)

    def _paths(self, start: Tensor, node_w: Tensor, A: SparseTensor, L=4):
        v, out = start, torch.zeros_like(start)
        for _ in range(1, L):
            dense = v * node_w
            v = A.t().matmul(dense.T).T
            out += v
        return out  # (B,N)

    # ---------------- objective (vectorised) ---------------------------------
    def objective(self, params):
        node_w, pred_w, bias = params
        A = self._make_A(pred_w)

        B, N = node_w.shape
        one_hot = torch.zeros(B, N, device=self.device)
        one_hot[torch.arange(B), [
            self.graph.node_idx[g] for g in self.gt.genes_list]] = 1.0

        scores = self._paths(one_hot, node_w, A, self.MAX_PATH)
        raw = scores[self.pair_g, self.pair_s] + bias
        prob = torch.sigmoid(raw)

        ce = -(self.pair_lbl * torch.log(prob + 1e-15)
               + (1 - self.pair_lbl) * torch.log(1 - prob + 1e-15))
        w_ce = ce * self.pair_freq / self.gene_denom[self.pair_g]
        loss_main = w_ce.sum() / B

        l1 = (self.L1_REG * node_w.abs().sum(dim=1) / N).mean()
        l2 = (self.L2_REG * torch.sqrt((node_w ** 2).sum(dim=1)) / N).mean()
        pred_l2 = (self.PRED_L2_REG * (pred_w ** 2).sum()) / self.graph.num_predicates
        return loss_main + l1 + l2 + pred_l2

    # ---------------- training -----------------------------------------------
    def train(self):
        logging.info("TRAIN …")
        params = [self.node_w, self.pred_w, self.bias]
        opt = torch.optim.Adam(params, lr=self.LR)
        best, patience = float("inf"), 0
        t0 = time.time()
        for it in range(1000):
            opt.zero_grad(set_to_none=True)
            loss = self.objective(params)
            loss.backward()
            opt.step()
            if loss < best - self.MIN_DELTA:
                best, patience = loss.item(), 0
            else:
                patience += 1
            if it % 100 == 0:
                logging.debug(f"iter {it:04d} loss={loss:.4f}")
            if patience >= self.PATIENCE:
                break
        logging.info(f"Joint phase: {(time.time()-t0):.1f}s  best={best:.4f}")

        # ---------- per‑gene fine‑tune (unchanged, cheap) ---------------------
        self.pred_w.requires_grad_(False)
        self.bias.requires_grad_(False)
        self.node_w.requires_grad_(False)

        for gi, gene in enumerate(self.gt.genes_list):
            local = self.node_w[gi].clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([local], lr=self.LR)
            best_l, pat = float("inf"), 0
            for _ in range(250):            # fewer iters now – faster
                opt.zero_grad()
                l = self._gene_objective(local, gene)
                l.backward()
                opt.step()
                if l < best_l - self.MIN_DELTA:
                    best_l, pat = l.item(), 0
                else:
                    pat += 1
                if pat >= self.PATIENCE:
                    break
            with torch.no_grad():
                self.node_w[gi].copy_(local)
            logging.debug(f"[gene] {gene} loss={best_l:.4f}")

    # ------------- gene‑specific objective (unchanged maths) -----------------
    def _gene_objective(self, node_w_g: Tensor, gene: str):
        A = self._make_A(self.pred_w.detach())
        N = self.graph.num_nodes
        one_hot = torch.zeros(1, N, device=self.device)
        one_hot[0, self.graph.node_idx[gene]] = 1.0
        paths = self._paths(one_hot, node_w_g.unsqueeze(0), A, self.MAX_PATH)[0]

        pos = self.gt.gene_to_symptoms.get(gene, set())
        neg = self.neg_by_gene.get(gene, set())
        syms = pos | neg
        idx = torch.tensor([self.graph.node_idx[s] for s in syms],
                           device=self.device)
        raw = paths[idx] + self.bias.detach()
        prob = torch.sigmoid(raw)

        lbl = torch.tensor([self.gt.assoc(gene, s) for s in syms],
                           dtype=torch.float32, device=self.device)
        frq = torch.tensor([self._freq(gene, s) for s in syms],
                           dtype=torch.float32, device=self.device)

        ce = -(lbl * torch.log(prob + 1e-15)
               + (1 - lbl) * torch.log(1 - prob + 1e-15))
        weighted = (frq * ce).sum() / self._sum_freqs(gene)

        l1 = self.L1_REG * node_w_g.abs().sum() / N
        l2 = self.L2_REG * torch.sqrt((node_w_g ** 2).sum()) / N
        return weighted + l1 + l2

    # ---------------- evaluation (vectorised) -------------------------------
    @torch.no_grad()
    def evaluate(self):
        A = self._make_A(self.pred_w)
        B, N = len(self.gt.genes), self.graph.num_nodes
        one_hot = torch.zeros(B, N, device=self.device)
        one_hot[torch.arange(B),
                [self.graph.node_idx[g] for g in self.gt.genes_list]] = 1.0
        paths = self._paths(one_hot, self.node_w, A, self.MAX_PATH)
        y_scores = torch.sigmoid(paths[self.pair_g, self.pair_s] + self.bias).cpu().numpy()
        y_true = self.pair_lbl.cpu().numpy()
        y_pred = (y_scores >= 0.5).astype(int)

        metrics = {
            "AUC": roc_auc_score(y_true, y_scores)
            if len(np.unique(y_true)) > 1 else float("nan"),
            "F1": f1_score(y_true, y_pred),
            "Accuracy": accuracy_score(y_true, y_pred),
            "ConfusionMatrix": confusion_matrix(y_true, y_pred).ravel(),
        }
        return metrics

    # ---------------- utils --------------------------------------------------
    def _freq(self, g, s):
        if s in self.gt.gene_to_max_symptom_frequencies.get(g, {}):
            return self.gt.gene_to_max_symptom_frequencies[g][s]
        return float(AVG_FREQ)

    def _sum_freqs(self, g):
        return sum(self.gt.gene_to_max_symptom_frequencies[g].values())


# ────────────────────────── data IO ──────────────────────────────────────────
def load_data(graph_dir, device):
    logging.info("LOAD DATA …")
    with jsonlines.open(os.path.join(graph_dir, "nodes.jsonl")) as r:
        nodes = {row["id"]: row for row in r}
    with jsonlines.open(os.path.join(graph_dir, "edges.jsonl")) as r:
        edges = {row["id"]: row for row in r}
    with open(os.path.join(graph_dir, "gene_to_diseases.json")) as f:
        g2d = {k: set(v) for k, v in json.load(f).items()}
    with open(os.path.join(graph_dir, "disease_symptom_frequencies.json")) as f:
        d2sf_lbl = json.load(f)

    d2sf = {d: {s: SYMPTOM_FREQ[lbl] for s, lbl in m.items()}
            for d, m in d2sf_lbl.items()}

    g2s = defaultdict(set)
    g2s_freq_sets = defaultdict(lambda: defaultdict(set))
    for g, dis in g2d.items():
        for d in dis:
            for s, f in d2sf[d].items():
                if f > 0:
                    g2s[g].add(s)
                    g2s_freq_sets[g][s].add(f)

    gpairs = {(g, s) for g, ss in g2s.items() for s in ss}
    g2s_max = {g: {s: max(fs) for s, fs in m.items()}
               for g, m in g2s_freq_sets.items()}

    gt = GroundtruthHelper(
        g2d, d2sf, g2s, gpairs, g2s_freq_sets, g2s_max,
        genes=set(g2d), diseases=set(d2sf), symptoms={s for d in d2sf.values() for s in d}
    )
    graph = GraphHelper(nodes, edges, device)
    return graph, gt


# ────────────────────────── main ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("graph_dir", help="directory with Orphanet graph files")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"device: {device}")

    model = GeneSymptomClassifier(args.graph_dir, device)
    model.train()
    metrics = model.evaluate()

    print("\n=== METRICS ===")
    tn, fp, fn, tp = metrics["ConfusionMatrix"]
    print(f"Confusion Matrix: tn={tn} fp={fp} fn={fn} tp={tp}")
    for k in ("AUC", "F1", "Accuracy"):
        print(f"{k}: {metrics[k]:.4f}")

    with open("training.log", "w") as f:
        f.write("\n".join(_log))
    with open("performance.txt", "w") as f:
        tn, fp, fn, tp = metrics["ConfusionMatrix"]
        f.write(f"ConfusionMatrix tn={tn} fp={fp} fn={fn} tp={tp}\n")
        for k in ("AUC", "F1", "Accuracy"):
            f.write(f"{k}: {metrics[k]:.6f}\n")
    logging.info("logs written.")


if __name__ == "__main__":
    random.seed(22)
    torch.manual_seed(22)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(22)
    main()

