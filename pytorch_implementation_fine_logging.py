import argparse
import logging
import time

import torch

from pytorch_implementation import (
    GeneSymptomClassifier,
)


class GeneSymptomClassifierFineLogging(GeneSymptomClassifier):
    """Same as :class:`GeneSymptomClassifier` but with more verbose progress logs."""

    def train_model(self):
        logging.info("TRAINING MODEL..")
        logging.debug(f"Predicate weights tensor is: {self.predicate_weights}")
        logging.debug(f"Baseline offset is: {self.baseline_offset}")
        logging.debug(f"Shape of node weights tensor is: {self.node_weights_tensor.shape}")

        logging.info("Starting first optimization..")
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

            total_time = step_end - iter_start
            forward_time = fwd_end - fwd_start
            backward_time = bwd_end - fwd_end
            step_time = step_end - bwd_end
            mem_info = ""
            device = self.node_weights_tensor.device
            if device.type == "cuda":
                mem_info = (
                    f" | GPU mem: {torch.cuda.memory_allocated(device)/1e6:.1f}MB"
                )
            logging.info(
                f"Iter {iteration_num} - loss: {loss:.4f} "
                f"| fwd: {forward_time:.2f}s bwd: {backward_time:.2f}s "
                f"step: {step_time:.2f}s total: {total_time:.2f}s" + mem_info
            )
            if stable_rounds >= self.STABLE_ROUNDS_REQUIRED:
                logging.info(
                    f"Reached stability after {iteration_num} iterations. Loss is {loss:.4f}"
                )
                break
        logging.info(
            f"Done with first optimization. Took {round((time.time() - start) / 60)} minutes."
        )

        logging.info(f"Predicate weights tensor is now: {self.predicate_weights}")
        logging.info(f"Baseline offset tensor is: {self.baseline_offset}")
        self.show_predicted_labels()
        first_metrics = self.evaluate_classification(threshold=0.5, log_output=False)
        self.show_top_intermediate_nodes()

        logging.info(
            "Starting second optimization (focused on gene-specific node weights)"
        )
        start = time.time()

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
                iter_start = time.time()
                optimizer_2.zero_grad()
                fwd_start = time.time()
                loss = self.objective_function_gene(params, gene=gene)
                fwd_end = time.time()
                loss.backward()
                bwd_end = time.time()
                optimizer_2.step()
                step_end = time.time()

                if loss < best_loss - self.MIN_DELTA:
                    best_loss = loss
                else:
                    stable_rounds += 1

                total_time = step_end - iter_start
                forward_time = fwd_end - fwd_start
                backward_time = bwd_end - fwd_end
                step_time = step_end - bwd_end
                mem_info = ""
                device = gene_node_weights.device
                if device.type == "cuda":
                    mem_info = (
                        f" | GPU mem: {torch.cuda.memory_allocated(device)/1e6:.1f}MB"
                    )
                logging.info(
                    f"Iter {iteration_num} for {gene} - loss: {loss:.4f} "
                    f"| fwd: {forward_time:.2f}s bwd: {backward_time:.2f}s "
                    f"step: {step_time:.2f}s total: {total_time:.2f}s" + mem_info
                )
                if stable_rounds >= self.STABLE_ROUNDS_REQUIRED:
                    break

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


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "graph_dir",
        help=(
            "Path to the directory containing files for the KG you want to use. "
            "(Must include nodes.jsonl, edges.jsonl, gene_to_diseases.json, and "
            "disease_symptom_frequencies.json.)"
        ),
    )
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    classifier = GeneSymptomClassifierFineLogging(args.graph_dir, device=device)
    classifier.train_model()


if __name__ == "__main__":
    main()
