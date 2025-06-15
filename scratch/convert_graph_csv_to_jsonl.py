import argparse
import csv
import os

import jsonlines

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("csv_dir", help="Patch to directory containing edges csv files.")
args = arg_parser.parse_args()

dir_files = os.listdir(args.csv_dir)

edges = []
edges_csv_files = [file_name for file_name in dir_files if "edges" in file_name and file_name.endswith(".csv")]
print(edges_csv_files)
for edges_csv_file in edges_csv_files:
    csv_path = f"{args.csv_dir}/{edges_csv_file}"
    with open(csv_path, "r") as edges_csv:
        reader = csv.reader(edges_csv)
        next(reader)  # Skip header
        for row in reader:
            edge = {"id": len(edges), "subject": row[0], "predicate": row[1], "object": row[2]}
            edges.append(edge)

with jsonlines.open(f"{args.csv_dir}/edges.jsonl", "w") as jsonl_writer:
    jsonl_writer.write_all(edges)


nodes = []
nodes_csv_files = [file_name for file_name in dir_files if "nodes" in file_name and file_name.endswith(".csv")]
print(nodes_csv_files)
for nodes_csv_file in nodes_csv_files:
    csv_path = f"{args.csv_dir}/{nodes_csv_file}"
    with open(csv_path, "r") as nodes_csv:
        reader = csv.reader(nodes_csv)
        next(reader)  # Skip header
        for row in reader:
            node = {"id": row[0], "name": row[1], "category": row[2]}
            nodes.append(node)

with jsonlines.open(f"{args.csv_dir}/nodes.jsonl", "w") as jsonl_writer:
    jsonl_writer.write_all(nodes)

