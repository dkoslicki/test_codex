import json
import os
import string
from collections import defaultdict

import wget
from lxml import etree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUNCTUATION_CHAR_MAP = {ord(char): None for char in string.punctuation}
DIGITS_CHAR_MAP = {ord(char): None for char in string.digits}


def _serialize_with_sets(obj: any) -> any:
    return list(obj) if isinstance(obj, set) else obj


local_xml_file_path = f"{SCRIPT_DIR}/en_product6.xml"
if not os.path.exists(local_xml_file_path):
    print(f"Downloading orphanet disease--genes data file..")
    symptoms_data_url = "https://www.orphadata.com/data/xml/en_product6.xml"
    wget.download(symptoms_data_url, local_xml_file_path)

tree = etree.parse(local_xml_file_path)

diseases = tree.xpath("//Disorder")
print(len(diseases))

disease_to_genes = defaultdict(set)
equivalent_genes_map = dict()
print(f"Extracting genes from XML file..")
for disease in diseases:
    disease_id_num = disease.xpath(".//OrphaCode/text()")[0]
    disease_id = f"orphanet:{disease_id_num}"
    disease_name = next(child.text for child in disease if child.tag == "Name")
    print(disease_id, disease_name)

    gene_associations = disease.xpath(".//DisorderGeneAssociation")
    print(f"  --> {len(gene_associations)} gene associations")
    for gene_association in gene_associations:

        # Grab all ids provided for this gene (from different source vocabs/ontologies)
        xrefs = gene_association.xpath(".//ExternalReference")
        equivalent_gene_ids = set()
        for xref in xrefs:
            sources = xref.xpath(".//Source")
            ref_ids = xref.xpath(".//Reference")
            assert len(sources) == 1
            assert len(ref_ids) == 1
            prefix = sources[0].text
            reference_id = ref_ids[0].text
            gene_id = f"{prefix}:{reference_id}"
            equivalent_gene_ids.add(gene_id)

        # Choose which identifier to use as 'preferred' (favor HGNC, then Ensembl)
        hgnc_ids = [curie for curie in equivalent_gene_ids if curie.upper().startswith("HGNC:")]
        ensembl_ids = [curie for curie in equivalent_gene_ids if curie.upper().startswith("ENSEMBL:")]
        if hgnc_ids:
            preferred_gene_id = hgnc_ids[0]
        elif ensembl_ids:
            preferred_gene_id = ensembl_ids[0]
        else:
            print(f"    --> no HGNC or Ensembl id, randomly picking one...")
            preferred_gene_id = list(equivalent_gene_ids)[0]

        # Record our mappings
        equivalent_genes_map[preferred_gene_id] = equivalent_gene_ids
        disease_to_genes[disease_id].add(preferred_gene_id)


unique_genes = {gene_id for disease, gene_ids in disease_to_genes.items()
                for gene_id in gene_ids}
print(f"In the end, our map includes {len(disease_to_genes)} diseases, with a "
      f"total of {len(unique_genes)} (distinct) associated genes.")


with open(f"{SCRIPT_DIR}/orphanet_disease_to_genes.json", "w+") as output_file:
    json.dump(disease_to_genes, output_file, indent=2, default=_serialize_with_sets)


# Store our disease --> gene mappings in reverse format as well
gene_to_diseases = defaultdict(set)
for disease, genes in disease_to_genes.items():
    for gene in genes:
        gene_to_diseases[gene].add(disease)
with open(f"{SCRIPT_DIR}/orphanet_gene_to_diseases.json", "w+") as output_file:
    json.dump(gene_to_diseases, output_file, indent=2, default=_serialize_with_sets)


with open(f"{SCRIPT_DIR}/equivalent_genes.json", "w+") as output_file:
    json.dump(equivalent_genes_map, output_file, indent=2, default=_serialize_with_sets)
