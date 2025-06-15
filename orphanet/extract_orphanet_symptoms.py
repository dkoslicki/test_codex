import json
import os
import string

import wget
from lxml import etree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUNCTUATION_CHAR_MAP = {ord(char): None for char in string.punctuation}
DIGITS_CHAR_MAP = {ord(char): None for char in string.digits}


local_xml_file_path = f"{SCRIPT_DIR}/en_product4.xml"
if not os.path.exists(local_xml_file_path):
    print(f"Downloading orphanet disease--symptom frequencies data file..")
    symptoms_data_url = "https://www.orphadata.com/data/xml/en_product4.xml"
    wget.download(symptoms_data_url, local_xml_file_path)

tree = etree.parse(local_xml_file_path)
diseases = tree.xpath("//Disorder")
print(len(diseases))

disease_to_symptom_frequencies = dict()
frequency_raws = set()
print(f"Extracting frequencies from XML file..")
for disease in diseases:
    disease_id_num = disease.xpath(".//OrphaCode/text()")[0]
    disease_id = f"orphanet:{disease_id_num}"
    disease_name = next(child.text for child in disease if child.tag == "Name")
    print(disease_id, disease_name)
    disease_to_symptom_frequencies[disease_id] = dict()

    symptom_associations = disease.xpath(".//HPODisorderAssociation")
    print(f"  --> {len(symptom_associations)} symptom associations")
    for symptom_association in symptom_associations:
        symptom_id = symptom_association.xpath(".//HPOId/text()")[0]
        symptom_name = symptom_association.xpath(".//HPOTerm/text()")[0]
        frequency_raw = symptom_association.xpath(".//HPOFrequency/Name/text()")[0]
        frequency_raws.add(frequency_raw)
        frequency_stripped = frequency_raw.lower().translate(PUNCTUATION_CHAR_MAP).translate(DIGITS_CHAR_MAP).strip()
        frequency_id = frequency_stripped.replace(" ", "_")
        print(f"    {symptom_id} ({symptom_name}): {frequency_id}")
        if frequency_id != "excluded":
            disease_to_symptom_frequencies[disease_id][symptom_id] = frequency_id

unique_symptoms = {symptom_id for disease, symptoms in disease_to_symptom_frequencies.items()
                   for symptom_id in symptoms}
print(f"In the end, our map includes {len(disease_to_symptom_frequencies)} diseases, with a "
      f"total of {len(unique_symptoms)} (distinct) associated symptoms.")
symptom_frequency_values = {frequency_id for disease, symptoms in disease_to_symptom_frequencies.items()
                            for frequency_id in symptoms.values()}
print(f"Unique frequency values are: {sorted(symptom_frequency_values)}")
print(f"Raw text versions of these were: {sorted(frequency_raws)}")


with open(f"{SCRIPT_DIR}/orphanet_disease_to_symptoms.json", "w+") as output_file:
    json.dump(disease_to_symptom_frequencies, output_file, indent=2)
