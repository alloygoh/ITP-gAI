from xmltodict import parse
from json import dump
import csv
import re

mitigation_lookup: dict[str, str] = {}

with open("1400.csv", "r") as f:
    mitigation_reader = csv.DictReader(f)
    for row in mitigation_reader:
        groups = re.findall(r":(\w+):([^:]+)", row["Potential Mitigations"])
        formatted = ""
        preliminary = [": ".join(x) for x in groups]
        for entry in preliminary:
            if entry.startswith("PHASE") and formatted != "":
                formatted += "\n"
            formatted += entry + "\n"
        mitigation_lookup[f"CWE-{row['CWE-ID']}"] = formatted

# get cwe xml from https://github.com/OWASP/cwe-sdk-javascript/blob/master/raw/cwe-archive.xml
with open("cwe-archive.xml", "r") as f:
    data = f.read()

cwe_archive = parse(data)

cwe_info = cwe_archive["Weakness_Catalog"]["Weaknesses"]["Weakness"]

lookup = {}

for k in cwe_info:
    mitigation = mitigation_lookup.get(f"CWE-{k['@ID']}", "")
    extended_description = (
        "" if k.get("Extended_Description") is None else k["Extended_Description"]
    )
    id = f"CWE-{k['@ID']}"
    cwe_entry = {
        "ID": id,
        "Description": f"{k['@Name']}. {k['Description']}. {extended_description}",
        "Potential_Mitigations": mitigation,
    }
    lookup[id] = cwe_entry


with open("cwe_dict_clean.json", "w") as f:
    dump(list(lookup.values()), f, indent=2)


# get from https://cwe.mitre.org/data/definitions/1400.html
with open("1400.xml", "r") as f:
    data = f.read()

cwe_dict = parse(data)["Weakness_Catalog"]

categories = []
for cat in cwe_dict["Categories"]["Category"]:
    members = []
    for member in cat["Relationships"]["Has_Member"]:
        cwe_id = f"CWE-{member['@CWE_ID']}"
        lookup_result = lookup[cwe_id]
        members.append(lookup_result)
    category_id = f"CWE-{cat['@ID']}"
    entry = {
        "category_id": category_id,
        "category_name": cat["@Name"],
        "summary": cat["Summary"],
        "members": members,
    }
    categories.append(entry)

with open("cwe_view_mapping.json", "w") as f:
    dump(categories, f, indent=2)
