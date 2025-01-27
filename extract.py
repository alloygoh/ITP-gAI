from xmltodict import parse
from json import dump

# get cwe xml from https://github.com/OWASP/cwe-sdk-javascript/blob/master/raw/cwe-archive.xml
with open("cwe-archive.xml", "r") as f:
    data = f.read()

cwe_archive = parse(data)

cwe_info = cwe_archive["Weakness_Catalog"]["Weaknesses"]["Weakness"]

lookup = {}

for k in cwe_info:
    mitigation = (
        [] if k.get("Potential_Mitigations") is None else k["Potential_Mitigations"]
    )
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

# get from https://cwe.mitre.org/data/definitions/1400.html
with open("1400.xml", "r") as f:
    data = f.read()

cwe_dict = parse(data)["Weakness_Catalog"]

# print(cwe_dict.keys())
# print(cwe_dict["Categories"]["Category"][0].keys())

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

# with open("cwe_dict_clean.json", "w") as f:
#     dump(new_set, f, indent=2)
