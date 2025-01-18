from xmltodict import parse
from json import dump

# get cwe xml from https://github.com/OWASP/cwe-sdk-javascript/blob/master/raw/cwe-archive.xml
with open("cwe-archive.xml", "r") as f:
    data = f.read()

cwe_dict = parse(data)
cwe_info_sorted = sorted(
    cwe_dict["Weakness_Catalog"]["Weaknesses"]["Weakness"], key=lambda x: int(x["@ID"])
)

new_set = []

for k in cwe_info_sorted:
    mitigation = (
        [] if k.get("Potential_Mitigations") is None else k["Potential_Mitigations"]
    )
    extended_description = (
        "" if k.get("Extended_Description") is None else k["Extended_Description"]
    )
    cwe_entry = {
        "ID": f"CWE-{k['@ID']}",
        "Description": f"{k['@Name']}. {k['Description']}. {extended_description}",
        "Potential_Mitigations": mitigation,
    }
    new_set.append(cwe_entry)

with open("cwe_dict_clean.json", "w") as f:
    dump(new_set, f, indent=2)
