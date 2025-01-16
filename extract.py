from json import load, dump
from hmac import new

# get cwe dictionary from https://github.com/OWASP/cwe-sdk-javascript/blob/master/raw/cwe-dictionary.json
with open('cwe-dictionary.json', 'r') as f:
    data = load(f)

new_set = []
for k in list(data.keys()):
    subkeys = data[k].keys()
    for sk in list(subkeys):
        if sk.lower() not in ('description', 'potential_mitigations'):
            del data[k][sk]
    if 'Potential_Mitigations' not in data[k].keys():
        data[k]['Potential_Mitigations'] = {"Mitigation": []}
    current = data.pop(k)
    new_dict = {"ID": f'CWE-{k}', "Description": current['Description'], "Potential_Mitigations": current['Potential_Mitigations']}
    new_set.append(new_dict)

with open('cwe_dict_clean.json', 'w') as f:
    dump(new_set, f, indent=2)
