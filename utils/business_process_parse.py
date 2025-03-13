import pylightxl as xl
from pylightxl.pylightxl import Worksheet
from json import dump


def get_ws_as_string(ws: Worksheet) -> str:
    out: list[str] = []
    for row in ws.rows:
        # skip header
        if not str(row[0]).isnumeric():
            continue
        line = f"{row[2]}: {row[6]}"
        out.append(line)
    return "\n".join(out)


data = []

db = xl.readxl("process-classification.xlsx")

catagories = db.ws("Categories")
for row in catagories.rows:
    if not str(row[0]).isnumeric():
        continue
    hierachy_id = row[1]
    details = db.ws(str(hierachy_id))
    description = get_ws_as_string(details)
    entry = {"ID": row[0], "Name": row[2], "Description": description}
    data.append(entry)

with open("process_categories.json", "w") as f:
    dump(data, f, indent=2)
