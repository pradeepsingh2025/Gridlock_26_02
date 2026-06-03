import json

with open('traffic_demand_pred.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code = []
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            source = ''.join(source)
        code.append(source)

with open('notebook_runner_correct.py', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(code))
