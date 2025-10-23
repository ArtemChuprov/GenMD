Run **generate_dataset.py** to generate a dataset of pair json-extyz file.
**file_analyzer.py** contains a function of backward converting extyz->json.
Example:
```
from file_analyzer import restore_json_from_extxyz
res = restore_json_from_extxyz("generated_dataset/extxyz/000.extxyz")
print(res)
```
