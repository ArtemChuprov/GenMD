# GenMD project
This project aims to create a system for careful analysis and interpretation of the MD.

## Tools details
Run **generate_dataset.py** to generate a dataset of pair json-extyz file.

**file_analyzer.py** contains a function of backward converting extyz->json.
Example usage:
```
from file_analyzer import restore_json_from_extxyz
res = restore_json_from_extxyz("generated_dataset/extxyz/000.extxyz")
print(res)
```
