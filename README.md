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

>>>
{
  "material": "C",
  "atomic_mass": 12.011,
  "density": 3.455431,
  "temperature": 346.94,
  "pressure_GPa": 0.434694,
  "lattice_type": "diamond",
  "lattice_parameter": 3.5876109603706006,
  "width_x": 24.202367,
  "width_y": 22.887389,
  "width_z": 27.62795,
  "n_atoms": 2724
}
```


