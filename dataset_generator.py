#!/usr/bin/env python3
"""
generate_dataset_ase_widths_pressure.py

Generates JSONs and matching .extxyz files (numbered 000, 001, ...) with:
- JSON contains: material, atomic_mass, density, temperature, lattice_type, lattice_parameter (a),
  width_x, width_y, width_z, pressure (GPa)
- Atoms placed inside axis-aligned box [0,width_x] x [0,width_y] x [0,width_z] (no absolute shifts)
- Non-graphite materials use cubic unit cells; graphite uses Graphite builder internally
- Maxwell–Boltzmann velocities, masses and metadata are written into .extxyz
"""

import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

# ASE imports
from ase.build import bulk
from ase.io import write
from ase import Atoms
from ase.lattice.hexagonal import Graphite
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# ---- User parameters (edit these) ----
N_FILES = 50  # how many JSON/.extxyz pairs to generate
OUT_ROOT = Path("./generated_dataset")
JSON_DIR = OUT_ROOT / "jsons"
XYZ_DIR = OUT_ROOT / "extxyz"
SEED = 42  # reproducibility
# --------------------------------------

random.seed(SEED)

# Physical constants and small DB
AVOGADRO = 6.02214076e23
ANGSTROM_PER_CM = 1e8
IDEAL_HCP_C_OVER_A = 1.633
GRAPHITE_C_OVER_A = 2.72

# Minimal material DB: atomic mass (g/mol), default density (g/cm3) and default lattice
MATERIAL_DB = {
    "Cu": {"atomic_mass": 63.546, "density": 8.96, "lattice": "fcc"},
    "Al": {"atomic_mass": 26.9815, "density": 2.70, "lattice": "fcc"},
    "Fe": {"atomic_mass": 55.845, "density": 7.87, "lattice": "bcc"},
    "Ag": {"atomic_mass": 107.8682, "density": 10.49, "lattice": "fcc"},
    "Au": {"atomic_mass": 196.96657, "density": 19.32, "lattice": "fcc"},
    "C": {"atomic_mass": 12.011, "density": 3.51, "lattice": "diamond"},
}

ATOMS_PER_UC = {
    "fcc": 4,
    "bcc": 2,
    "diamond": 8,
    "hcp": 6,
    "hexagonal": 4,
}


def lattice_param_from_density(
    atomic_mass_g_per_mol: float, density_g_per_cm3: float, lattice: str
) -> Tuple[float, float]:
    """Return (a_ang, c_ang_or_None)."""
    lattice_l = lattice.lower()
    if lattice_l not in ATOMS_PER_UC:
        raise ValueError(f"Unsupported lattice '{lattice}' for density->a conversion.")

    n = ATOMS_PER_UC[lattice_l]
    mass_uc_g = n * atomic_mass_g_per_mol / AVOGADRO
    vol_uc_cm3 = mass_uc_g / density_g_per_cm3  # cm^3

    if lattice_l in ("fcc", "bcc", "diamond"):
        a_cm = vol_uc_cm3 ** (1.0 / 3.0)
        return float(a_cm * ANGSTROM_PER_CM), None

    if lattice_l == "hcp":
        c_over_a = IDEAL_HCP_C_OVER_A
        factor = (math.sqrt(3) / 2.0) * c_over_a
        a_cm = (vol_uc_cm3 / factor) ** (1.0 / 3.0)
        return float(a_cm * ANGSTROM_PER_CM), float(a_cm * ANGSTROM_PER_CM * c_over_a)

    if lattice_l == "hexagonal":
        c_over_a = GRAPHITE_C_OVER_A
        factor = (math.sqrt(3) / 2.0) * c_over_a
        a_cm = (vol_uc_cm3 / factor) ** (1.0 / 3.0)
        a_ang = float(a_cm * ANGSTROM_PER_CM)
        c_ang = float(a_ang * c_over_a)
        return a_ang, c_ang

    raise RuntimeError("Unhandled lattice in lattice_param_from_density.")


def safe_bulk_build(symbol: str, structure: str, a: float, c: float = None) -> Atoms:
    """
    Build unit cell. For non-graphite structures use cubic=True to get cubic conventional cell.
    For graphite, use Graphite builder (no c parameter in JSON).
    """
    structure_l = structure.lower()
    if structure_l == "graphite" or structure_l == "hexagonal":
        # Use ASE Graphite builder (set a and c). We pick reasonable c if not provided.
        if c is None:
            c = 6.708  # typical graphite c (Å)
        return Graphite(symbol="C", latticeconstant={"a": a, "c": c})
    else:
        # For cubic materials we force cubic conventional cell where possible
        return bulk(symbol, structure_l, a=a, cubic=True)


def clean_and_make_dirs(*dirs: Path):
    for d in dirs:
        if d.exists() and d.is_dir():
            for f in d.iterdir():
                if f.is_file() or f.is_symlink():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
        else:
            d.mkdir(parents=True, exist_ok=True)


def trim_supercell_to_box(atoms: Atoms, widths: Tuple[float, float, float]) -> Atoms:
    """
    Keep atoms in [0,width_x]x[0,width_y]x[0,width_z]. Returns new Atoms with positions unchanged
    (i.e. positions assumed absolute wrt origin at 0).
    """
    wx, wy, wz = widths
    positions = atoms.get_positions()
    mask = (
        (positions[:, 0] >= 0.0)
        & (positions[:, 0] <= wx)
        & (positions[:, 1] >= 0.0)
        & (positions[:, 1] <= wy)
        & (positions[:, 2] >= 0.0)
        & (positions[:, 2] <= wz)
    )
    symbols = atoms.get_chemical_symbols()
    new_symbols = [s for s, m in zip(symbols, mask) if m]
    new_positions = positions[mask]
    new = Atoms(new_symbols, positions=new_positions)
    return new


def generate_velocities_for_atoms(atoms: Atoms, temperature_K: float):
    """Assign Maxwell-Boltzmann velocities to atoms in-place."""
    if temperature_K is None:
        return
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K)
    except Exception:
        atoms.set_velocities([[0.0, 0.0, 0.0]] * len(atoms))


def build_scene_from_dict(scene: Dict[str, Any]) -> Tuple[Dict[str, Any], Atoms]:
    """
    New expected JSON fields:
      - width_x, width_y, width_z (Angstrom)
      - pressure (GPa)
    The function uses the widths as the bounding box anchored at origin (0,0,0).
    """
    material = scene["material"]
    if material not in MATERIAL_DB:
        raise ValueError(f"Material '{material}' not in DB.")

    atomic_mass = float(scene.get("atomic_mass", MATERIAL_DB[material]["atomic_mass"]))
    density = float(scene.get("density", MATERIAL_DB[material]["density"]))
    temperature = float(scene.get("temperature", 300.0))
    pressure = float(scene.get("pressure_GPa", 0.0))
    lattice_type = scene.get("lattice_type", MATERIAL_DB[material]["lattice"])
    lattice_param = scene.get("lattice_parameter", None)
    region_widths = (
        float(scene["width_x"]),
        float(scene["width_y"]),
        float(scene["width_z"]),
    )
    wx, wy, wz = region_widths

    # compute lattice parameter a if missing (c only internal for hexagonal)
    if lattice_param is None:
        a_ang, c_ang = lattice_param_from_density(atomic_mass, density, lattice_type)
        lattice_param = a_ang
    else:
        a_ang = float(lattice_param)
        c_ang = None

    # map lattice token to builder structure
    ase_struct = lattice_type.lower()
    if ase_struct == "hexagonal":
        ase_struct_for_build = "hexagonal"
    elif ase_struct == "diamond":
        ase_struct_for_build = "diamond"
    else:
        ase_struct_for_build = ase_struct

    # build unit cell (cubic for non-hexagonal)
    symbol = material
    atoms_uc = safe_bulk_build(symbol, ase_struct_for_build, a=a_ang, c=c_ang)

    # make a supercell big enough to cover the box anchored at origin
    cell = atoms_uc.get_cell()
    cell_vec_lengths = [
        math.sqrt((cell[i][0]) ** 2 + (cell[i][1]) ** 2 + (cell[i][2]) ** 2)
        for i in range(3)
    ]
    min_cell_len = min([l for l in cell_vec_lengths if l > 1e-8])
    max_region_axis = max(wx, wy, wz)
    base_repeats = max(1, math.ceil(max_region_axis / min_cell_len))
    repeats = (base_repeats + 2, base_repeats + 2, base_repeats + 2)

    atoms_super = atoms_uc.repeat(repeats)

    # Trim to box [0,wx]x[0,wy]x[0,wz]
    atoms_trimmed = trim_supercell_to_box(atoms_super, region_widths)

    # shift trimmed atoms to be inside [0,width] (they already are since we anchored at origin)
    # set masses (g/mol = amu)
    if len(atoms_trimmed) > 0:
        masses = [atomic_mass] * len(atoms_trimmed)
        atoms_trimmed.set_masses(masses)

    # velocities
    if len(atoms_trimmed) > 0:
        generate_velocities_for_atoms(atoms_trimmed, temperature)

    atoms_trimmed.set_pbc((False, False, False))
    atoms_trimmed.set_cell([[wx, 0.0, 0.0], [0.0, wy, 0.0], [0.0, 0.0, wz]])

    # metadata
    atoms_trimmed.info["material"] = material
    atoms_trimmed.info["atomic_mass"] = atomic_mass
    atoms_trimmed.info["density"] = density
    atoms_trimmed.info["temperature"] = temperature
    atoms_trimmed.info["pressure_GPa"] = pressure
    atoms_trimmed.info["lattice_type"] = lattice_type
    atoms_trimmed.info["lattice_parameter_a"] = float(lattice_param)
    atoms_trimmed.info["width_x"] = wx
    atoms_trimmed.info["width_y"] = wy
    atoms_trimmed.info["width_z"] = wz
    atoms_trimmed.info["n_atoms"] = len(atoms_trimmed)

    # produce JSON output (no x1..x2, only widths and pressure)
    scene_out = {
        "material": material,
        "atomic_mass": atomic_mass,
        "density": density,
        "temperature": temperature,
        "pressure_GPa": pressure,
        "lattice_type": lattice_type,
        "lattice_parameter": float(lattice_param),
        "width_x": wx,
        "width_y": wy,
        "width_z": wz,
        "n_atoms": len(atoms_trimmed),
    }

    return scene_out, atoms_trimmed


def sample_random_scene_dict(
    material: str = None,
    size_range: Tuple[float, float] = (8.0, 30.0),
    density_jitter: float = 0.02,
    pressure_range_GPa: Tuple[float, float] = (0.0, 5.0),
) -> Dict[str, Any]:
    """Sample JSON that uses widths (no absolute shifts). Pressure randomly chosen in GPa."""
    if material is None:
        material = random.choice(list(MATERIAL_DB.keys()))
    db = MATERIAL_DB[material]
    atomic_mass = db["atomic_mass"]
    density = db["density"]
    lattice_default = db["lattice"]

    density_sample = float(
        density * random.uniform(1.0 - density_jitter, 1.0 + density_jitter)
    )
    temperature = float(
        random.choice([300, 500, 100, 1000])
        if random.random() < 0.3
        else random.uniform(200, 800)
    )

    if material == "C":
        lattice_type = random.choice(["diamond", "hexagonal"])
    else:
        lattice_type = lattice_default

    lattice_parameter = None

    size_x = random.uniform(*size_range)
    size_y = random.uniform(*size_range)
    size_z = random.uniform(*size_range)

    pressure = float(random.uniform(pressure_range_GPa[0], pressure_range_GPa[1]))

    scene = {
        "material": material,
        "atomic_mass": atomic_mass,
        "density": round(density_sample, 6),
        "temperature": round(temperature, 2),
        "pressure_GPa": round(pressure, 6),
        "lattice_type": lattice_type,
        "lattice_parameter": lattice_parameter,
        "width_x": round(size_x, 6),
        "width_y": round(size_y, 6),
        "width_z": round(size_z, 6),
    }
    return scene


def save_json(scene: Dict[str, Any], path: Path):
    with open(path, "w") as f:
        json.dump(scene, f, indent=2)


def save_extxyz(atoms: Atoms, path: Path):
    write(str(path), atoms, format="extxyz")


def main_generate(n_files: int):
    clean_and_make_dirs(JSON_DIR, XYZ_DIR)

    for i in range(n_files):
        idx = f"{i:03d}"
        json_path = JSON_DIR / f"{idx}.json"
        xyz_path = XYZ_DIR / f"{idx}.extxyz"

        scene = sample_random_scene_dict()
        scene_out, atoms = build_scene_from_dict(scene)

        save_json(scene_out, json_path)
        save_extxyz(atoms, xyz_path)

        print(f"Saved: {json_path.name}  ->  {xyz_path.name}  (n_atoms={len(atoms)})")


if __name__ == "__main__":
    print(
        f"Generating {N_FILES} scenes into:\n  JSONs: {JSON_DIR}\n  EXTXYZ: {XYZ_DIR}"
    )
    main_generate(N_FILES)
    print("Done.")
