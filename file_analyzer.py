from pathlib import Path
from ase.io import read
import numpy as np

AVOGADRO = 6.02214076e23
ANG3_TO_CM3 = 1e-24

def _safe_get_info(atoms, key):
    return atoms.info.get(key) if atoms.info and key in atoms.info else None

def _compute_density_from_atoms(atoms, atomic_mass_g_per_mol=None):
    n = len(atoms)
    if n == 0:
        return None
    # atomic_mass: prefer provided, otherwise average of atoms.get_masses()
    if atomic_mass_g_per_mol is None:
        masses = atoms.get_masses()
        if masses is None or len(masses) == 0:
            return None
        # masses typically in amu (g/mol)
        atomic_mass_g_per_mol = float(np.mean(masses))
    # total mass in grams
    mass_total_g = n * atomic_mass_g_per_mol / AVOGADRO
    # volume in Å^3
    try:
        vol_A3 = float(atoms.get_volume())
    except Exception:
        # if no cell, try bounding box
        pos = atoms.get_positions()
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        vol_A3 = float(np.prod(maxs - mins))
        if vol_A3 <= 0:
            return None
    vol_cm3 = vol_A3 * ANG3_TO_CM3
    return mass_total_g / vol_cm3

def _infer_lattice_a_from_cell(atoms):
    """Try to infer a lattice parameter 'a' from the cell vectors (simple heuristic)."""
    try:
        cell = atoms.get_cell().array  # (3,3)
    except Exception:
        return None
    lengths = np.linalg.norm(cell, axis=1)  # three vector lengths
    # If roughly cubic (all similar), return their average
    if np.all(np.isfinite(lengths)):
        if np.std(lengths) / (np.mean(lengths) + 1e-12) < 0.05:
            return float(np.mean(lengths))
        # otherwise return the smallest nonzero vector (common for hexagonal a = length[0])
        nonzero = lengths[lengths > 1e-8]
        if len(nonzero) > 0:
            return float(nonzero[0])
    return None

def restore_json_from_extxyz(path, include_n_atoms=False):
    """
    Read an .extxyz file and reconstruct a lightweight JSON-like dict with
    only the fields you want (no positions/lists).
    Returns dict with keys chosen from:
      'material', 'atomic_mass', 'density', 'temperature',
      'pressure_GPa', 'lattice_type', 'lattice_parameter',
      'width_x', 'width_y', 'width_z' (only present when available/inferred).
    If include_n_atoms=True, also include 'n_atoms'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")

    atoms = read(str(p), format="extxyz")

    out = {}

    # 1) Straight from header info if present (preferred)
    info = atoms.info if atoms.info is not None else {}

    # textual fields
    if "material" in info:
        out["material"] = str(info["material"])
    if "lattice_type" in info:
        out["lattice_type"] = str(info["lattice_type"])

    # numeric fields
    if "atomic_mass" in info:
        out["atomic_mass"] = float(info["atomic_mass"])
    if "density" in info:
        out["density"] = float(info["density"])
    if "temperature" in info:
        out["temperature"] = float(info["temperature"])
    # pressure may be saved as 'pressure_GPa' or 'pressure'
    if "pressure_GPa" in info:
        out["pressure_GPa"] = float(info["pressure_GPa"])
    elif "pressure" in info:
        out["pressure_GPa"] = float(info["pressure"])  # assume it's GPa if that's what you stored

    # widths: prefer explicit values in info, else compute from cell
    # width_x/width_y/width_z correspond to box lengths along cell axes
    if "width_x" in info and "width_y" in info and "width_z" in info:
        out["width_x"] = float(info["width_x"])
        out["width_y"] = float(info["width_y"])
        out["width_z"] = float(info["width_z"])
    else:
        # try to infer from the cell vectors (if present)
        try:
            cell = atoms.get_cell().array
            lengths = np.linalg.norm(cell, axis=1)
            if np.all(np.isfinite(lengths)) and np.prod(lengths) > 0:
                out["width_x"], out["width_y"], out["width_z"] = [float(l) for l in lengths]
        except Exception:
            pass

    # lattice parameter: first prefer explicit header, else try to infer from cell
    if "lattice_parameter_a" in info:
        out["lattice_parameter"] = float(info["lattice_parameter_a"])
    else:
        a_guess = _infer_lattice_a_from_cell(atoms)
        if a_guess is not None:
            out["lattice_parameter"] = float(a_guess)

    # density: if not in header, compute from atoms (needs atomic_mass somewhere)
    if "density" not in out:
        # determine atomic mass
        atomic_mass = out.get("atomic_mass", None)
        if atomic_mass is None:
            # try from info or from atoms.get_masses()
            if "atomic_mass" in info:
                atomic_mass = float(info["atomic_mass"])
            else:
                masses = atoms.get_masses()
                if masses is not None and len(masses) > 0:
                    atomic_mass = float(np.mean(masses))
        rho = _compute_density_from_atoms(atoms, atomic_mass_g_per_mol=atomic_mass)
        if rho is not None:
            out["density"] = float(rho)

    # include n_atoms only if asked (user requested minimal fields)
    if include_n_atoms:
        out["n_atoms"] = int(len(atoms))

    # Keep keys ordered consistently (optional)
    # desired_keys = ["material","atomic_mass","density","temperature","pressure_GPa",
    #                 "lattice_type","lattice_parameter","width_x","width_y","width_z"]
    # out = {k: out[k] for k in desired_keys if k in out}

    return out