"""
ase_scene_generator_chunked.py

Simple chunked multiprocessing generator. Each worker gets a list of indices and processes them.
This updated version **removes any existing files** in OUTDIR/xyz and OUTDIR/meta at the start of the run.

Edit CONFIG at the top and run `python ase_scene_generator_chunked.py`.

WARNING: The cleaning step deletes files in OUTDIR/xyz and OUTDIR/meta without confirmation. Back up anything important.
"""

import os
import json
import random
import math
import time
import shutil
from pathlib import Path
from multiprocessing import Process
import numpy as np
from ase import Atoms
from ase.build import bulk, surface
from ase.io import write

# ----------------------------- CONFIG ----------------------------------------
CONFIG = {
    'OUTDIR': 'dataset_chunked',
    'N_SAMPLES': 2000,      # total number of scenes
    'N_WORKERS': 5,        # number of processes; each gets ~N_SAMPLES/N_WORKERS indices
    'SEED_BASE': 1234,     # base seed; reproducible if unchanged
    'ADD_NOISE': True,
    'NOISE_SIGMA': 0.01,   # Å
    # scene ranges (tune to avoid huge generation times)
    'RANGES': {
        'bulk_repeat': (3, 7),
        'bulk_scale': (0.9, 1.3),
        'slab_size': (2, 5),
        'slab_layers': (3, 10),
        'slab_vacuum': (12.0, 60.0),
        'nanoparticle_radius': (30, 90.0),
        'amorph_n_atoms': (120, 700),
        'amorph_box': (60.0, 180.0),
    }
}

# ----------------------------- MATERIALS ------------------------------------
MATERIALS = {
    'Cu': {'a': 3.615, 'structure': 'fcc'},
    'Al': {'a': 4.05,  'structure': 'fcc'},
    'Au': {'a': 4.08,  'structure': 'fcc'},
    'Si': {'a': 5.431, 'structure': 'diamond'},
    'Fe': {'a': 2.866, 'structure': 'bcc'},
}

# ----------------------------- HELPERS --------------------------------------
def rand_int(rng): return random.randint(int(rng[0]), int(rng[1]))
def rand_float(rng): return random.uniform(float(rng[0]), float(rng[1]))

# ----------------------------- SCENE GENERATORS ------------------------------
# These are simple and self-contained. Worker will copy base structures and use them.
def build_base_structures():
    """Build primitive cells once (called inside each worker)."""
    base = {}
    for mat, info in MATERIALS.items():
        try:
            base[mat] = bulk(mat, crystalstructure=info['structure'], a=info['a'])
        except Exception:
            base[mat] = Atoms([mat])
    return base

def generate_bulk(material, lat_a, scale_factors, repeat=(2,2,2)):
    """
    Build a conventional orthogonal fcc supercell:
      - material: e.g. 'Al'
      - lat_a: lattice constant (Å)
      - scale_factors: (sx,sy,sz) scale for each axis (keeps orthogonality)
      - repeat: (nx,ny,nz) repeats of the conventional cell
    Returns: ASE Atoms and params dict
    """
    nx, ny, nz = int(repeat[0]), int(repeat[1]), int(repeat[2])
    sx, sy, sz = float(scale_factors[0]), float(scale_factors[1]), float(scale_factors[2])

    # conventional fcc basis (fractional coordinates in conventional cubic cell)
    basis_frac = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])

    # real-space cell lengths per repeat (orthogonal)
    a_x = lat_a * sx
    a_y = lat_a * sy
    a_z = lat_a * sz

    positions = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # offset of conventional cell origin
                origin = np.array([i * a_x, j * a_y, k * a_z])
                for bf in basis_frac:
                    pos = origin + np.array([bf[0]*a_x, bf[1]*a_y, bf[2]*a_z])
                    positions.append(pos)

    atoms = Atoms([material] * len(positions))
    atoms.set_positions(np.array(positions))
    atoms.set_cell(np.diag([a_x * nx, a_y * ny, a_z * nz]))
    # do not enable random noise here — keep perfect for verification

    params = {
        'type': 'bulk',
        'material': material,
        'lattice_parameter': lat_a,
        'scale_factors': [sx, sy, sz],
        'repeat': [nx, ny, nz],
        'cell': atoms.get_cell().tolist(),
        'n_atoms': len(atoms),
    }
    return atoms, params

def generate_slab(base_structures, material, lat_a, miller=(1,0,0), size=(3,3), layers=4, vacuum=10.0):
    bul = base_structures[material].copy()
    sl = surface(bul, miller, layers)
    sl = sl.repeat((size[0], size[1], 1))
    cell = sl.get_cell().copy()
    cell[2,2] = cell[2,2] + vacuum
    sl.set_cell(cell, scale_atoms=False)
    pos = sl.get_positions(); pos[:,2] += vacuum/2.0; sl.set_positions(pos)
    thickness = float(max(pos[:,2]) - min(pos[:,2]))
    params = {'type':'slab','material':material,'miller':tuple(miller),'thickness':thickness,'cell':sl.get_cell().tolist(),'n_atoms':len(sl),'vacuum':vacuum}
    return sl, params

def generate_nanoparticle(base_structures, material, lat_a, radius, margin=3.0):
    bul = base_structures[material].copy()
    cell_size = 2*(radius+margin)
    approx_n = max(3, math.ceil(cell_size/lat_a))
    sup = bul.repeat((approx_n,approx_n,approx_n))
    pos = sup.get_positions(); cell = sup.get_cell(); center = cell.diagonal()/2.0
    dists = np.linalg.norm(pos-center,axis=1); mask = dists<=radius
    cluster = sup[mask]
    if len(cluster)==0: cluster = sup[:min(50,len(sup))]
    box = 2*(radius+2.0); cluster.set_cell(np.diag([box,box,box]), scale_atoms=False)
    cpos = cluster.get_positions(); com = cpos.mean(axis=0); shift = np.array([box/2,box/2,box/2]) - com
    cluster.set_positions(cpos+shift)
    params = {'type':'nanoparticle','material':material,'radius':float(radius),'box':float(box),'n_atoms':len(cluster),'cell':cluster.get_cell().tolist()}
    return cluster, params

def generate_amorphous(base_structures, material, n_atoms, box_lengths, min_dist=0.8, max_tries=200000):
    """
    Fast amorphous generator using a spatial hash / cell-list.
    Average complexity ~O(n); deterministic w.r.t RNG seed.
    """
    a, b, c = box_lengths
    # choose cell size slightly larger than min_dist
    cell_size = float(min_dist)
    nx = max(1, int(math.ceil(a / cell_size)))
    ny = max(1, int(math.ceil(b / cell_size)))
    nz = max(1, int(math.ceil(c / cell_size)))

    grid = {}            # dict: (ix,iy,iz) -> list of point indices
    positions = []       # list of np.array points
    tries = 0

    while len(positions) < n_atoms and tries < max_tries:
        tries += 1
        pos = np.array([random.random() * a, random.random() * b, random.random() * c])

        ix = int(pos[0] // cell_size)
        iy = int(pos[1] // cell_size)
        iz = int(pos[2] // cell_size)

        ok = True
        # check neighbors in the 3x3x3 surrounding cells
        for dx in (-1, 0, 1):
            nx_i = ix + dx
            if nx_i < 0 or nx_i >= nx:
                continue
            for dy in (-1, 0, 1):
                ny_i = iy + dy
                if ny_i < 0 or ny_i >= ny:
                    continue
                for dz in (-1, 0, 1):
                    nz_i = iz + dz
                    if nz_i < 0 or nz_i >= nz:
                        continue
                    key = (nx_i, ny_i, nz_i)
                    if key in grid:
                        for j in grid[key]:
                            if np.linalg.norm(pos - positions[j]) < min_dist:
                                ok = False
                                break
                        if not ok:
                            break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            idx = len(positions)
            positions.append(pos)
            key0 = (ix, iy, iz)
            if key0 in grid:
                grid[key0].append(idx)
            else:
                grid[key0] = [idx]

    if len(positions) < n_atoms:
        raise RuntimeError(f"Could not place {n_atoms} atoms with min_dist {min_dist} (placed {len(positions)}) after {tries} tries)")

    atoms = Atoms([material] * n_atoms)
    atoms.set_positions(np.array(positions))
    atoms.set_cell(np.diag([a, b, c]))
    params = {'type': 'amorphous', 'material': material, 'n_atoms': n_atoms, 'cell': atoms.get_cell().tolist()}
    return atoms, params

# simple one-line description builder
def make_description(params):
    t = params.get('type','unk')
    if t=='bulk':
        a=np.linalg.norm(params['cell'][0]); b=np.linalg.norm(params['cell'][1]); c=np.linalg.norm(params['cell'][2]); v=a*b*c
        return f"It is {params.get('material','X')} bulk; cell: {a:.3f}x{b:.3f}x{c:.3f} Å (V={v:.3f})."
    if t=='slab':
        return f"{params.get('material','X')} slab {''.join(map(str,params.get('miller',(0,0,0))))} thickness {params.get('thickness',0):.3f} Å."
    if t=='nanoparticle':
        return f"{params.get('material','X')} nanoparticle radius {params.get('radius',0):.3f} Å, atoms~{params.get('n_atoms')}."
    if t=='amorphous':
        return f"Amorphous-like {params.get('material','X')} with {params.get('n_atoms')} atoms."
    return "Unknown."

# ----------------------------- WORKER ------------------------------------------------
def worker_loop(indices_list, worker_id, config):
    """
    Each worker:
      - builds base structures once,
      - loops over indices_list and generates + writes scenes.
    """
    print(f"[W{worker_id}] starting, {len(indices_list)} indices (e.g. {indices_list[:5]}...)")
    # Build base structures once (preprocessing step)
    base_structures = build_base_structures()
    outdir = Path(config['OUTDIR'])
    xyz_dir = outdir / 'xyz'; meta_dir = outdir / 'meta'
    xyz_dir.mkdir(parents=True, exist_ok=True); meta_dir.mkdir(parents=True, exist_ok=True)

    for local_count, idx in enumerate(indices_list):
        # per-sample deterministic seed
        seed = config['SEED_BASE'] + idx
        random.seed(seed); np.random.seed(seed % (2**32-1))
        sid = f"sample_{idx:06d}"
        try:
            material = random.choice(list(MATERIALS.keys()))
            lat = MATERIALS[material]['a']
            scene_types = ['bulk','slab','nanoparticle','amorphous']
            t = random.choices(scene_types, weights=[0.35,0.25,0.2,0.2])[0]
            ranges = config['RANGES']

            if t == 'bulk':
                sx = rand_float((ranges['bulk_scale'][0], ranges['bulk_scale'][1]))
                sy = rand_float((ranges['bulk_scale'][0], ranges['bulk_scale'][1]))
                sz = rand_float((ranges['bulk_scale'][0], ranges['bulk_scale'][1]))
                rep_val = rand_int((ranges['bulk_repeat'][0], ranges['bulk_repeat'][1]))
                rep = (rep_val, rep_val, rep_val)
                atoms, params = generate_bulk(material, lat, (sx,sy,sz), repeat=rep)
            # elif t == 'slab':
            #     miller = random.choice([(1,0,0),(1,1,0),(1,1,1)])
            #     nx = rand_int((ranges['slab_size'][0], ranges['slab_size'][1]))
            #     ny = rand_int((ranges['slab_size'][0], ranges['slab_size'][1]))
            #     layers = rand_int((ranges['slab_layers'][0], ranges['slab_layers'][1]))
            #     vacuum = rand_float((ranges['slab_vacuum'][0], ranges['slab_vacuum'][1]))
            #     atoms, params = generate_slab(base_structures, material, lat, miller=miller, size=(nx,ny), layers=layers, vacuum=vacuum)
            elif t == 'nanoparticle':
                radius = rand_float((ranges['nanoparticle_radius'][0], ranges['nanoparticle_radius'][1]))
                atoms, params = generate_nanoparticle(base_structures, material, lat, radius)
            # else:  # amorphous
            #     n_atoms = rand_int((ranges['amorph_n_atoms'][0], ranges['amorph_n_atoms'][1]))
            #     side = rand_float((ranges['amorph_box'][0], ranges['amorph_box'][1]))
            #     atoms, params = generate_amorphous(base_structures, material, n_atoms, (side,side,side))

            # noise (optional)
            if config['ADD_NOISE']:
                pos = atoms.get_positions()
                pos += np.random.normal(scale=config['NOISE_SIGMA'], size=pos.shape)
                atoms.set_positions(pos)

            # write extended xyz and metadata JSON
            atoms.info['id'] = sid
            xyz_path = xyz_dir / (sid + '.extxyz')
            write(str(xyz_path), atoms, format='extxyz')

            meta = {'id':sid, 'params':params, 'description': make_description(params)}
            with open(meta_dir / (sid + '.json'), 'w') as f:
                json.dump(meta, f, separators=(',',':'))
            # time.sleep(0.1)
            if local_count % 50 == 0:
                print(f"[W{worker_id}] wrote {sid} (local count {local_count})")

        except Exception as e:
            print(f"[W{worker_id}] ERROR idx={idx} : {e}")

    print(f"[W{worker_id}] finished (processed {len(indices_list)} samples)")

# ----------------------------- MAIN ------------------------------------------------
def main():
    cfg = CONFIG
    n = cfg['N_SAMPLES']
    k = max(1, int(cfg['N_WORKERS']))

    # Create list of indices
    all_indices = list(range(n))

    # Round-robin assignment: each worker gets every k-th index
    procs = []

    # ---- CLEAN OUTPUT DIRECTORIES BEFORE STARTING (user requested) ----
    outdir = Path(cfg['OUTDIR'])
    xyz_dir = outdir / 'xyz'
    meta_dir = outdir / 'meta'

    # Remove directories safely if they exist, then recreate empty ones
    if xyz_dir.exists():
        try:
            shutil.rmtree(xyz_dir)
            print(f"Removed existing directory: {xyz_dir}")
        except Exception as e:
            print(f"Warning: failed to remove {xyz_dir}: {e}")
    if meta_dir.exists():
        try:
            shutil.rmtree(meta_dir)
            print(f"Removed existing directory: {meta_dir}")
        except Exception as e:
            print(f"Warning: failed to remove {meta_dir}: {e}")

    # Recreate clean directories
    xyz_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    for worker_id in range(k):
        indices_for_worker = all_indices[worker_id::k]   # round-robin assignment
        p = Process(target=worker_loop, args=(indices_for_worker, worker_id, cfg))
        p.start()
        procs.append(p)

    # wait for all
    for p in procs:
        p.join()

    print("All workers done.")

if __name__ == '__main__':
  start = time.time()
  main()
  end = time.time()
  print(f'Elapsed time {end-start} seconds')
