from ase import Atoms
from ase.io import write
from ase.build import bulk, surface, niggli_reduce
import numpy as np
import spglib

def generate_primitive_slab(bulk_atoms, miller_index, layers=4, vacuum=8.0, symprec=1e-5):
    """
    Generate a reduced (primitive) surface slab from a bulk structure.

    Parameters
    ----------
    bulk_atoms : ase.Atoms
        ASE Atoms object representing the bulk conventional cell.
    miller_index : tuple of int
        Miller indices (h, k, l) defining the surface orientation.
    layers : int, optional
        Number of atomic layers in the slab (default: 4).
    vacuum : float, optional
        Thickness of vacuum layer along the surface normal in Å (default: 8.0).
    symprec : float, optional
        Symmetry tolerance for spglib operations (default: 1e-5).

    Returns
    -------
    primitive_slab : ase.Atoms
        ASE Atoms object of the reduced (primitive) surface slab.
    """
    # 1. Build the surface slab from the bulk structure
    slab = surface(bulk_atoms, indices=miller_index, layers=layers, periodic=True)
    slab.center(vacuum=vacuum, axis=2)

    # 2. Extract lattice, positions, and atomic numbers for symmetry analysis
    lattice = slab.get_cell().array
    positions = slab.get_scaled_positions()
    numbers = slab.get_atomic_numbers()

    # 3. Use spglib to find the primitive cell of the slab
    cell = (lattice, positions, numbers)
    primitive = spglib.find_primitive(cell, symprec=symprec)
    if primitive is None:
        raise RuntimeError("Could not find primitive slab. Try adjusting symprec.")

    primitive_lattice, primitive_positions, primitive_numbers = primitive

    # 4. Reconstruct ASE Atoms object for the primitive slab
    primitive_slab = Atoms(
        numbers=primitive_numbers,
        scaled_positions=primitive_positions,
        cell=primitive_lattice,
        pbc=[True, True, True]
    )
    primitive_slab = primitive_slab[primitive_slab.numbers.argsort()]
    return primitive_slab

def reduce_2d_lattice(lattice_matrix):
    """
    Implements the Zur-McGill lattice reduction algorithm.

    Parameters:
    lattice_matrix: A 2x2 numpy array representing the lattice basis vectors as columns.

    Returns:
    reduced_a, reduced_b: Reduced vectors
    steps: Steps of the reduction process
    """
    # Extract basis vectors from the matrix (assuming columns are vectors)
    vec_a = np.array(lattice_matrix[:, 0], dtype=float)
    vec_b = np.array(lattice_matrix[:, 1], dtype=float)
    print(vec_a)
    print(vec_b)

    step_count = 0
    max_iterations = 100 # Prevent infinite loops

    while step_count < max_iterations:
        step_count += 1
        changed = False

        # Condition 1: If |a| > |b|, swap a and b
        len_a = np.linalg.norm(vec_a)
        len_b = np.linalg.norm(vec_b)

        if len_a > len_b:
            vec_a, vec_b = vec_b.copy(), vec_a.copy()
            changed = True
            continue

        # Condition 2: If |b - a| < |b|, replace b with b-a
        new_b = vec_b - vec_a
        if np.linalg.norm(new_b) < np.linalg.norm(vec_b):
            vec_b = new_b.copy()
            changed = True
            continue

        # Condition 3: If |b + a| < |b|, replace b with b+a
        new_b = vec_b + vec_a
        if np.linalg.norm(new_b) < np.linalg.norm(vec_b):
            vec_b = new_b.copy()
            changed = True
            continue

        # Condition 4: If a·b < 0, replace b with -b
        if np.dot(vec_a, vec_b) < 0:
            vec_b = -vec_b
            changed = True
            continue

        # Terminate if no further reduction is possible
        if not changed:
            break

    if step_count >= max_iterations:
        print("Warning: Maximum iterations reached.")
    print(vec_a)
    print(vec_b)
    lattice_matrix = np.vstack((vec_a, vec_b))
    return lattice_matrix


if __name__ == "__main__":
    a = 5.653 
    _primitive = bulk('GaAs', crystalstructure='zincblende', a=a)
    print("Primitive cell:")
    print(_primitive)
    
    lattice, positions, numbers = _primitive.cell.array, _primitive.get_scaled_positions(), _primitive.get_atomic_numbers()
    converted = spglib.standardize_cell((lattice, positions, numbers), to_primitive=False, symprec=1e-5)
    
    if converted is not None:
        conventional_lattice, conventional_positions, conventional_numbers = converted
        _conv = Atoms(
            numbers=conventional_numbers,
            scaled_positions=conventional_positions,
            cell=conventional_lattice,
            pbc=[True, True, True]
        )
        _conv = _conv[_conv.numbers.argsort()]
        print("\nConventional cell (fcc):")
        print(_conv)
    else:
        print("\nCould not obtain conventional cell using spglib.")
    write('POSCAR_GaAs.vasp', _conv, format='vasp')
    
    a = 6.481
    _primitive = bulk('CdTe', crystalstructure='zincblende', a=a)
    print("Primitive cell:")
    print(_primitive)
    
    lattice, positions, numbers = _primitive.cell.array, _primitive.get_scaled_positions(), _primitive.get_atomic_numbers()
    converted = spglib.standardize_cell((lattice, positions, numbers), to_primitive=False, symprec=1e-5)
    
    if converted is not None:
        conventional_lattice, conventional_positions, conventional_numbers = converted
        _conv = Atoms(
            numbers=conventional_numbers,
            scaled_positions=conventional_positions,
            cell=conventional_lattice,
            pbc=[True, True, True]
        )
        _conv = _conv[_conv.numbers.argsort()]
        print("\nConventional cell (fcc):")
        print(_conv)
    else:
        print("\nCould not obtain conventional cell using spglib.")
    write('POSCAR_CdTe.vasp', _conv, format='vasp')
    
