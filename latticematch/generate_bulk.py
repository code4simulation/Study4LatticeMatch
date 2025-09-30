from ase import Atoms
from ase.io import write
from ase.build import bulk 
import numpy as np
import spglib

def generate_bulk(materials, lattice, cyrstalstructure):
    _primitive = bulk(materials, crystalstructure=crystalstructure, a=lattice)
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
    else:
        print("\nCould not obtain conventional cell using spglib.")
    write(f'POSCAR_{materials}.vasp', _conv, format='vasp')

generate_bulk('CdTe', 6.481, 'zincblende')
generate_bulk('GaAs', 5.653, 'zincblende')
