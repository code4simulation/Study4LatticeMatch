# Check the result of (110) surface.
from ase import Atoms
from ase.build import bulk, surface
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

if __name__ == "__main__":
    # 1. CdTe Zinc Blende 벌크 구조 생성 (primitive cell)
    a = 6.48  # CdTe 격자 상수 (Å)
    _primitive = bulk('CdTe', crystalstructure='zincblende', a=a)
    print("Primitive cell:")
    print(_primitive)
    
    # 2. Primitive cell을 conventional cell (fcc)로 변환
    # Use spglib.standardize_cell to get the conventional cell
    lattice, positions, numbers = _primitive.cell.array, _primitive.get_scaled_positions(), _primitive.get_atomic_numbers()
    converted = spglib.standardize_cell((lattice, positions, numbers), to_primitive=False, symprec=1e-5)
    print(converted)
    
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
    
    # 2. (100) 표면 슬랩 생성
    #    layers: 4 (원자층 수), vacuum: 10 Å
    slab = surface(_conv, indices=(1,0,0), layers=4, periodic=True)
    slab.center(vacuum=10.0, axis=2)
    slab = slab[slab.numbers.argsort()]
    print(slab)
    
    # 3. ASE Atoms 객체를 spglib 형식으로 변환
    lattice = slab.get_cell().array
    positions = slab.get_scaled_positions()
    numbers   = slab.get_atomic_numbers()
    
    # 4. spglib을 사용해 원시셀(primitive slab) 찾기
    cell = (lattice, positions, numbers)
    primitive = spglib.find_primitive(cell, symprec=1e-5)
    
    if primitive is None:
        raise RuntimeError("원시셀을 찾지 못했습니다. symprec 값을 조정해보세요.")
    
    primitive_lattice, primitive_positions, primitive_numbers = primitive
    
    # 5. 결과 출력
    print("=== 원시 슬랩 구조 ===")
    print("격자 벡터:")
    for vec in primitive_lattice:
        print(f"  [{vec[0]:8.3f}, {vec[1]:8.3f}, {vec[2]:8.3f}]")
    
    # 6. 필요 시 ASE Atoms로 재생성
    primitive_atoms = Atoms(
        numbers=primitive_numbers,
        scaled_positions=primitive_positions,
        cell=primitive_lattice,
        pbc=[True, True, True]
    )
    print(primitive_atoms)

    res = generate_primitive_slab(_conv, (1, 0, 0), layers=4, vacuum=10.0)
    print(res)
    
    res = generate_primitive_slab(_conv, (1, 1, 0), layers=4, vacuum=10.0, symprec=1e-5)
    print(res)
    slab = surface(_conv, indices=(1,1,0), layers=3, periodic=True)
    slab.center(vacuum=10, axis=2)
    print(slab)   
    
    res = generate_primitive_slab(_conv, (1, 1, 1), layers=4, vacuum=10.0)
    print(res)
