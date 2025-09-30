# Reproduce results in Zur-McGill paper (1984)
import numpy as np
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import CoherentInterfaceBuilder
from pymatgen.analysis.structure_matcher import StructureMatcher
import warnings
warnings.filterwarnings('ignore')

def main():
    CdTe = Structure.from_file('POSCAR_CdTe.vasp')
    GaAs = Structure.from_file('POSCAR_GaAs.vasp')
    #TiN = Structure.from_file('POSCAR_TiN.vasp')
    #tZr = Structure.from_file('POSCAR_tZrO2.vasp')

    builder = MyBuilder()
    adaptor = AseAtomsAdaptor()
    matcher = StructureMatcher(
        ltol=0.2,       # 격자 길이 허용 오차 (기본값: 0.2)
        stol=0.3,       # 원자 위치 허용 오차 (기본값: 0.3)
        angle_tol=5.0,  # 격자 각도 허용 오차 (기본값: 5.0)
        primitive_cell=False
    )
    subs = builder.create_slabs(CdTe, (1,1,1))
    subs_vec = subs[0].lattice.matrix[:2,:2]
    print(subs_vec)
    film = builder.create_slabs(GaAs, (1,0,0))
    film_vec = film[0].lattice.matrix[:2,:2]
    print(film_vec)

    interfaces = builder.generate_interface(CdTe, GaAs, (1,1,1), (1,0,0))
    screening = matcher.group_structures(interfaces)
    uniques = [rep[0] for rep in screening]
    for i, interface in enumerate(uniques):
        lattice = interface.lattice.matrix
        area = np.linalg.det(lattice[:2,:2])
        atoms = adaptor.get_atoms(interface)
        #if np.sqrt(area) > 15 and np.sqrt(area) <17:
        if area < 150:
            prop = interface.interface_properties
            a, b, alpha = calc_mismatch(prop)
            print(f"{i}\n {a = }\n {b = }\n {alpha = }")
            write(f'interface_{i}.extxyz',atoms, format='extxyz')
    print(len(interfaces), len(uniques))

#
class MyBuilder:
    def __init__(self):
        self.max_atoms = 1000

    def create_slabs(self, structure, miller_index, min_slab_size=10.0, min_vacuum_size=15.0):
        slab_gen = SlabGenerator(
            structure,
            miller_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=True,
            in_unit_planes=False
        )

        slabs = slab_gen.get_slabs()
        return slabs

    def generate_interface(self, substrate_slab, film_slab, substrate_miller, film_miller, max_area=400):
        zsl = ZSLGenerator(
            max_area_ratio_tol=0.10, # 10% 면적 불일치 허용
            max_area=max_area, # 최대 면적 제한
            max_length_tol=0.05, # 5% 길이 불일치 허용  
            max_angle_tol=0.05 # 5도 각도 불일치 허용
        )

        interface_builder = CoherentInterfaceBuilder(
            substrate_structure=substrate_slab,
            film_structure=film_slab,
            substrate_miller=substrate_miller,
            film_miller=film_miller,
            zslgen=zsl,
        )
        #print(dir(interface_builder))
        terminations = interface_builder.terminations
        #print(terminations)

        interfaces = interface_builder.get_interfaces(
            termination=terminations[0],
            gap=2.0
        )

        valid_interfaces = []
        for interface in interfaces:
            natoms= len(interface)
            lattice = interface.lattice.matrix
            area = np.linalg.det(lattice[:2,:2])
            if natoms <= self.max_atoms and area < max_area:
                valid_interfaces.append(interface)

        return valid_interfaces

def calc_mismatch(prop):
    subs = prop['substrate_sl_vectors']
    film = prop['film_sl_vectors']
    [a_subs, b_subs] = subs.copy()
    [a_film, b_film] = film.copy()
    [a_subs_len, b_subs_len] = np.linalg.norm(subs, axis=1)
    [a_film_len, b_film_len] = np.linalg.norm(film, axis=1)
    alpha_subs = np.arccos(np.clip(np.dot(a_subs,b_subs)/(a_subs_len*b_subs_len), -1.0,1.0))
    alpha_film = np.arccos(np.clip(np.dot(a_film,b_film)/(a_film_len*b_film_len), -1.0,1.0))
    alpha_subs_deg = np.degrees(alpha_subs)
    alpha_film_deg = np.degrees(alpha_film)

    # Calculate mismatch %
    mismatch_a = 100 * abs(a_film_len - a_subs_len) / a_subs_len
    mismatch_b = 100 * abs(b_film_len - b_subs_len) / b_subs_len
    mismatch_alpha = 100 * abs(alpha_film_deg - alpha_subs_deg) / alpha_subs_deg

    return mismatch_a, mismatch_b, mismatch_alpha

if __name__ == "__main__":
    main()
