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

    builder = MyBuilder()
    adaptor = AseAtomsAdaptor()
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=False
    )

    interfaces = builder.generate_interface(CdTe, GaAs, (1,0,0), (1,0,0))
    screening = matcher.group_structures(interfaces)
    uniques = [rep[0] for rep in screening]
    for i, interface in enumerate(uniques):
        lattice = interface.lattice.matrix
        area = np.linalg.det(lattice[:2,:2])
        atoms = adaptor.get_atoms(interface)
        if np.sqrt(area) < 15:
            #print(dir(interface))
            prop = interface.interface_properties
            print(dir(prop))
            keys = prop.keys()
            for k, key in enumerate(keys):
                print(key, prop[key])
            #print(interface.interface_properties['strain'])
            write(f'interface_{i}.extxyz',atoms, format='extxyz')
        break
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
            max_area_ratio_tol=0.10,
            max_area=max_area,
            max_length_tol=0.05,
            max_angle_tol=0.05
        )

        interface_builder = CoherentInterfaceBuilder(
            substrate_structure=substrate_slab,
            film_structure=film_slab,
            substrate_miller=substrate_miller,
            film_miller=film_miller,
            zslgen=zsl,
        )
        print(dir(interface_builder))
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

if __name__ == "__main__":
    main()
