#!/usr/bin/env runaiida
"""Launch script for the MagneticAnisotropyEnergyWorkChain."""
import numpy as np

from aiida import orm
from aiida.engine import submit
from aiida_common_workflows.common import ElectronicType
import ase.io

from aiida_fourstate import MagneticAnisotropyEnergyWorkChain


num_machines = 1
### Snellius
code_label = 'siesta-5.4.1-foss-2023a-xml-v1@Snellius'
num_mpiprocs_per_machine = 96
num_cores_per_mpiproc = 1
queue_name = 'genoa'
max_memory_kb = 176160768  # 1/2 of total memory of 192-core genoa node



protocol = 'custom'

kpoints = [6, 12, 1]
kpt_node = orm.KpointsData()
kpt_node.set_kpoints_mesh(kpoints)
# Will be stored later

abacus_basis = orm.load_node(uuid='0756b7c4-9c88-484a-b40b-ce6102829902')

# LDA four-state v0 protocol for SIESTA (with coarser k-points)
custom_protocol = {
    'parameters': {
        'block xc-mix': '\n  2\n  LDA LIBXC-001 1.0 0.0\n  LDA LIBXC-012 0.0 1.0\n%endblock xc-mix',
        'max-scf-iterations': 500,
        'scf-mixer-method': 'Pulay',
        'scf-mixer-weight': '0.01',
        'scf-mixer-history': 10,
        'scf-dm-tolerance': '1.0d-6',
        'scf-h-tolerance': '1.0d-5 eV',
        #'solution-method': 'ELSI',
        #'elsi-solver': 'ELPA',
        #'elsi-elpa-flavor': 2,
        #'elsi-broadening-method': 'fermi',
        'solution-method': 'diagon',
        'diag-algorithm': 'Divide-and-Conquer',
        'diag-paralleloverk': 'false',
        'mesh-cutoff': '100 Ry',
        'electronictemperature': '1 meV',
        'write-mulliken-pop': 1,
        'write-hirshfeld-pop': 'true',
        'user-basis-netcdf': 'true',
        'psml-kb-projectors': 'false',
    },
    'basis': {
        'pao-basistype': 'split',
        'pao-basissize': 'DZP',
        'pao-energyshift': '0.01 Ry',
        'pao-splitnorm': '0.15',
        'pao-splittailnorm': 'true',
        'pao-softdefault': 'true',
        'pao-softinnerradius': '0.9',
        'pao-softpotential': '40 Ry',
    },
    'kpoints': kpt_node,
    'pseudo_family': 'PseudoDojo/0.4/LDA/FR/standard/psml',
    # Here we "abuse" the lua.input_files to copy the ion.nc basis files for abacus
    'lua': {'input_files': abacus_basis},
    'description': 'Protocol for the mae verification with LDA functional and PseudoDojo pseudopotentials.'
}


def launch_mae_calculation(ask_for_confirmation=True):
    """Launch a magnetic anisotropy energy calculation.
    """
    global code_label, num_mpiprocs_per_machine, kpt_node, custom_protocol

    ase_atoms = ase.io.read('nii2_2x1_15A.cif')
    structure_unitcell = orm.StructureData(ase=ase_atoms)

    print(f"Unit cell has {len(structure_unitcell.sites)} atoms")
    
    magnetic_sites = [structure_unitcell.get_kind(kind_name).symbol == "Ni" for kind_name in structure_unitcell.get_site_kindnames()]
    magnetic_sites_idx = np.flatnonzero(magnetic_sites)
    magnetization_magnitude = 2.0 

    print(f"Unit cell has {magnetic_sites_idx.size} magnetic sites")

    # Generator inputs for the common workflows
    generator_inputs = {
        'engines': {
            'relax': {
                'code': code_label,
                'options': {
                    'resources': {
                        'num_machines': num_machines,
                        'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                        'num_cores_per_mpiproc': num_cores_per_mpiproc,
                    },
                    'max_wallclock_seconds': 24*3600, # 24 hours
                    'queue_name': queue_name,
                    'max_memory_kb': max_memory_kb,
                    'withmpi': True,
                },
            }
        },
        'protocol': protocol,
        'custom_protocol': custom_protocol if protocol == 'custom' else None,
        'electronic_type': ElectronicType.METAL.value, 
    }
        
    inputs = {
        'structure': structure_unitcell,
        'dir1': orm.List([0., 0., 1.]),  # FM_oop (along z)
        'dir2': orm.List([1., 0., 0.]),  # FM_ip  (along x)
        'magnetization_magnitude': orm.Float(magnetization_magnitude),
        'magnetic_sites': orm.List(list(magnetic_sites_idx)),
        'generator_inputs': generator_inputs,
        'engine_name': orm.Str('siesta'),
    }

    if ask_for_confirmation:
        print(f"Submitting MAE for NiI2 to {code_label} on {num_machines} node(s) with {num_mpiprocs_per_machine=}.")
        print("Submit? [Ctrl+C to stop]")
        input()
    
    # We first need to store the kpoints node (if not done already)
    kpt_node.store()

    print("Submitting MagneticAnisotropyEnergyWorkChain...")
    node = submit(MagneticAnisotropyEnergyWorkChain, **inputs)
    print(f"\nSubmitted: MAE, PK = {node.pk}")

    return node

if __name__ == '__main__':
    print(f"\n\nLaunching SIESTA calculation of MAE\n")
    launch_mae_calculation()
