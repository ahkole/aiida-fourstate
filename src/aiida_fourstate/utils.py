from aiida.engine import calcfunction
from aiida import orm


@calcfunction
def compute_FM_AFM_diff(energy_upup, energy_updown, num_effective_NN, num_magnetic_atoms):
    e_upup = energy_upup.value
    e_updown = energy_updown.value
    n = num_effective_NN.value
    N = num_magnetic_atoms.value

    return orm.Float((e_upup - e_updown) / (n * N))
