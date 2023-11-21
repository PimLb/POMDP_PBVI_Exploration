import sys
sys.path.append('../..')
from src.pomdp import *

from typing import Union

import pandas as pd

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')



def compute_extra_steps(simulations:Union[SimulationHistory, list[SimulationHistory]]) -> Union[int, list[int]]:
    '''
    Function to compute the extra steps needed to reach any given end states in a simulation compared to the optimal trajectory.
    The distance is computed using the manhatan distance in 2D.

    Parameters
    ----------
    simulations : SimulationHistory | list[SimulationHistory]
        The simulation to compute the extra steps required from.

    Returns
    -------
    all_extra_steps : int | list[int]
    '''
    sim_list = [simulations] if not isinstance(simulations, list) else simulations

    model = sim_list[0].model.cpu_model

    # Computation of extra steps dataframe
    goal_coords = model.get_coords(model.end_states)

    all_extra_steps = []

    for sim in sim_list:
        opt_traj = np.inf
        len_traj = len(sim)

        start_pos = sim.grid_point_sequence[0]

        for goal in goal_coords:
            man_dist = np.abs(start_pos[0] - goal[0]) + np.abs(start_pos[1] - goal[1])
            if man_dist < opt_traj:
                opt_traj = man_dist

        extra_steps = len_traj - opt_traj
        all_extra_steps.append(extra_steps)

    return all_extra_steps[0] if not isinstance(simulations, list) else all_extra_steps