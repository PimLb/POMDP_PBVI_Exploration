import sys
sys.path.append('../..')
from src.pomdp import *

from typing import Union

import pandas as pd
import json

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


def save_simulations_to_csv(file:str, simulation_histories:list[SimulationHistory]) -> None:
    '''
    Function to save a set of simulations to a csv file

    Parameters
    ----------
    file : str
        The file on which the simulation set has to be saved to saved to.
    simulation_histories : list[SimulationHistory]
        The set of simulations to save
    '''
    log('Saving simulation logs')
    max_sim_length = max([len(sim) for sim in simulation_histories])
    all_seq = np.empty((len(simulation_histories), max_sim_length), dtype=object)
    for sim_i, sim in enumerate(simulation_histories):
        seq = []
        for s, a, o, r in zip(sim.states, sim.actions+[], sim.observations+[], sim.rewards+[]):
            seq.append(json.dumps({'s':s, 'a':a, 'o':o, 'r':r}))
        all_seq[sim_i, :len(seq)] = seq

    sim_df = pd.DataFrame(all_seq.T, columns=[f'Sim-{sim_i}' for sim_i in range(len(simulation_histories))])
    sim_df.to_csv(file)


def load_simulations_from_csv(file:str, model:Model) -> list[SimulationHistory]:
    '''
    Function to load a set of simulations that have been saved to a csv file

    Parameters
    ----------
    file : str
        The file on which the simulation set has been saved to.
    model : pomdp.Model
        The model on which the simulations have been run.

    Returns
    -------
    sims : list[SimulationHistory]
        The list of simulations histories.
    '''
    simulations_df = pd.read_csv(file, index_col=0)

    sims = []
    for col in simulations_df.columns:
        sim_steps = simulations_df[col].tolist()
        sim_steps = [json.loads(step) for step in sim_steps if isinstance(step, str)]

        # Creation of simulation history
        sim_hist = SimulationHistory(model, sim_steps[0]['s'], Belief(model))

        sim_hist.states = [step['s'] for step in sim_steps]
        sim_hist.actions = [step['a'] for step in sim_steps]
        sim_hist.observations = [step['o'] for step in sim_steps]
        sim_hist.rewards = [step['r'] for step in sim_steps]

        sims.append(sim_hist)

    return sims