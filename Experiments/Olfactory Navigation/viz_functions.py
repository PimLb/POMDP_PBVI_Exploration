import sys
sys.path.append('../..')
from src.pomdp import *
from util_functions import compute_extra_steps

from scipy.stats import entropy
import pandas as pd

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')



def plot_steps(sim_hist:SimulationHistory, ax=None) -> None:
    '''
    Plots a special version of the simulation plot for olfactory navigation
    
    Parameters
    ----------
    sim_hist : SimulationHistory
        The completed simulation history.
    ax : optional
        The ax the produce the plot on, if not a new one will be created.
    '''
    # Generate ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(18,3))

    # Get cpu model
    model = sim_hist.model.cpu_model

    # Plot setup
    env_shape = model.cpu_model.state_grid.shape
    ax.set_xlim(0, env_shape[1])
    ax.set_ylim(env_shape[0], 0)

    # Start
    start_coord = sim_hist.grid_point_sequence[0]
    ax.scatter(start_coord[1], start_coord[0], c='green', label='Start')

    # Goal
    goal_coord = np.array([np.argwhere(model.state_grid == g)[0].tolist() for g in model.end_states])
    ax.scatter(goal_coord[:,1], goal_coord[:,0], c='red', label='Goal')

    # Path
    seq = np.array(sim_hist.grid_point_sequence)
    ax.plot(seq[:,1], seq[:,0], zorder=-1, c='black', label='Path')

    # Something sensed
    something_obs_id = model.observation_labels.index('something')
    obs_ts = np.where(np.array(sim_hist.observations) == something_obs_id)
    points_obs = seq[obs_ts[0],:]
    ax.scatter(points_obs[:,1], points_obs[:,0], zorder=1, label='Something observed')

    # Points sniff
    sniff_air_action_id = -1
    for i, al in enumerate(model.action_labels):
        if 'air' in al.lower():
            sniff_air_action_id = i
    sniff_in_air = np.where(np.array(sim_hist.actions) == sniff_air_action_id)
    points_sniff = seq[sniff_in_air[0],:]
    if len(points_sniff) > 0:
        ax.scatter(points_sniff[:,1], points_sniff[:,0], zorder=2, marker='x', label='Sniff in the air')

    # Generate legend
    ax.legend()


def plot_entropy_value(sim_hist:SimulationHistory, value_function:ValueFunction, ax=None) -> None:
    '''
    Plots a graph of the entropy of the beliefs passed through during a simulation versus the value of a the same beliefs.

    Parameters
    ----------
    sim_hist : SimulationHistory
        The completed simulation history.
    value_function : ValueFunction
        The value function to compute the value of a belief with.
    ax : optional
        The ax the produce the plot on, if not a new one will be created.
    '''
    # Computing entropies and values
    belief_values = [b.values if (not gpu_support) or (cp.get_array_module(b.values) == np) else cp.asnumpy(b.values) for b in sim_hist.beliefs]

    b_ents = [entropy(b_val) for b_val in belief_values]

    b_array = np.array(belief_values)
    av_array = value_function.to_cpu().alpha_vector_array
    b_vals = np.max(np.matmul(av_array, b_array.T), axis=0)

    # Generate ax is not provided
    if ax is None:
        _, ax = plt.subplots()

    # Actual plot
    ax.set_xlabel('Time')

    # Entropy plt
    ax.plot(np.arange(len(b_ents)), b_ents, color='blue')
    ax.set_ylabel('Entropy', color='blue')

    ax2 = ax.twinx()

    # Value plt
    ax2.plot(np.arange(len(b_vals)), b_vals, color='red')
    ax2.set_ylabel('Value', color='red')


def plot_extra_steps_from_pandas(df:pd.DataFrame, ax=None) -> None:
    '''
    Plot box plots of additional steps along simulations (columns) and for each row representing different runs.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the extra steps data.
    ax : optional
        The ax the produce the plot on, if not a new one will be created.
    '''
    # If not there, generate a IDs column
    if 'IDs' not in df:
        df['IDs'] = [f'Run-{i}' for i in df.index]

    # Processing the extra steps values
    extra_steps_array = df.drop(columns='IDs').to_numpy().T
    extra_steps_df = pd.DataFrame(extra_steps_array, columns=df['IDs'].to_list())

    # Actual plot
    if ax is None:
        _, ax = plt.subplots()

    extra_steps_df.plot(kind='box', ax=ax, rot=45)
    ax.set_ylabel('Additional steps to target')


def plot_extra_steps_from_file(file:str, ax=None) -> None:
    '''
    Plot box plots of additional steps along simulations (columns) and for each row representing different runs.
    Reads a csv file and plots the extra steps in this file.

    Parameters
    ----------
    file : str
        The csv file containing a dataframe with the extra steps data.
    ax : optional
        The ax the produce the plot on, if not a new one will be created.
    '''
    df = pd.read_csv(file)
    if 'Average' in df:
        df = df.drop(columns='Average')

    plot_extra_steps_from_pandas(df, ax)
    

def plot_extra_steps(all_sim_histories:list[SimulationHistory], ax=None) -> None:
    '''
    Plot box plots of additional steps along simulations in the list provided.
    Will produce a single box plot as it is a single run.

    Parameters
    ----------
    all_sim_histories : list[SimulationHistory]
        The list of simulations to compute the extra steps from and plot.
    ax : optional
        The ax the produce the plot on, if not a new one will be created.
    '''
    # Computation of extra steps list
    all_extra_steps = compute_extra_steps(all_sim_histories)

    # Generation of dataframe
    df = pd.DataFrame(np.array(all_extra_steps)[None,:], columns=[f'Sim-{i}' for i in range(len(all_sim_histories))])
    df['IDs'] = ['Run-0']

    # Plotting extra steps
    plot_extra_steps_from_pandas(df, ax)


def plot_grid_extra_steps(points_df, ax=None) -> None:
    '''
    Plot the average extra steps required in a grid format
    '''
    # Computing averages per cell and cell position
    average_per_cell = points_df.groupby('cell').mean('extra_steps')['extra_steps'].to_list()
    cell_xs = pd.unique(points_df['cell_x'])
    cell_ys = pd.unique(points_df['cell_y'])

    average_grid = []
    item = 0
    for _ in cell_ys:
        row = []
        for _ in cell_xs:
            row.append(average_per_cell[item])
            item += 1
        average_grid.append(row)
    average_grid_array = np.array(average_grid)

    # Actual plot
    if ax is None:
        _, ax = plt.subplots()

    im = ax.imshow(average_grid_array, cmap=plt.cm.get_cmap('RdYlGn').reversed())
    plt.colorbar(im, orientation='horizontal', ax=ax)

    # Axes
    ax.set_xticks(np.arange(average_grid_array.shape[1]), labels=cell_xs.astype(int), rotation=90)
    ax.set_yticks(np.arange(average_grid_array.shape[0]), labels=cell_ys.astype(int))


def plot_grid_extra_steps_from_file(file) -> None:
    plot_grid_extra_steps(pd.read_csv(file))