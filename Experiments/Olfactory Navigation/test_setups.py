import sys
sys.path.append('../..')
from src.pomdp import *
from util_functions import compute_extra_steps

import pandas as pd
import numpy as np
import cupy as cp
import json
import os

from cupy.cuda import runtime as cuda_runtime
from csv import writer as csv_writer
from datetime import datetime



def grid_test(value_function:ValueFunction, cell_size:int=10, points_per_cell:int=10, zone=None, ax=None) -> pd.DataFrame:
    '''
    Function to test a given value function with a certain amount of simulations within cells of the state space.
    It then plots the average extra steps required for each cell to reach the goal given an optimal trajectory computed with the manhatan distance.

    '''
    model = value_function.model

    # Getting grid zone
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    if zone is None:
        start_coords = np.array(model.cpu_model.get_coords(model.cpu_model.states[model.cpu_model.start_probabilities > 0]))
        (min_x, max_x) = (np.min(start_coords[:,1]), np.max(start_coords[:,1]))
        (min_y, max_y) = (np.min(start_coords[:,0]), np.max(start_coords[:,0]))
    else:
        ((min_x,max_x),(min_y,max_y)) = zone

    # Generation of points
    random_points = []

    for i in range(min_y, max_y, cell_size):
        for j in range(min_x, max_x, cell_size):
            for _ in range(points_per_cell):
                rand_x = np.random.randint(j, min([max_x, j+cell_size]))
                rand_y = np.random.randint(i, min([max_y, i+cell_size]))

                random_points.append([rand_x, rand_y])

    rand_points_array = np.array(random_points)

    points_df = pd.DataFrame(rand_points_array, columns=['x','y'])

    # # Cells
    points_df['cell'] = np.repeat(np.arange(len(points_df)/points_per_cell, dtype=int), points_per_cell)

    # Traj and ids
    goal_state_coords = model.get_coords(model.end_states[0])
    points_df['opt_traj'] = np.abs(goal_state_coords[1] - rand_points_array[:,0]) + np.abs(goal_state_coords[0] - rand_points_array[:,1])
    points_df['point_id'] = (model.state_grid.shape[1] * rand_points_array[:,1]) + rand_points_array[:,0]

    # Setup agent
    a = Agent(model, value_function)

    # Run test
    _, all_sim_hist = a.run_n_simulations_parallel(len(rand_points_array), start_state=points_df['point_id'].to_list())

    # Adding sim results
    points_df['steps_taken'] = [len(sim) for sim in all_sim_hist]
    points_df['extra_steps'] = points_df['steps_taken'] - points_df['opt_traj']

    # Computing averages per cell and cell position
    average_per_cell = points_df.groupby('cell').mean('extra_steps')['extra_steps'].to_list()
    cell_centers = []
    average_grid = []
    item = 0
    for i in range(min_y, max_y, cell_size):
        row = []
        for j in range(min_x, max_x, cell_size):
            row.append(average_per_cell[item])

            cell_centers.append([j+int(cell_size/2), i+int(cell_size/2)])

            item += 1
        average_grid.append(row)

    average_grid_array = np.array(average_grid)
    cell_centers_array = np.array(cell_centers)

    # Actual plot
    if ax is None:
        _, ax = plt.subplots()

    ax.set_title(f'Additional steps needed\nAvg of {points_per_cell} realizations per tile')

    im = ax.imshow(average_grid_array, cmap=plt.cm.get_cmap('RdYlGn').reversed())
    plt.colorbar(im, orientation='horizontal', ax=ax)

    # Axes
    ax.set_xticks(np.arange(average_grid_array.shape[1]), labels=np.unique(cell_centers_array[:,0]), rotation=90)
    ax.set_yticks(np.arange(average_grid_array.shape[0]), labels=np.unique(cell_centers_array[:,1]))

    # Return results
    return points_df


def run_solve_test(
        model_file:str,
        expand_function:str,
        gamma:float=0.99,
        eps:float=1e-6,
        expansions:int=300,
        belief_growth:int=100,
        runs:int=20,
        simulations:int=300,
        sim_starts:Union[int,list[int],None]=None,
        sim_horizon:int=1000,
        name:Union[str,None]=None
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Test folder creation
    test_name = name if name is not None else model_file.replace('.pck','').split('/')[-1]
    test_folder = f"./Test_{test_name}_{expand_function}_{expansions}it_{belief_growth}exp_{str(gamma).replace('.','')}g_{eps}eps_{runs}run_{simulations}sim_{timestamp}"
    os.mkdir(test_folder)

    # Simulations folder
    sim_folder = test_folder + '/Simulations'
    os.mkdir(sim_folder)

    # Value functions folder
    vf_folder = test_folder + '/ValueFunctions'
    os.mkdir(vf_folder)

    # Making extra steps file
    extra_steps_file = test_folder + '/extra_steps.csv'
    with open(extra_steps_file, 'w') as f_object:
        writer_object = csv_writer(f_object)
        writer_object.writerow(['Average'] + [f'Sim-{i}' for i in range(simulations)])

    # Actual test loop
    for iter in range(runs):
        try:
            
            print('--------------------------------------------------------------------------------')
            print(f'Run {iter+1} of {runs} (Run-{iter})')
            print('--------------------------------------------------------------------------------')

            
            # Loading model
            model = Model.load_from_file(model_file)

            # Solving model
            pbvi_solver = PBVI_Solver(gamma=gamma, eps=eps, expand_function=expand_function)
            solution, hist = pbvi_solver.solve(model=model,
                                               expansions=expansions,
                                               max_belief_growth=belief_growth,
                                               print_progress=False,
                                               use_gpu=True)
            print(hist.summary)

            # Saving value function
            solution.save(path=vf_folder, file_name=f'run-{iter}-VF', compress=True)

            # Running simulation
            print()
            log('Starting simulations')
            a = Agent(model, solution)
            all_rewards, all_sim_histories = a.run_n_simulations_parallel(
                n=simulations,
                max_steps=sim_horizon,
                print_progress=False,
                start_state=sim_starts)

            # Computing extra steps
            print()
            extra_steps = np.array(compute_extra_steps(all_sim_histories))
            log(f'Average Extra steps count: {np.average(extra_steps)}')

            # Saving extra step results
            with open(extra_steps_file, 'a') as f_object:
                writer_object = csv_writer(f_object)
                writer_object.writerow([np.average(extra_steps)] + extra_steps.tolist())
            
            # Saving simulations
            log('Saving results')
            all_seq = np.empty((simulations, sim_horizon+1), dtype=object)
            for sim_i, sim in enumerate(all_sim_histories):
                seq = []
                for s, a, o, r in zip(sim.states, sim.actions+[], sim.observations+[], sim.rewards+[]):
                    seq.append(json.dumps({'s':s, 'a':a, 'o':o, 'r':r}))
                
                all_seq[sim_i, :len(seq)] = seq

            sim_df = pd.DataFrame(all_seq.T, columns=[f'Sim-{sim_i}' for sim_i in range(len(all_sim_histories))])
            sim_df.to_csv(sim_folder + f'/run-{iter}-sims.csv')

            print('\n\n')

        except Exception as e:
            print()
            print(e)
            log('/!\\ Error Happend /!\\ \n\n')
        cp._default_memory_pool.free_all_blocks()
    
    print('\n\n')
    print('--------------------------------------------------------------------------------')
    print(f'DONE')
    print('--------------------------------------------------------------------------------')


def run_single_solve_test(
        model_file:str,
        expand_function:str,
        gamma:float=0.99,
        eps:float=1e-6,
        expansions:int=300,
        belief_growth:int=100,
        runs:int=20,
        simulations:int=300,
        sim_starts:Union[int,list[int],None]=None,
        sim_horizon:int=1000,
        name:Union[str,None]=None
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Test folder creation
    test_name = name if name is not None else model_file.replace('.pck','').split('/')[-1]
    test_folder = f"./Test_{test_name}_{expand_function}_{expansions}it_{belief_growth}exp_{str(gamma).replace('.','')}g_{eps}eps_{runs}run_{simulations}sim_{timestamp}"
    os.mkdir(test_folder)

    # Simulations folder
    sim_folder = test_folder + '/Simulations'
    os.mkdir(sim_folder)

    # Value functions folder
    vf_folder = test_folder + '/ValueFunctions'
    os.mkdir(vf_folder)

    # Making extra steps file
    extra_steps_file = test_folder + '/extra_steps.csv'
    with open(extra_steps_file, 'w') as f_object:
        writer_object = csv_writer(f_object)
        writer_object.writerow(['Average'] + [f'Sim-{i}' for i in range(simulations)])

    # Prev solution file
    solution_file = None

    # Actual test loop
    for iter in range(runs):
        try:
            
            print('--------------------------------------------------------------------------------')
            print(f'Run {iter+1} of {runs} (Run-{iter})')
            print('--------------------------------------------------------------------------------')

            
            # Loading model
            model = Model.load_from_file(model_file)

            # Value function loading
            value_function = None
            if solution_file is not None:
                value_function = ValueFunction.load_from_file(solution_file, model)

            # Solving model
            pbvi_solver = PBVI_Solver(gamma=gamma, eps=eps, expand_function=expand_function)
            solution, hist = pbvi_solver.solve(model=model,
                                               expansions=int(expansions/runs),
                                               max_belief_growth=belief_growth,
                                               initial_value_function=value_function,
                                               print_progress=False,
                                               use_gpu=True)
            print(hist.summary)

            # Saving value function
            solution_file = vf_folder + f'/run-{iter}-VF.csv.gzip'
            solution.save(path=vf_folder, file_name=f'run-{iter}-VF', compress=True)

            # Running simulation
            print()
            log('Starting simulations')
            a = Agent(model, solution)
            all_rewards, all_sim_histories = a.run_n_simulations_parallel(
                n=simulations,
                max_steps=sim_horizon,
                print_progress=False,
                start_state=sim_starts)

            # Computing extra steps
            print()
            extra_steps = np.array(compute_extra_steps(all_sim_histories))
            log(f'Average Extra steps count: {np.average(extra_steps)}')

            # Saving extra step results
            with open(extra_steps_file, 'a') as f_object:
                writer_object = csv_writer(f_object)
                writer_object.writerow([np.average(extra_steps)] + extra_steps.tolist())
            
            # Saving simulations
            log('Saving results')
            all_seq = np.empty((simulations, sim_horizon+1), dtype=object)
            for sim_i, sim in enumerate(all_sim_histories):
                seq = []
                for s, a, o, r in zip(sim.states, sim.actions+[], sim.observations+[], sim.rewards+[]):
                    seq.append(json.dumps({'s':s, 'a':a, 'o':o, 'r':r}))
                
                all_seq[sim_i, :len(seq)] = seq

            sim_df = pd.DataFrame(all_seq.T, columns=[f'Sim-{sim_i}' for sim_i in range(len(all_sim_histories))])
            sim_df.to_csv(sim_folder + f'/run-{iter}-sims.csv')

            print('\n\n')

        except Exception as e:
            print()
            print(e)
            log('/!\\ Error Happend /!\\ \n\n')
        cp._default_memory_pool.free_all_blocks()
    
    print('\n\n')
    print('--------------------------------------------------------------------------------')
    print(f'DONE')
    print('--------------------------------------------------------------------------------')