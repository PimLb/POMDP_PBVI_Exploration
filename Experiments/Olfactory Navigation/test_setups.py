import sys
sys.path.append('../..')
from src.pomdp import *
from util_functions import compute_extra_steps
from simulations import RealSimulationSetQComp
from util_functions import save_simulations_to_csv

import pandas as pd
import numpy as np
import cupy as cp
import json
import os
import cv2

from cupy.cuda import runtime as cuda_runtime
from csv import writer as csv_writer
from datetime import datetime



def grid_test(value_function:ValueFunction,
              cell_size:int=10,
              points_per_cell:int=10,
              zone=None,
              print_progress:bool=False) -> pd.DataFrame:
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
    cell_centers_x = []
    cell_centers_y = []

    for i in range(min_y, max_y, cell_size):
        for j in range(min_x, max_x, cell_size):
            cell_centers_x.append((j + min([max_x, j+cell_size])) / 2)
            cell_centers_y.append((i + min([max_y, i+cell_size])) / 2)

            for _ in range(points_per_cell):
                rand_x = np.random.randint(j, min([max_x, j+cell_size]))
                rand_y = np.random.randint(i, min([max_y, i+cell_size]))

                random_points.append([rand_x, rand_y])

    rand_points_array = np.array(random_points)

    points_df = pd.DataFrame(rand_points_array, columns=['x','y'])

    # # Cells
    points_df['cell'] = np.repeat(np.arange(len(points_df)/points_per_cell, dtype=int), points_per_cell)
    points_df['cell_x'] = np.repeat(np.array(cell_centers_x), points_per_cell)
    points_df['cell_y'] = np.repeat(np.array(cell_centers_y), points_per_cell)

    # Traj and ids
    goal_state_coords = model.get_coords(model.end_states[0])
    points_df['opt_traj'] = np.abs(goal_state_coords[1] - rand_points_array[:,0]) + np.abs(goal_state_coords[0] - rand_points_array[:,1])
    points_df['point_id'] = (model.state_grid.shape[1] * rand_points_array[:,1]) + rand_points_array[:,0]

    # Setup agent
    a = Agent(model, value_function)

    # Run test
    _, all_sim_hist = a.run_n_simulations_parallel(len(rand_points_array), start_state=points_df['point_id'].to_list(), print_progress=print_progress)

    # Adding sim results
    points_df['steps_taken'] = [len(sim) for sim in all_sim_hist]
    points_df['extra_steps'] = points_df['steps_taken'] - points_df['opt_traj']

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
        use_gpu:bool=True,
        name:Union[str,None]=None
):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Test folder creation
    test_name = name if name is not None else model_file.replace('.pck','').split('/')[-1]
    test_folder = f"./Test_{test_name}_{expand_function}_{expansions}it_{belief_growth}exp_{str(gamma).replace('.','')}g_{eps}eps_{runs}run_{simulations}sim_{timestamp}"
    os.mkdir(test_folder)

    # Simulations folder
    sim_folder = test_folder + '/Simulations'
    if simulations > 0:
        os.mkdir(sim_folder)

    # Value functions folder
    vf_folder = test_folder + '/ValueFunctions'
    os.mkdir(vf_folder)

    # Making extra steps file
    extra_steps_file = test_folder + '/extra_steps.csv'
    if simulations > 0:
        with open(extra_steps_file, 'w') as f_object:
            writer_object = csv_writer(f_object)
            writer_object.writerow(['Average'] + [f'Sim-{i}' for i in range(simulations)])

    # Prev solution file
    solution = None
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
            if use_gpu and solution_file is not None:
                solution = ValueFunction.load_from_file(solution_file, model)

            # Solving model
            pbvi_solver = PBVI_Solver(gamma=gamma, eps=eps, expand_function=expand_function)
            solution, hist = pbvi_solver.solve(model=model,
                                               expansions=int(expansions/runs),
                                               max_belief_growth=belief_growth,
                                               initial_value_function=solution,
                                               print_progress=False,
                                               use_gpu=use_gpu)
            print(hist.summary)

            # Saving value function
            solution_file = vf_folder + f'/run-{iter}-VF.csv.gzip'
            solution.save(path=vf_folder, file_name=f'run-{iter}-VF', compress=True)

            # Running simulation
            if simulations > 0:
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


def run_grid_test(model_file:str,
                  folder:str,
                  points_per_cell:int=20,
                  use_gpu:bool=True):
    
    if not folder.endswith('/'):
        folder += '/'

    log('Gathering value function files...')
    value_function_files = os.listdir(folder + 'ValueFunctions')

    print('Running Grid test with parameters:')
    print(f'\t- model: "{model_file}"')
    print(f'\t- folder: "{folder}"')
    print(f'\t- vf count: {len(value_function_files)}')
    print()

    log('Creation of GridSimulations folder is doesnt exist yet')
    if not os.path.isdir(folder + 'GridSimulations'):
        os.mkdir(folder + 'GridSimulations')

    for i, vf_file in enumerate(value_function_files):
        print('--------------------------------------------------------------------------------')
        print(f'Run {i+1} of {len(value_function_files)} (Run-{i})')
        print('--------------------------------------------------------------------------------')

        model = Model.load_from_file(model_file)

        # Value functions
        log(f'Loading Value function: {vf_file}')
        vf = ValueFunction.load_from_file(folder + f'ValueFunctions/' + vf_file, model)

        if use_gpu:
            vf = vf.to_gpu()

        # Run grid test
        log('Starting simulations')
        res_df = grid_test(vf, points_per_cell=points_per_cell)

        # Save results
        log('Simulations done, saving results')
        res_df.to_csv(folder + f'GridSimulations/Grid-run-{i}-{len(res_df)}sims.csv', index=False)

        # Refresh memory
        cp._default_memory_pool.free_all_blocks()

        # Print average extra steps
        print(f'\nAverage Extra steps count: {res_df["extra_steps"].mean()}')
        print('\n\n')
    
    # Summarize extra steps in grid_extra_steps.csv file
    print('--------------------------------------------------------------------------------')
    log('Summarizing extra steps')
    all_averages = []
    for i in range(len(value_function_files)):
        df = pd.read_csv(folder + f'GridSimulations/Grid-run-{i}-{len(res_df)}sims.csv')
        all_averages.append(df['extra_steps'].tolist())

    df = pd.DataFrame(np.array(all_averages), columns=[f'Sim-{i}' for i in range(len(res_df))])
    df.to_csv(folder + 'grid_extra_steps.csv', index=False)

    print('--------------------------------------------------------------------------------')
    print(f'DONE')
    print('--------------------------------------------------------------------------------')


def get_stats_data(dx_mod:float=1.0, dy_mod:float=1.0) -> tuple[np.ndarray, np.ndarray]:
    # The full statistics
    nose_data = pd.read_csv('./Data/statistics_abs_nose_3e6.dat', sep=' ', skiprows=[0], names=list(range(320)), index_col=False).to_numpy().T
    ground_data = pd.read_csv('./Data/statistics_abs_ground_3e6.dat', sep=' ', skiprows=[0], names=list(range(320)), index_col=False).to_numpy().T
    
    # General variables
    points_per_unit = 30
    env_x = [-2,10]
    env_y = [-1,1]

    width = ((env_x[1] - env_x[0]) * points_per_unit) + 1
    height = ((env_y[1] - env_y[0]) * points_per_unit) + 1

    env_shape = [height, width]

    # Computing the width and height
    dx = int((4 * points_per_unit) * dx_mod)
    dx += 1 if dx % 2 == 0 else 0

    dy = int((points_per_unit) * dy_mod)
    dy += 1 if dy % 2 == 0 else 0

    top_y = int(points_per_unit - ((dy-1) / 2))

    nose_data = cv2.resize(nose_data, dsize=(dx, dy))
    ground_data = cv2.resize(ground_data, dsize=(dx, dy))

    # Generation of the final arrays
    nose_data_padded = np.zeros(env_shape)
    nose_data_padded[top_y:top_y+dy, 60:60+dx] = nose_data

    ground_data_padded = np.zeros(env_shape)
    ground_data_padded[top_y:top_y+dy, 60:60+dx] = ground_data

    nose_data = nose_data_padded
    ground_data = ground_data_padded

    return nose_data, ground_data


class AltSimulationSet(SimulationSet):
    # An alternative to the SimulationSet to have an alternate observation probabilities
    def __init__(self, model:Model, alt_air:np.ndarray, alt_ground:np.ndarray):
        self.alt_air = alt_air.ravel()
        self.alt_ground = alt_ground.ravel()
        super().__init__(model)

    
    def run_actions(self, actions:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rewards,_ =  super().run_actions(actions)

        # GPU support
        xp = np if not self.model.is_on_gpu else cp

        # Generate observations
        obs_probs = xp.where(actions == 5, self.alt_air[self.agent_states], self.alt_ground[self.agent_states])
        observations = (xp.random.random(self.n) < obs_probs).astype(int)

        # Check if at goal
        observations = xp.where(xp.isin(self.agent_states, xp.array(self.model.end_states)), 2, observations)

        return rewards, observations


def grid_test_alt_model(model_file:str,
                        vf_file:str,
                        dx_alt:float,
                        dy_alt:float
                        ) -> tuple[pd.DataFrame, list[SimulationHistory]]:
    
    # Retreiving model and vf from storage
    log(f'Loading Value function and model')
    model = Model.load_from_file(model_file)
    vf = ValueFunction.load_from_file(vf_file, model)

    #Retrieving observations arrays
    log(f'Loading olfactory data')
    nose_data, ground_data = get_stats_data(dx_alt,dy_alt)

    # Sending arrays to gpu
    model = model.gpu_model
    value_function = vf.to_gpu()
    nose_data = cp.array(nose_data)
    ground_data = cp.array(ground_data)

    # Getting grid zone
    zone = None
    cell_size = 10
    points_per_cell = 10

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
    cell_centers_x = []
    cell_centers_y = []

    for i in range(min_y, max_y, cell_size):
        for j in range(min_x, max_x, cell_size):
            cell_centers_x.append((j + min([max_x, j+cell_size])) / 2)
            cell_centers_y.append((i + min([max_y, i+cell_size])) / 2)

            for _ in range(points_per_cell):
                rand_x = np.random.randint(j, min([max_x, j+cell_size]))
                rand_y = np.random.randint(i, min([max_y, i+cell_size]))

                random_points.append([rand_x, rand_y])

    rand_points_array = np.array(random_points)

    points_df = pd.DataFrame(rand_points_array, columns=['x','y'])

    # # Cells
    points_df['cell'] = np.repeat(np.arange(len(points_df)/points_per_cell, dtype=int), points_per_cell)
    points_df['cell_x'] = np.repeat(np.array(cell_centers_x), points_per_cell)
    points_df['cell_y'] = np.repeat(np.array(cell_centers_y), points_per_cell)

    # Traj and ids
    goal_state_coords = model.get_coords(model.end_states[0])
    points_df['opt_traj'] = np.abs(goal_state_coords[1] - rand_points_array[:,0]) + np.abs(goal_state_coords[0] - rand_points_array[:,1])
    points_df['point_id'] = (model.state_grid.shape[1] * rand_points_array[:,1]) + rand_points_array[:,0]

    # Make agent and simulte
    a = Agent(model, value_function)

    _, all_sim_hist = a.run_n_simulations_parallel(len(rand_points_array),
                                                    simulator_set=AltSimulationSet(model, alt_air=nose_data, alt_ground=ground_data),
                                                    start_state=points_df['point_id'].to_list(),
                                                    print_progress=False)

    # Adding sim results
    points_df['steps_taken'] = [len(sim) for sim in all_sim_hist]
    points_df['extra_steps'] = points_df['steps_taken'] - points_df['opt_traj']

    # Return results
    return points_df, all_sim_hist


def run_grid_test_alt(folder:str):
    
    changes = [i/10 for i in range(2,19)]

    print('Running Grid test for alternate observation probabilities:')
    print(f'\t- model: "Alt_Wrap_GroundAir"')
    print(f'\t- value function: "./Test_GroundAir_FSVI_300it_100exp_099g_e6eps_20run_20231121_165329/ValueFunctions/run-3-VF.gzip"')
    print(f'\t- Amounts of tests: {str(len(changes)**2)}')
    print()

    log('Creation of GridSimulations folder is doesnt exist yet')
    if not os.path.isdir(folder + '/GridSimulations'):
        os.mkdir(folder + '/GridSimulations')

    log('Creation of SimulationsLog folder is doesnt exist yet')
    sim_folder = folder + '/SimulationsLog'
    if not os.path.isdir(sim_folder):
        os.mkdir(sim_folder)

    change_sets = []
    for c_x in changes:
        for c_y in changes:
            change_sets.append([c_x, c_y])

    for i, c_set in enumerate(change_sets):
        c_x, c_y = c_set
        print('--------------------------------------------------------------------------------')
        print(f'Run {i+1} of {len(change_sets)} (Run-{i}) (dx: {int(c_x*100)}%; dy: {int(c_y*100)}%)')
        print('--------------------------------------------------------------------------------')

        # Run grid test
        log('Starting simulations')
        res_df, all_sim_histories = grid_test_alt_model(model_file='./Models/Alt_Wrap_GroundAir.pck',
                                                        vf_file='./Test_GroundAir_FSVI_300it_100exp_099g_e6eps_20run_20231121_165329/ValueFunctions/run-3-VF.gzip',
                                                        dx_alt=c_x,
                                                        dy_alt=c_y)

        # Save results
        log('Simulations done, saving results')
        res_df.to_csv(folder + f'/GridSimulations/Grid-run-{i}-dx{c_x}-dy{c_y}-{len(res_df)}sims.csv', index=False)

        # Saving simulations
        log('Saving simulation logs')
        all_seq = np.empty((780, 1001), dtype=object)
        for sim_i, sim in enumerate(all_sim_histories):
            seq = []
            for s, a, o, r in zip(sim.states, sim.actions+[], sim.observations+[], sim.rewards+[]):
                seq.append(json.dumps({'s':s, 'a':a, 'o':o, 'r':r}))
            
            all_seq[sim_i, :len(seq)] = seq

        sim_df = pd.DataFrame(all_seq.T, columns=[f'Sim-{sim_i}' for sim_i in range(len(all_sim_histories))])
        sim_df.to_csv(sim_folder + f'/run-{i}-dx{c_x}-dy{c_y}-sims.csv')


        # Refresh memory
        cp._default_memory_pool.free_all_blocks()

        # Print average extra steps
        print(f'\nAverage Extra steps count: {res_df["extra_steps"].mean()}')
        print('\n\n')
    
    # Summarize extra steps in grid_extra_steps.csv file
    print('--------------------------------------------------------------------------------')
    log('Summarizing extra steps')
    all_averages = []
    for i in range(len(change_sets)):
        df = pd.read_csv(folder + f'/GridSimulations/Grid-run-{i}-{len(res_df)}sims.csv')
        all_averages.append(df['extra_steps'].tolist())

    df = pd.DataFrame(np.array(all_averages), columns=[f'Sim-{i}' for i in range(len(res_df))])
    df.to_csv(folder + '/grid_extra_steps.csv', index=False)

    print('--------------------------------------------------------------------------------')
    print(f'DONE')
    print('--------------------------------------------------------------------------------')


def run_all_start_test_q(folder:str):
    
    shifts = [i*10 for i in range(1,101)]
    model_file = './Models/Nose_Only_27_123.pck'
    value_function_file = './Test_Results/Test_Q_Learning_Comparison/20240207_182536_value_function.parquet'

    print('Running simulations from all possible starting positions with shifts at each 10 timesteps:')
    print(f'\t- model: "{model_file}"')
    print(f'\t- value function: "{value_function_file}"')
    print(f'\t- Amounts of tests: {str(len(shifts))}')
    print()

    log('Creation of SimulationSets folder is doesnt exist yet')
    sim_folder = folder + '/SimulationSets'
    if not os.path.isdir(sim_folder):
        os.mkdir(sim_folder)

    log('Creation of SimulationStats folder is doesnt exist yet')
    sim_stats_folder = folder + '/SimulationStats'
    if not os.path.isdir(sim_stats_folder):
        os.mkdir(sim_stats_folder)

    for i, shift in enumerate(shifts):
        print('--------------------------------------------------------------------------------')
        print(f'Run {i+1} of {len(shifts)} (Run-{i}) (Start frame: {shift})')
        print('--------------------------------------------------------------------------------')

        # Loading model and vf to gpu
        log('Loading model and value function')
        model = Model.load_from_file(model_file).gpu_model
        vf = ValueFunction.load_from_parquet(value_function_file, model).to_gpu()

        # Run grid test
        log('Starting simulations')
        a = Agent(model=model, value_function=vf)
        start_positions = np.argwhere(model.start_probabilities > 0)[:,0].tolist()

        simulator_set = RealSimulationSetQComp(model, '/storage/rando/data/nose_data_27_123.npy', shift=shift)

        _, all_sims = a.run_n_simulations_parallel(n=len(start_positions),
                                                   simulator_set=simulator_set,
                                                   max_steps=400,
                                                   start_state=start_positions,
                                                   print_progress=False)

        # Save results
        log('Simulations done, saving results')
        env_shape = [56, 247]
        source = [28,62]

        sim_stats_df = pd.DataFrame()
        sim_stats_df['start_id'] = [sim.states[0] for sim in all_sims]
        sim_stats_df['start_x'] = [np.unravel_index(sim.states[0], env_shape)[1] for sim in all_sims]
        sim_stats_df['start_y'] = [np.unravel_index(sim.states[0], env_shape)[0] for sim in all_sims]
        sim_stats_df['steps_taken'] = [len(sim) for sim in all_sims]
        sim_stats_df['opt_steps'] = np.abs(sim_stats_df['start_x'] - source[1]) + np.abs(sim_stats_df['start_y'] - source[0])
        sim_stats_df['converged'] = sim_stats_df['steps_taken'] < 401
        sim_stats_df['Tmin_over_T'] = sim_stats_df['opt_steps'] / sim_stats_df['steps_taken']
        sim_stats_df['extra_steps'] = sim_stats_df['steps_taken'] - sim_stats_df['opt_steps']

        sim_stats_df.to_csv(sim_stats_folder + f'/run-{i}-shift{shift}-sim_stats.csv')

        # Saving simulations
        log('Saving simulation logs')
        save_simulations_to_csv(sim_folder + f'/run-{i}-shift{shift}-sims.csv', all_sims)

        # Refresh memory
        cp._default_memory_pool.free_all_blocks()

        # Print average extra steps
        print(f'\nAverage Extra steps count: {sim_stats_df["extra_steps"].mean()}')
        print('\n\n')

    print('--------------------------------------------------------------------------------')
    print(f'DONE')
    print('--------------------------------------------------------------------------------')