# Imports
import sys
sys.path.append('../..')
from src.pomdp import *

import pandas as pd
import numpy as np
import cupy as cp
import json
import os

from cupy.cuda import runtime as cuda_runtime
from csv import writer as csv_writer
from datetime import datetime

def reward_func(s,a,sn,o):
    return np.where(sn == 10890, 1.0, 0.0)

def main():
    # Set GPU used
    cuda_runtime.setDevice(1)

    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # TEST_COUNT = 100
    # MODEL_FILE = './Models/Alt_Wrap_GroundAir.pck'
    # SIM_FOLDER = f'{timestamp}_test_simulations_ground_air'
    # EXTRA_STEP_FILE = f'./{timestamp}-extra_steps_ground_air.csv'

    # # Creating simulation dir
    # os.mkdir(SIM_FOLDER)

    # # Create extra steps file
    # with open(EXTRA_STEP_FILE, 'w') as f_object:
    #     writer_object = csv_writer(f_object)
    #     writer_object.writerow(['Average'] + [f'Sim-{i}' for i in range(300)])


    def run(iter):
        print('--------------------------------------------------------------------------------')
        print(f'Run {iter}')
        print('--------------------------------------------------------------------------------')

        
        # Loading model
        model = Model.load_from_file(MODEL_FILE)

        # Solving model
        fsvi_solver = FSVI_Solver(0.99, eps=1e-6)
        fsvi_solution, hist = fsvi_solver.solve(model=model,
                                        expansions=300,
                                        max_belief_growth=100,
                                        print_progress=False,
                                        use_gpu=True)
        print(hist.summary)

        # Running simulation
        print()
        log('Starting simulations')
        a = Agent(model, fsvi_solution)
        all_rewards, all_sim_histories = a.run_n_simulations_parallel(
            n=300,
            print_progress=False,
            start_state=[
            (361*30)+300, # Center
            (361*15)+300, # Above center
            (361*45)+300  # Bellow center
        ])

        # Computing extra steps
        print()
        all_length = [len(sim) for sim in all_sim_histories]
        opt_traj = np.ones(300) * 240
        opt_traj[100:] += 15

        extra_steps = (np.array(all_length) - opt_traj)
        log(f'Average Extra steps count: {np.average(extra_steps)}')

        # Saving extra step results
        with open(EXTRA_STEP_FILE, 'a') as f_object:
            writer_object = csv_writer(f_object)
            writer_object.writerow([np.average(extra_steps)] + extra_steps.tolist())
        
        # Saving simulations
        log('Saving results')
        all_seq = np.empty((300,1001), dtype=object)
        for sim_i, sim in enumerate(all_sim_histories):
            seq = []
            for s, a, o, r in zip(sim.states, sim.actions+[], sim.observations+[], sim.rewards+[]):
                seq.append(json.dumps({'s':s, 'a':a, 'o':o, 'r':r}))
            
            all_seq[sim_i, :len(seq)] = seq

        sim_df = pd.DataFrame(all_seq.T, columns=[f'Sim-{sim_i}' for sim_i in range(len(all_sim_histories))])
        sim_df.to_csv(f'./{SIM_FOLDER}/run-{iter}-sims.csv')

        print('\n\n')

    # # Actual test loop
    # for i in range(TEST_COUNT):
    #     try:
    #         run(i)
    #     except:
    #         print()
    #         log('/!\\ Error Happend /!\\ \n\n')
    #     cp._default_memory_pool.free_all_blocks()

    '''
    ----------------------------------------------------------------------------------------------------------------
    GROUND ONLY
    ----------------------------------------------------------------------------------------------------------------
    '''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    TEST_COUNT = 20
    MODEL_FILE = './Models/Alt_Wrap_GroundOnly.pck'
    SIM_FOLDER = f'{timestamp}_test_simulations_ground_only'
    EXTRA_STEP_FILE = f'./{timestamp}-extra_steps_ground_only.csv'

    # Creating simulation dir
    os.mkdir(SIM_FOLDER)

    # Create extra steps file
    with open(EXTRA_STEP_FILE, 'w') as f_object:
        writer_object = csv_writer(f_object)
        writer_object.writerow(['Average'] + [f'Sim-{i}' for i in range(300)])
    
    # Actual test loop
    for i in range(TEST_COUNT):
        try:
            run(i)
        except Exception as e:
            print()
            print(e)
            log('/!\\ Error Happend /!\\ \n\n')
        cp._default_memory_pool.free_all_blocks()
    
    print('\n\n')
    print('--------------------------------------------------------------------------------')
    print(f'DONE')
    print('--------------------------------------------------------------------------------')


if __name__ == "__main__":
    main()