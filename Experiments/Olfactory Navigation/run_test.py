# Imports
import sys
sys.path.append('../..')
from src.pomdp import *

import pandas as pd
import numpy as np
import cupy as cp
import json
import datetime

from cupy.cuda import runtime as cuda_runtime
from csv import writer as csv_writer


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    TEST_COUNT = 10
    MODEL_FILE = 'Alt_Wrap_GroundAir.pck'
    SIM_FOLDER = f'{timestamp}_test_simulations'
    EXTRA_STEP_FILE = f'./{timestamp}-extra_steps.csv'

    # Set GPU used
    cuda_runtime.setDevice(1)

    # Create extra steps file        
    with open(EXTRA_STEP_FILE, 'w') as f_object:
        writer_object = csv_writer(f_object)
        writer_object.writerow(['Average'] + [f'Sim-{i}' for i in range(300)])


    for i in range(TEST_COUNT):
        print('--------------------------------------------------------------------------------')
        print(f'Run {i}')
        print('--------------------------------------------------------------------------------')

        
        # Loading model
        model = Model.load_from_file(MODEL_FILE)

        # Solving model
        fsvi_solver = FSVI_Solver(0.99, eps=1e-6)
        fsvi_solution, hist = fsvi_solver.solve(model=model,
                                        expansions=300,
                                        max_belief_growth=100,
                                        # prune_level=2, # Useless because of belief domination
                                        # prune_interval=25,
                                        # history_tracking_level=2,
                                        use_gpu=True)
        print(hist.summary)

        # Running simulation
        a = Agent(model, fsvi_solution)
        all_rewards, all_sim_histories = a.run_n_simulations_parallel(
            n=300, 
            start_state=[
            (361*30)+300, # Center
            (361*15)+300, # Above center
            (361*45)+300  # Bellow center
        ])

        # Saving simulations
        all_length = [len(sim) for sim in all_sim_histories]
        max_length = max(all_length)
        all_cols = {}
        for i, sim in enumerate(all_sim_histories):
            seq = []
            for s, a, o, r in zip(sim.states, sim.actions+[], sim.observations+[], sim.rewards+[]):
                seq.append(json.dumps({'s':s, 'a':a, 'o':o, 'r':r}))
            
            all_cols[f'Sim-{i}'] = seq + ((max_length + 1 - len(sim)) * [None])

        sim_df = pd.DataFrame(all_cols)
        sim_df.to_csv(f'./{SIM_FOLDER}/run-{i}-sims.csv')

        # Processing results
        opt_traj = np.ones(300) * 240
        opt_traj[100:] += 15

        extra_steps = (np.array(all_length) - opt_traj)
        

        with open(EXTRA_STEP_FILE, 'a') as f_object:
            writer_object = csv_writer(f_object)
            writer_object.writerow([np.average(extra_steps)] + extra_steps.tolist())

if __name__ == "__main__":
    main()