# Imports
import sys
sys.path.append('../..')
from src.pomdp import *
from test_setups import grid_test

import numpy as np

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(1)

    MODEL_FILE = './Models/Alt_Wrap_GroundAir.pck'
    FOLDER = './Test_GroundAir_FSVI_300it_100exp_099g_e6eps_20run_20231121_165329/'
    VALUE_FUNCTIONS = 20

    for i in range(VALUE_FUNCTIONS):
        print('--------------------------------------------------------------------------------')
        print(f'Run {i+1} of {VALUE_FUNCTIONS} (Run-{i})')
        print('--------------------------------------------------------------------------------')

        model = Model.load_from_file(MODEL_FILE)

        log(f'Loading Value function {i}')
        vf = ValueFunction.load_from_file(FOLDER + f'ValueFunctions/run-{i}-VF.gzip', model)

        # Run grid test
        log('Starting simulations')
        res_df = grid_test(vf.to_gpu(), points_per_cell=20)

        # Save results
        res_df.to_csv(FOLDER + f'GridSimulations/Grid-run-{i}-{len(res_df)}sims', index=False)

        # Refresh memory
        cp._default_memory_pool.free_all_blocks()

        print('\n\n')


if __name__ == "__main__":
    main()