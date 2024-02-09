# Imports
import sys
sys.path.append('../..')
from src.pomdp import *
from test_setups import run_grid_test

import numpy as np

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(0)

    run_grid_test(
        model_file='./Models/Alt_WrapVert_GroundOnly.pck',
        folder='./Test_ProgVertGround_fsvi_300it_100exp_099g_1e-06eps_30run_300sim_20231126_010508/'
    )

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    run_grid_test(
        model_file='./Models/Alt_WrapVert_GroundAir.pck',
        folder='./Test_ProgVertGroundAir_fsvi_300it_100exp_099g_1e-06eps_30run_300sim_20231126_010553/'
    )

    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    run_grid_test(
        model_file='./Models/Alt_WrapVert_GroundOnly.pck',
        folder='./Test_VertGround_fsvi_300it_100exp_099g_1e-06eps_20run_300sim_20231125_230820/'
    )
    
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    run_grid_test(
        model_file='./Models/Alt_WrapVert_GroundAir.pck',
        folder='./Test_VertGroundAir_fsvi_300it_100exp_099g_1e-06eps_20run_300sim_20231125_230106/'
    )


if __name__ == "__main__":
    main()