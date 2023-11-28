# Imports
import sys
sys.path.append('../..')
from src.pomdp import *
from test_setups import run_grid_test

import numpy as np

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(2)

    run_grid_test(
        model_file='./Models/Alt_Wrap_GroundOnly.pck',
        folder='./Test_ProgGround_fsvi_300it_100exp_099g_1e-06eps_30run_300sim_20231126_010344/'
    )


if __name__ == "__main__":
    main()