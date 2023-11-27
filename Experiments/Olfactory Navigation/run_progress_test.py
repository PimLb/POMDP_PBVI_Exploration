# Imports
from test_setups import run_single_solve_test

import numpy as np

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(3)

    run_single_solve_test(
        model_file='./Models/Alt_Wrap_GroundAir.pck',
        expand_function='fsvi',
        expansions=600,
        runs=60,
        simulations=0,
        sim_starts=[
            (361*30)+300, # Center
            (361*15)+300, # Above center
            (361*45)+300  # Bellow center
        ],
        use_gpu=False,
        name='LongerGroundAir'
    )


if __name__ == "__main__":
    main()