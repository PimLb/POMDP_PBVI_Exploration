# Imports
from test_setups import run_solve_test

import numpy as np

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(0)

    run_solve_test(
        model_file='./Models/Alt_Wrap_GroundOnly.pck',
        expand_function='fsvi',
        runs=20,
        sim_starts=[
            (361*30)+300, # Center
            (361*15)+300, # Above center
            (361*45)+300  # Bellow center
        ],
        name='Ground'
    )


if __name__ == "__main__":
    main()
