# Imports
from test_setups import run_single_solve_test

import numpy as np

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(3)

    run_single_solve_test(
        model_file='./Models/Alt_WrapVert_GroundAir.pck',
        expand_function='fsvi',
        runs=30,
        sim_starts=[
            (361*30)+300, # Center
            (361*15)+300, # Above center
            (361*45)+300  # Bellow center
        ],
        name='ProgVertGroundAir'
    )


if __name__ == "__main__":
    main()