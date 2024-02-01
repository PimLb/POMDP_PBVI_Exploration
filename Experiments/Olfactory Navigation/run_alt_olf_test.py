# Imports
from test_setups import run_grid_test_alt

from datetime import datetime
import numpy as np
import os

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(2)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    test_folder = f"./Test_olf_nav_alt_{timestamp}"
    os.mkdir(test_folder)

    run_grid_test_alt(
        folder=test_folder
    )


if __name__ == "__main__":
    main()
