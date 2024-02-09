# Imports
from test_setups import run_all_start_test_q

from datetime import datetime
import numpy as np
import os

from cupy.cuda import runtime as cuda_runtime



def main():
    # Set GPU used
    cuda_runtime.setDevice(3)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    test_folder = f"./Test_Results/Test_Q_Learning_Comparison_{timestamp}"
    os.mkdir(test_folder)

    run_all_start_test_q(
        folder=test_folder
    )


if __name__ == "__main__":
    main()