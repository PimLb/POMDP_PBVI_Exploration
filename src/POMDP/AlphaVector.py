import numpy as np
from typing import Union

class AlphaVector(np.ndarray):
    def __new__(cls, input_array, action:int):
        obj = np.asarray(input_array).view(cls)
        obj.action = action
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        assert obj['action'] is not None, "action parameter cannot be None"
        self.action = obj['action']