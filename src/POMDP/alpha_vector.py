import numpy as np
from typing import Union

class AlphaVector(np.ndarray):
    '''
    A class to represent an Alpha Vector, a vector representing a plane in |S| dimension for POMDP models.

    ...

    Attributes
    ----------
    input_array: 
        The actual vector with the value for each state.
    action: int
        The action associated with the vector.
    '''
    def __new__(cls, input_array, action:int):
        obj = np.asarray(input_array).view(cls)
        obj._action = action
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._action = getattr(obj, '_action', None)

    @property
    def action(self) -> int:
        assert self._action is not None
        return self._action


# TESTING
def main():
    a = AlphaVector(np.zeros(3), 0)
    print(a.action)


if __name__ == "__main__":
    main()