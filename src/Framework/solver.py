import datetime

from src.MDP.mdp_model import MDP_Model
from src.POMDP.value_function import ValueFunction

class Solver:
    def __init__(self, model:MDP_Model):
        self.model = model

        self._solved = False
        self._solve_run_ts = datetime.datetime.min
        self._solve_steps_count = 0
        self._solve_history = []


    def solve(self):
        pass

    
    @property
    def solution(self) -> ValueFunction:
        assert self._solved, "solve() has to be run first..."
        return self._solve_history[-1]['value_functions']

