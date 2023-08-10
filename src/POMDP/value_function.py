import copy
import numpy as np

from scipy.optimize import milp, LinearConstraint
from typing import Self

from src.POMDP.alpha_vector import AlphaVector

class ValueFunction(list[AlphaVector]):
    '''
    Class representing a set of AlphaVectors. One such set approximates the value function of the MDP model.

    ...

    Methods
    -------
    prune(level:int=1):
        Provides a ValueFunction where the alpha vector set is pruned with a certain level of pruning.
    '''

    def prune(self, level:int=1) -> Self:
        '''
        Function returning a new value function with the set of alpha vector composing it being it pruned.
        The pruning is as thorough as the level:
            - 0: No pruning, returns a value function with the alpha vector set being an exact copy of the current one.
            - 1: Simple deduplication of the alpha vectors.
            - 2: 1+ Check of absolute domination (check if dominated at each state).
            - 3: 2+ Solves Linear Programming problem for each alpha vector to see if it is dominated by combinations of other vectors.
        '''
        if level < 1:
            return copy.deepcopy(self)
        
        alpha_set = []

        # Level 1 pruning: Check for duplicates
        if level >= 1:
            L = {array.tobytes(): array for array in self}
            alpha_set = list(L.values())
        
        # Level 2 pruning: Check for absolute domination
        if level >= 2:
            new_alpha_set = []
            for alpha_vector in alpha_set:
                dominated = False
                for compare_alpha_vector in alpha_set:
                    if all(alpha_vector < compare_alpha_vector):
                        dominated = True
                if not dominated:
                    new_alpha_set.append(alpha_vector)
            alpha_set = new_alpha_set

        # Level 3 pruning: LP to check for more complex domination
        elif level >= 3:
            pruned_alphas = []
            state_count = alpha_set[0].shape[0]

            for i, alpha_vect in enumerate(alpha_set):
                other_alphas = alpha_set[:i] + alpha_set[(i+1):]

                # Objective function
                c = np.concatenate([np.array([1]), -1*alpha_vect])

                # Alpha vector contraints
                other_count = len(other_alphas)
                A = np.c_[np.ones(other_count), np.multiply(np.array(other_alphas), -1)]
                # b_l = np.zeros(other_count)
                # b_u = np.full_like(b_l, np.inf)
                alpha_constraints = LinearConstraint(A, 0, np.inf)

                # Constraints that sum of beliefs is 1
                belief_constraint = LinearConstraint(np.array([0] + ([1]*state_count)), 1, 1)

                # Solve problem
                res = milp(c=c, constraints=[alpha_constraints, belief_constraint])

                # Check if dominated
                is_dominated = (res.x[0] - np.dot(res.x[1:], alpha_vect)) >= 0
                if is_dominated:
                    print(alpha_vect)
                    print(' -> Dominated\n')

                else:
                    pruned_alphas.append(alpha_vect)

            alpha_set = pruned_alphas
        
        return ValueFunction(alpha_set)
