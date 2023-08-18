from typing import Union

import copy
import datetime
import numpy as np

from src.framework import AlphaVector, ValueFunction, Solver

class Model:
    '''
    MDP Model class.

    ...

    Attributes
    ----------
    states: int|list
        A list of state labels or an amount of states to be used.
    actions: int|list
        A list of action labels or an amount of actions to be used.
    transitions:
        The transition matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.
    rewards:
        The reward matrix, has to be |S| x |A|. If none is provided, it will be randomly generated.

    Methods
    -------
    transition(s:int, a:int):
        Returns a random state given a prior state and an action.
    '''
    def __init__(self,
                 states:Union[int, list],
                 actions:Union[int, list],
                 transitions=None,
                 rewards=None
                 ):
        
        # States
        if isinstance(states, int):
            self.state_labels = [f's_{i}' for i in range(states)]
        else:
            self.state_labels = states
        self.state_count = len(self.state_labels)
        self.states = [state for state in range(self.state_count)]

        # Actions
        if isinstance(actions, int):
            self.action_labels = [f'a_{i}' for i in range(actions)]
        else:
            self.action_labels = actions
        self.action_count = len(self.action_labels)
        self.actions = [action for action in range(self.action_count)]

        # Transitions
        if transitions is None:
            # If no transitiong matrix given, generate random one
            random_probs = np.random.rand(self.state_count, self.action_count, self.state_count)
            # Normalization to have s_p probabilies summing to 1
            self.transition_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.transition_table = np.array(transitions)
            assert self.transition_table.shape == (self.state_count, self.action_count, self.state_count), "transitions table doesnt have the right shape, it should be SxAxS"

        # Rewards
        if rewards is None:
            # If no reward matrix given, generate random one
            self.reward_table = np.random.rand(self.state_count, self.action_count)
        else:
            self.reward_table = rewards
            assert self.reward_table.shape == (self.state_count, self.action_count), "rewards table doesnt have the right shape, it should be SxA"

    
    def transition(self, s:int, a:int) -> int:
        '''
        Returns a random posterior state knowing we take action a in state t and weighted on the transition probabilities.

                Parameters:
                        s (int): The current state
                        a (int): The action to take

                Returns:
                        s_p (int): The posterior state
        '''
        s_p = int(np.argmax(np.random.multinomial(n=1, pvals=self.transition_table[s, a, :])))
        return s_p
    

class VI_Solver(Solver):
    def __init__(self, model: Model):
        super().__init__()
        self.model = model


    def solve(self, horizon:int=10000, gamma:float=0.99, eps:float=0.001):
        # Initiallize V as a |S| x |A| matrix of the reward expected when being in state s and taking action a
        V = ValueFunction([AlphaVector(self.model.reward_table[:,a], a) for a in self.model.actions])
        V_opt = V[0]
        converged = False

        self._solve_run_ts = datetime.datetime.now()
        self._solve_steps_count = 0
        self._solve_history = [{'value_function': V}]

        while (not converged) and (self._solve_steps_count < horizon):
            self._solve_steps_count += 1
            
            old_V_opt = copy.deepcopy(V_opt)

            V = []
            for a in self.model.actions:
                alpha_vect = []
                for s in self.model.states:
                    summer = sum(self.model.transition_table[s, a, s_p] * old_V_opt[s_p] for s_p in self.model.states)
                    alpha_vect.append(self.model.reward_table[s,a] + (gamma * summer))

                V.append(AlphaVector(alpha_vect, a))

            V_opt = np.max(np.array(V), axis=1)

            self._solve_history.append({'value_function': ValueFunction(V)})
                
            avg_delta = np.max(np.abs(V_opt - old_V_opt))
            if avg_delta < eps:
                return ValueFunction(V)