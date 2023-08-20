import random
from typing import Tuple, Union

import copy
import datetime
import numpy as np

from src.framework import Model as GeneralModel
from src.framework import AlphaVector, ValueFunction

class MDP_Model(GeneralModel):
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
        The reward matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.
    probabilistic_rewards: bool
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.

    Methods
    -------
    transition(s:int, a:int):
        Returns a random state given a prior state and an action.
    '''
    def __init__(self,
                 states:Union[int, list],
                 actions:Union[int, list],
                 transitions=None,
                 rewards=None,
                 probabilistic_rewards:bool=False
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
            self.reward_table = np.random.rand(self.state_count, self.action_count, self.state_count)
        else:
            self.reward_table = rewards
            assert self.reward_table.shape == (self.state_count, self.action_count, self.state_count), "rewards table doesnt have the right shape, it should be SxAxS"

        # Expected rewards
        self.expected_rewards_table = np.zeros((self.state_count, self.action_count))
        for s in self.states:
            for a in self.actions:
                self.expected_rewards_table[s,a] = np.dot(self.transition_table[s,a,:], self.reward_table[s,a,:])

        # Rewards are probabilistic
        self.probabilistic_rewards = probabilistic_rewards

    
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
    

    def reward(self, s:int, a:int, s_p:int) -> Union[int,float]:
        '''
        Returns the rewards of playing action a when in state s and landing in state s_p.
        If the rewards are probabilistic, it will return 0 or 1.

                Parameters:
                        s (int): The current state
                        a (int): The action taking in state s
                        s_p (int): The state landing in after taking action a in state s

                Returns:
                        reward (int, float): The reward received.
        '''
        reward = self.reward_table[s,a,s_p] 
        if self.probabilistic_rewards:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward


class MDP_SolverHistory(list[dict]):
    def __init__(self, model, **params):
        self.model = model
        self.params = params
        self.run_ts = datetime.datetime.now()


    @property
    def solution(self) -> ValueFunction:
        return self[-1]['value_function']


class VI_Solver:
    def __init__(self, model: MDP_Model):
        super().__init__()
        self.model = model


    def solve(self, horizon:int=10000, gamma:float=0.99, eps:float=0.001) -> tuple[ValueFunction, MDP_SolverHistory]:
        # Initiallize V as a |S| x |A| matrix of the reward expected when being in state s and taking action a
        V = ValueFunction([AlphaVector(self.model.expected_rewards_table[:,a], a) for a in self.model.actions])
        V_opt = V[0]
        converged = False

        solve_history = MDP_SolverHistory(self.model)
        solve_history.append({'value_function': V})

        while (not converged) and (len(solve_history) <= horizon):
            old_V_opt = copy.deepcopy(V_opt)

            V = []
            for a in self.model.actions:
                alpha_vect = []
                for s in self.model.states:
                    summer = sum(self.model.transition_table[s, a, s_p] * old_V_opt[s_p] for s_p in self.model.states)
                    alpha_vect.append(self.model.expected_rewards_table[s,a] + (gamma * summer))

                V.append(AlphaVector(alpha_vect, a))

            V_opt = np.max(np.array(V), axis=1)

            solve_history.append({'value_function': ValueFunction(V)})
                
            avg_delta = np.max(np.abs(V_opt - old_V_opt))
            if avg_delta < eps:
                break

        return ValueFunction(V), solve_history