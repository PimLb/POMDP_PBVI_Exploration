import numpy as np
from . import Belief
from typing import Union

class Model:
    def __init__(self,
                 states:Union[int, list],
                 actions:Union[int, list],
                 transitions=None,
                 rewards=None,
                 observations=None,
                 initial_belief=None
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

        # Observations - specific to POMDPs 
        if observations is None:
            # If no observation matrix given, generate random one
            random_probs = np.random.rand(self.state_count, self.action_count, self.state_count)
            # Normalization to have s_p probabilies summing to 1
            self.observation_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.observation_table = np.array(observations)
            assert self.observation_table.shape == (self.state_count, self.action_count, self.state_count), "observations table doesnt have the right shape, it should be SxAxS"

        # Initial belief
        if initial_belief is None:
            self.initial_belief = Belief(self)
        else:
            self.initial_belief = Belief(self, np.array(initial_belief))
            assert self.initial_belief.shape == (self.state_count), "initial_belief needs to be an array of length S"

    
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
    

    def observe(self, s:int, a:int) -> int:
        '''
        Returns a random observation knowing action a is taken from state s, it is weighted by the observation probabilities.

                Parameters:
                        s (int): The current state
                        a (int): The action to take

                Returns:
                        o (int): An observation
        '''
        o = int(np.argmax(np.random.multinomial(n=1, pvals=self.observation_table[s, a, :])))
        return o