import numpy as np
from typing import Union

class Model:
    '''
    POMDP Model class.

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
    observations:
        The observation matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.

    Methods
    -------
    transition(s:int, a:int):
        Returns a random state given a prior state and an action.
    observe(s_p:int, a:int):
        Returns a random observation based on the posterior state and the action that was taken.
    '''
    def __init__(self,
                 states:Union[int, list],
                 actions:Union[int, list],
                 transitions=None,
                 rewards=None,
                 observations=None
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
    

    def observe(self, s_p:int, a:int) -> int:
        '''
        Returns a random observation knowing action a is taken from state s, it is weighted by the observation probabilities.

                Parameters:
                        s_p (int): The state landed on after having done action a
                        a (int): The action to take

                Returns:
                        o (int): An observation
        '''
        o = int(np.argmax(np.random.multinomial(n=1, pvals=self.observation_table[s_p, a, :])))
        return o