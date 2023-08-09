import numpy as np
from typing import Union

from src.MDP.mdp_model import MDPModel

class POMDPModel(MDPModel):
    '''
    POMDP Model class. Partially Observable Markov Decision Process Model.

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
        
        super(POMDPModel, self).__init__(states, actions, transitions, rewards)

        # Observations - specific to POMDPs 
        if observations is None:
            # If no observation matrix given, generate random one
            random_probs = np.random.rand(self.state_count, self.action_count, self.state_count)
            # Normalization to have s_p probabilies summing to 1
            self.observation_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.observation_table = np.array(observations)
            assert self.observation_table.shape == (self.state_count, self.action_count, self.state_count), "observations table doesnt have the right shape, it should be SxAxS"
    

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