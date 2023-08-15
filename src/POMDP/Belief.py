import numpy as np
from typing import Self,Union

from src.POMDP.pomdp_model import POMDP_Model

class Belief(np.ndarray):
    '''
    A class representing a belief in the space of a given model. It is the belief to be in any combination of states:
    eg:
        - In a 2 state POMDP: a belief of (0.5, 0.5) represent the complete ignorance of which state we are in. Where a (1.0, 0.0) belief is the certainty to be in state 0.

    ...

    Attributes
    ----------
    model: Model
        The model on which the belief applies on.
    state_probabilities: np.ndarray|None
        A vector of the probabilities to be in each state of the model. The sum of the probabilities must sum to 1. If not specifies it will be 1/|S| for each state belief.

    Methods
    -------
    update(a:int, o:int):
        Function to provide a new Belief object updated using an action 'a' and observation 'o'.
    random_state():
        Function to give a random state based with the belief as the probability distribution. 
    '''
    def __new__(cls, model:POMDP_Model, state_probabilities:Union[np.ndarray,None]=None):
        if state_probabilities is not None:
            assert state_probabilities.shape[0] == model.state_count, "Belief must contain be of dimension |S|"
            prob_sum = np.round(sum(state_probabilities), decimals=3)
            assert prob_sum == 1, f"States probabilities in belief must sum to 1 (found: {prob_sum})"
            obj = np.asarray(state_probabilities).view(cls)
        else:
            obj = np.ones(model.state_count) / model.state_count
        obj._model = model # type: ignore
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._model = getattr(obj, '_model', None)

    @property
    def model(self) -> POMDP_Model:
        assert self._model is not None
        return self._model


    def update(self, a:int, o:int) -> Self:
        '''
        Returns a new belief based on this current belief, the most recent action (a) and the most recent observation (o).

                Parameters:
                        a (int): The most recent action
                        o (int): The most recent observation

                Returns:
                        new_belief (Belief): An updated belief

        '''
        new_state_probabilities = np.zeros((self.model.state_count))
        
        for s_p in self.model.states:
            new_state_probabilities[s_p] = sum([(self.model.observation_table[s_p, a, o] * self.model.transition_table[s, a, s_p] * self[s]) 
                              for s in self.model.states])

        # Formal definition of the normailizer as in paper
        # normalizer = 0
        # for s in STATES:
        #     normalizer += b_prev[s] * sum([(transition_function(s, a, o) * observation_function(a, s_p, o)) for s_p in STATES])
        new_state_probabilities /= np.sum(new_state_probabilities)
        new_belief = Belief(self.model, new_state_probabilities)
        return new_belief
    

    def random_state(self) -> int:
        '''
        Returns a random state of the model weighted by the belief probabily.

                Returns:
                        rand_s (int): A random state
        '''
        rand_s = int(np.argmax(np.random.multinomial(n=1, pvals=self)))
        return rand_s
