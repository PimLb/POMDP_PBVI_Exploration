import numpy as np
from typing import Self,Union
from . import Model

class Belief(np.ndarray):

    def __new__(cls, model:Model, state_probabilities:Union[np.ndarray,None]=None):
        if state_probabilities is not None:
            assert state_probabilities.shape == (model.state_count), "Belief must contain be of dimension |S|"
            assert sum(state_probabilities) == 1, "States probabilities in belief must sum to 1"
            obj = np.asarray(state_probabilities).view(cls)
        else:
            obj = np.ones(model.state_count) / model.state_count
        obj.model = model # type: ignore
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        assert obj['model'] is not None, "model parameter cannot be None"
        self.model = obj['model']


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
            new_state_probabilities[s_p] = sum([(self.model.observation_table[s, a, o] * self.model.transition_table[s, a, s_p] * self[s]) 
                              for s in self.model.states])

        # Formal definition of the normailizer as in paper
        # normalizer = 0
        # for s in STATES:
        #     normalizer += b_prev[s] * sum([(transition_function(s, a, o) * observation_function(a, s_p, o)) for s_p in STATES])
        
        normalizer = sum(new_state_probabilities)
        
        for s in self.model.states:
            new_state_probabilities[s] /= normalizer
        
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
