import sys
from typing import Union

from numpy import ndarray
sys.path.append('../..')
from src.pomdp import *


class RealSimulationAlt(Simulation):
    '''
    Version of the simulator for the alternation model (reproducing the results of the Rigolli N. et al paper) that uses real simulation sequence files to gather observations.
    The nose and ground files have to
    '''
    def __init__(self,
                 model:Model,
                 nose_file:str|np.ndarray,
                 ground_file:str|np.ndarray,
                 shift:int=0
                 ) -> None:
        self.iter = None
        self.shift = shift

        # Processing simulation matrices
        nose_data = np.load(nose_file) if isinstance(nose_file, str) else nose_file
        ground_data = np.load(ground_file) if isinstance(ground_file, str) else ground_file

        assert nose_data.shape == (nose_data.shape[0], 31, 121)
        assert ground_data.shape == (ground_data.shape[0], 31, 121)

        self.nose_resized = np.zeros((nose_data.shape[0], *model.state_grid.shape))
        self.ground_resized = np.zeros((ground_data.shape[0], *model.state_grid.shape))

        self.nose_resized[:,15:46,60:181] = nose_data
        self.ground_resized[:,15:46,60:181] = ground_data

        super().__init__(model)


    def initialize_simulation(self, start_state: int | None = None) -> int:
        self.iter = 0
        return super().initialize_simulation(start_state)


    def run_action(self, a: int) -> tuple[int | float, int]:
        r,_ = super().run_action(a)

        x,y = self.model.get_coords(self.agent_state)
        o = 1 if (self.nose_resized if a == 5 else self.ground_resized)[self.shift + self.iter, x, y] else 0
        if self.agent_state in self.model.end_states:
            o = 2

        self.iter += 1
        return (r,o)
    

class RealSimulationSetAlt(SimulationSet):
    '''
    Same as the RealSimulationAlt but parallel
    '''
    def __init__(self,
                 model:Model,
                 nose_file:str|np.ndarray,
                 ground_file:str|np.ndarray,
                 shift:int=0
                 ) -> None:
        self.iter = None
        self.shift = shift

        # Processing simulation matrices
        nose_data = np.load(nose_file) if isinstance(nose_file, str) else nose_file
        ground_data = np.load(ground_file) if isinstance(ground_file, str) else ground_file

        assert nose_data.shape == (nose_data.shape[0], 31, 121)
        assert ground_data.shape == (ground_data.shape[0], 31, 121)

        self.nose_resized = np.zeros((nose_data.shape[0], *model.state_grid.shape))
        self.ground_resized = np.zeros((ground_data.shape[0], *model.state_grid.shape))

        self.nose_resized[:,15:46,60:181] = nose_data > 3e-6
        self.ground_resized[:,15:46,60:181] = ground_data > 3e-6

        self.nose_resized = np.reshape(self.nose_resized, (self.nose_resized.shape[0], -1)).astype(int)
        self.ground_resized = np.reshape(self.ground_resized, (self.ground_resized.shape[0], -1)).astype(int)

        super().__init__(model)


    def initialize_simulations(self, n: int = 1, start_state: list[int] | int | None = None) -> ndarray:
        self.iter = 0

        if self.model.is_on_gpu:
            self.nose_resized = cp.array(self.nose_resized)
            self.ground_resized = cp.array(self.ground_resized)

        return super().initialize_simulations(n, start_state)


    def run_actions(self, actions: ndarray) -> tuple[ndarray, ndarray]:
        rewards, _ = super().run_actions(actions)

        # GPU support
        xp = np if not self.model.is_on_gpu else cp

        # Generate observations
        observations = xp.where(actions == 5, self.nose_resized[self.shift + self.iter, self.agent_states], self.ground_resized[self.shift + self.iter, self.agent_states])

        # Check if at goal
        observations = xp.where(xp.isin(self.agent_states, xp.array(self.model.end_states)), 2, observations)

        self.iter += 1
        return rewards, observations


class RealSimulationSetQComp(SimulationSet):
    '''
    Version of the simulator set for the model comparing the implementation of Q learning
    '''
    def __init__(self,
                 model:Model,
                 file:str|np.ndarray,
                 shift:int|list|np.ndarray=0
                 ) -> None:
        self.iter = None
        self.shift = shift

        # Processing simulation matrices
        data = np.load(file) if isinstance(file, str) else file

        assert data.shape == (data.shape[0], 27, 123)
        self.data_resized = np.zeros((data.shape[0], *model.state_grid.shape))
        self.data_resized[:,14:41,62:185] = data > 3e-6
        self.data_resized = np.reshape(self.data_resized, (self.data_resized.shape[0], -1)).astype(int)

        super().__init__(model)


    def initialize_simulations(self, n: int = 1, start_state: list[int] | int | None = None) -> ndarray:
        self.iter = 0

        if not isinstance(self.shift, int):
            assert len(self.shift) == n, "If for shift, an array is provided, it has to match the argument n"
        
        if self.model.is_on_gpu:
            self.data_resized = cp.array(self.data_resized)
            self.shift = cp.array(self.shift)
        else:
            self.shift = np.array(self.shift)
        
        return super().initialize_simulations(n, start_state)


    def run_actions(self, actions: ndarray) -> tuple[ndarray, ndarray]:
        rewards, _ = super().run_actions(actions)

        # GPU support
        xp = np if not self.model.is_on_gpu else cp

        # Generate observations
        observations = self.data_resized[self.shift[self.simulations] + self.iter, self.agent_states]

        # Check if at goal
        observations = xp.where(xp.isin(self.agent_states, xp.array(self.model.end_states)), 2, observations)

        self.iter += 1
        return rewards, observations


class SimulationSetAltProb(SimulationSet):
    '''
    An alternative to the SimulationSet to have an alternate observation probabilities
    '''
    def __init__(self, model:Model, alt_air:np.ndarray, alt_ground:np.ndarray):
        self.alt_air = alt_air.ravel()
        self.alt_ground = alt_ground.ravel()
        super().__init__(model)

    
    def run_actions(self, actions:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rewards,_ =  super().run_actions(actions)

        # GPU support
        xp = np if not self.model.is_on_gpu else cp

        # Generate observations
        obs_probs = xp.where(actions == 5, self.alt_air[self.agent_states], self.alt_ground[self.agent_states])
        observations = (xp.random.random(self.n) < obs_probs).astype(int)

        # Check if at goal
        observations = xp.where(xp.isin(self.agent_states, xp.array(self.model.end_states)), 2, observations)

        return rewards, observations