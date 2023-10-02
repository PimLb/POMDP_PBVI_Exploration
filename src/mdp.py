from datetime import datetime
from inspect import signature
from matplotlib import animation, colors, patches
from matplotlib import pyplot as plt
from scipy.optimize import milp, LinearConstraint
from scipy.spatial.distance import cdist
from tqdm.auto import trange
from typing import Self, Union, Tuple

import copy
import json
import os
import pandas as pd
import random

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')


COLOR_LIST = [{
    'name': item.replace('tab:',''),
    'id': item,
    'hex': value,
    'rgb': [int(value.lstrip('#')[i:i + (len(value)-1) // 3], 16) for i in range(0, (len(value)-1), (len(value)-1) // 3)]
    } for item, value in colors.TABLEAU_COLORS.items()] # type: ignore

COLOR_ARRAY = np.array([c['rgb'] for c in COLOR_LIST])


def log(content:str) -> None:
    '''
    Function to print a log line with a timestamp 
    '''
    print(f'[{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}] ' + content)


class Model:
    '''
    MDP Model class.

    ...

    Parameters
    ----------
    states: int | list[str] | list[list[str]]
        A list of state labels or an amount of states to be used. Also allows to provide a matrix of states to define a grid model.
    actions: int|list
        A list of action labels or an amount of actions to be used.
    transitions:
        The transitions between states, an array can be provided and has to be |S| x |A| x |S| or a function can be provided. 
        If a function is provided, it has be able to deal with np.array arguments.
        If none is provided, it will be randomly generated.
    reachable_states:
        A list of states that can be reached from each state and actions. It must be a matrix of size |S| x |A| x |R| where |R| is the max amount of states reachable from any given state and action pair.
        It is optional but useful for speedup purposes.
    rewards:
        The reward matrix, has to be |S| x |A| x |S|.
        A function can also be provided here but it has to be able to deal with np.array arguments.
        If provided, it will be use in combination with the transition matrix to fill to expected rewards.
    rewards_are_probabilistic: bool
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.
    grid_states:
        Optional, if provided, the model will be converted to a grid model.
    start_probabilities: list
        Optional, the distribution of chances to start in each state. If not provided, there will be an uniform chance for each state.
    end_states: list
        Optional, entering either state in the list during a simulation will end the simulation.
    end_action: list
        Optional, playing action of the list during a simulation will end the simulation.
    
    Attributes # TODO: update this
    ----------

        
    Methods
    -------
    transition(s:int, a:int):
        Returns a random state given a prior state and an action.
    '''
    def __init__(self,
                 states:Union[int, list[str], list[list[str]]],
                 actions:Union[int, list],
                 transitions=None,
                 reachable_states=None,
                 rewards=None,
                 rewards_are_probabilistic:bool=False,
                 state_grid=None,
                 start_probabilities:Union[list,None]=None,
                 end_states:list[int]=[],
                 end_actions:list[int]=[]
                 ):
        
        # Empty variable
        self._alt_model = None
        self.is_on_gpu = False
        
        log('Instantiation of MDP Model:')
        
        # ------------------------- States -------------------------
        self.state_grid = None
        if isinstance(states, int): # State count
            self.state_labels = [f's_{i}' for i in range(states)]

        elif isinstance(states, list) and all(isinstance(item, list) for item in states): # 2D list of states
            dim1 = len(states)
            dim2 = len(states[0])
            assert all(len(state_dim) == dim2 for state_dim in states), "All sublists of states must be of equal size"
            
            self.state_labels = []
            for state_dim in states:
                for state in state_dim:
                    self.state_labels.append(state)

            self.state_grid = np.arange(dim1 * dim2).reshape(dim1, dim2)

        else: # Default: single of list of string items
            self.state_labels = [item for item in states if isinstance(item, str)]

        self.state_count = len(self.state_labels)
        self.states = np.arange(self.state_count)

        log(f'- {self.state_count} states')

        # ------------------------- Actions -------------------------
        if isinstance(actions, int):
            self.action_labels = [f'a_{i}' for i in range(actions)]
        else:
            self.action_labels = actions
        self.action_count = len(self.action_labels)
        self.actions = np.arange(self.action_count)

        log(f'- {self.action_count} actions')

        # ------------------------- Reachable states provided -------------------------
        self.reachable_states = None
        if reachable_states is not None:
            self.reachable_states = np.array(reachable_states)
            assert self.reachable_states.shape[:2] == (self.state_count, self.action_count), f"Reachable states provided is not of the expected shape (received {self.reachable_states.shape}, expected ({self.state_count}, {self.action_count}, :))"
            self.reachable_state_count = self.reachable_states.shape[2]

            log(f'- At most {self.reachable_state_count} reachable states per state-action pair')

        # ------------------------- Transitions -------------------------
        log('- Starting generation of transitions table')
        start_ts = datetime.now()

        self.transition_table = None
        self.transition_function = None
        if transitions is None:
            if reachable_states is None:
                # If no transitiong matrix and no reachable states given, generate random one
                print('[Warning] No transition matrix and no reachable states have provided so a random transition matrix is generated...')
                random_probs = np.random.rand(self.state_count, self.action_count, self.state_count)

                # Normalization to have s_p probabilies summing to 1
                self.transition_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
            else:
                # Make uniform transition probabilities over reachable states
                print(f'[Warning] No transition matrix or function provided but reachable states are, so probability to reach any reachable states will "1 / reachable state count" so here: {1/self.reachable_state_count:.3f}.')

        elif callable(transitions): # Transition function
            self.transition_function = transitions
            # Attempt to create transition table in memory
            t_arr = None
            try:
                t_arr = np.fromfunction(self.transition_function, (self.state_count, self.action_count, self.state_count))
            except MemoryError:
                print('[Warning] Not enough memory to store transition table, using transition function provided...')
            else:
                self.transition_table = t_arr

        else: # Array like
            self.transition_table = np.array(transitions)
            t_shape = self.transition_table.shape
            exp_shape = (self.state_count, self.action_count, self.state_count)
            assert t_shape == exp_shape, f"Transitions table provided doesnt have the right shape, it should be SxAxS (expected {exp_shape}, received {t_shape})"

        duration = (datetime.now() - start_ts).total_seconds()
        log(f'    > Done in {duration:.3f}s')
        if duration > 1:
            log(f'    > /!\\ Transition table generation took long, if not done already, try to use the reachable_states parameter to speedup the process.')

        # ------------------------- Rewards are probabilistic toggle -------------------------
        self.rewards_are_probabilistic = rewards_are_probabilistic

        # ------------------------- State grid -------------------------
        log('- Generation of state grid')
        if state_grid is None and self.state_grid is None:
            self.state_grid = np.arange(self.state_count).reshape((1,self.state_count))
        
        elif state_grid is not None:
            assert all(isinstance(l, list) for l in state_grid), "The provided states grid must be a list of lists."

            grid_shape = (len(state_grid), len(state_grid[0]))
            assert all(len(l) == grid_shape[1] for l in state_grid), "All rows must have the same length."

            if all(all(isinstance(e, int) for e in l) for l in state_grid):
                state_grid = np.array(state_grid)
                try:
                    self.states[state_grid]
                except:
                    raise Exception('An error occured with the list of state indices provided...')
                else:
                    self.state_grid = state_grid

            else:
                log('    > Warning: looping through all grid states provided to find the corresponding states, can take a while...')
                
                np_state_grid = np.zeros(grid_shape, dtype=int)
                states_covered = 0
                for i, row in enumerate(state_grid):
                    for j, element in enumerate(state_grid):
                        if isinstance(element, str) and (element in self.state_labels):
                            states_covered += 1
                            np_state_grid[i,j] = self.state_labels.index(element)
                        elif isinstance(element, int) and (element < self.state_count):
                            np_state_grid[i,j] = element
                        
                        else:
                            raise Exception(f'Countains a state (\'{state}\') not in the list of states...')

                assert states_covered == self.state_count, "Some states of the state list are missing..."

        # ------------------------- Start state probabilities -------------------------
        log('- Generating start probabilities table')
        if start_probabilities is not None:
            assert len(start_probabilities) == self.state_count
            self.start_probabilities = np.array(start_probabilities,dtype=float)
        else:
            self.start_probabilities = np.full((self.state_count), 1/self.state_count)

        # ------------------------- End state conditions -------------------------
        self.end_states = end_states
        self.end_actions = end_actions
        
        # ------------------------- Reachable states -------------------------
        # If not set yet
        if self.reachable_states is None:
            log('- Starting computation of reachable states from transition data')
            
            if self.state_count > 1000:
                log('-    > Warning: For models with large amounts of states, this operation can take time. Try generating it advance and use the parameter \'reachable_states\'...')
            
            start_ts = datetime.now()

            self.reachable_states = []
            self.reachable_state_count = 0
            for s in self.states:
                reachable_states_for_action = []
                for a in self.actions:
                    reachable_list = []
                    if self.transition_table is not None:
                        reachable_list = np.argwhere(self.transition_table[s,a,:] > 0)[:,0].tolist()
                    else:
                        for sn in self.states:
                            if self.transition_function(s,a,sn) > 0:
                                reachable_list.append(sn)
                    reachable_states_for_action.append(reachable_list)
                    
                    if len(reachable_list) > self.reachable_state_count:
                        self.reachable_state_count = len(reachable_list)

                self.reachable_states.append(reachable_states_for_action)

            # In case some state-action pairs lead to more states than other, we fill with the 1st non states not used
            for s in self.states:
                for a in self.actions:
                    to_add = 0
                    while len(self.reachable_states[s][a]) < self.reachable_state_count:
                        if to_add not in self.reachable_states[s][a]:
                            self.reachable_states[s][a].append(to_add)
                        to_add += 1

            # Converting to ndarray
            self.reachable_states = np.array(self.reachable_states, dtype=int)

            duration = (datetime.now() - start_ts).total_seconds()
            log(f'    > Done in {duration:.3f}s')
            log(f'- At most {self.reachable_state_count} reachable states per state-action pair')

        # ------------------------- Reachable state probabilities -------------------------
        log('- Starting computation of reachable state probabilities from transition data')
        start_ts = datetime.now()

        if self.transition_function is None and self.transition_table is None:
            self.reachable_probabilities = np.full(self.reachable_states.shape, 1/self.reachable_state_count)
        elif self.transition_table is not None:
            self.reachable_probabilities = self.transition_table[self.states[:,None,None], self.actions[None,:,None], self.reachable_states]
        else:
            self.reachable_probabilities = np.fromfunction((lambda s,a,ri: self.transition_function(s.astype(int), a.astype(int), self.reachable_states[s.astype(int), a.astype(int), ri.astype(int)])), self.reachable_states.shape)
            
        duration = (datetime.now() - start_ts).total_seconds()
        log(f'    > Done in {duration:.3f}s')

        # ------------------------- Rewards -------------------------
        self.immediate_reward_table = None
        self.immediate_reward_function = None
        if rewards == -1: # If -1 is set, it means the rewards are defined in the superclass POMDP
            pass
        elif rewards is None:
            # If no reward matrix given, generate random one
            self.immediate_reward_table = np.random.rand(self.state_count, self.action_count, self.state_count)
        elif callable(rewards):
            # Rewards is a function
            self.immediate_reward_function = rewards
            assert len(signature(rewards).parameters) == 3, "Reward function should accept 3 parameters: s, a, sn..."
        else:
            # Array like
            self.immediate_reward_table = np.array(rewards)
            r_shape = self.immediate_reward_table.shape
            exp_shape = (self.state_count, self.action_count, self.state_count)
            assert r_shape == exp_shape, f"Rewards table doesnt have the right shape, it should be SxAxS (expected: {exp_shape}, received {r_shape})"

        # ------------------------- Expected rewards -------------------------
        self.expected_rewards_table = None
        if rewards != -1:
            log('- Starting generation of expected rewards table')
            start_ts = datetime.now()

            reachable_rewards = None
            if self.immediate_reward_table is not None:
                reachable_rewards = self.immediate_reward_table[self.states[:,None,None], self.actions[None,:,None], self.reachable_states]
            else:
                def reach_reward_func(s,a,ri):
                    s = s.astype(int)
                    a = a.astype(int)
                    ri = ri.astype(int)
                    return self.immediate_reward_function(s,a,self.reachable_states[s,a,ri])
                
                reachable_rewards = np.fromfunction(reach_reward_func, self.reachable_states.shape)
                
            self.expected_rewards_table = np.einsum('sar,sar->sa', self.reachable_probabilities, reachable_rewards)

            duration = (datetime.now() - start_ts).total_seconds()
            log(f'    > Done in {duration:.3f}s')

    
    def transition(self, s:int, a:int) -> int:
        '''
        Returns a random posterior state knowing we take action a in state t and weighted on the transition probabilities.

                Parameters:
                        s (int): The current state
                        a (int): The action to take

                Returns:
                        s_p (int): The posterior state
        '''
        xp = cp if self.is_on_gpu else np
        s_p = int(xp.random.choice(a=self.reachable_states[s,a], size=1, p=self.reachable_probabilities[s,a])[0])
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
        reward = self.immediate_reward_table[s,a,s_p] if self.immediate_reward_table is not None else self.immediate_reward_function(s,a,s_p)
        if self.rewards_are_probabilistic:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward
    

    def save(self, file_name:str, path:str='./Models') -> None:
        '''
        Function to save the current model in a json file.
        By default, the model will be saved in 'Models' directory in the current working directory but this can be changed using the 'path' parameter.

                Parameters:
                        file_name (str): The name of the json file the model will be saved in.
                        path (str): The path at which the model will be saved. (Default: './Models')
        '''
        if not os.path.exists(path):
            print('Folder does not exist yet, creating it...')
            os.makedirs(path)

        if not file_name.endswith('.json'):
            file_name += '.json'

        # Taking the arguments of the CPU version of the model
        argument_dict = self.cpu_model.__dict__

        # Converting the numpy array to lists
        for k,v in argument_dict.items():
            argument_dict.__setattr__(k, v.tolist() if isinstance(v, np.ndarray) else v)

        json_object = json.dumps(argument_dict, indent=4)
        with open(path + '/' + file_name, 'w') as outfile:
            outfile.write(json_object)


    @classmethod
    def load_from_json(cls, file:str) -> Self:
        '''
        Function to load a MDP model from a json file. The json structure must contain the same items as in the constructor of this class.

                Parameters:
                        file (str): The file and path of the model to be loaded.
                Returns:
                        loaded_model (mdp.Model): An instance of the loaded model.
        '''
        with open(file, 'r') as openfile:
            json_model = json.load(openfile)

        loaded_model = super().__new__(cls)
        for k,v in json_model.items():
            loaded_model.__setattr__(k, np.array(v) if isinstance(v, list) else v)

        if 'grid_states' in json_model:
            loaded_model.convert_to_grid(json_model['grid_states'])

        return loaded_model


    @property
    def gpu_model(self) -> Self:
        '''
        The same model but on the GPU instead of the CPU. If already on the GPU, the current model object is returned.
        '''
        if self.is_on_gpu:
            return self
        
        assert gpu_support, "GPU Support is not available, try installing cupy..."
        
        if self._alt_model is None:
            log('Sending Model to GPU...')
            start = datetime.now()

            # Setting all the arguments of the new class and convert to cupy if numpy array
            new_model = super().__new__(self.__class__)
            for arg, val in self.__dict__.items():
                new_model.__setattr__(arg, cp.array(val) if isinstance(val, np.ndarray) else val)

            # GPU/CPU variables
            new_model.is_on_gpu = True
            new_model._alt_model = self
            self._alt_model = new_model
            
            duration = (datetime.now() - start).total_seconds()
            log(f'    > Done in {duration:.3f}s')

        return self._alt_model


    @property
    def cpu_model(self) -> Self:
        '''
        The same model but on the CPU instead of the GPU. If already on the CPU, the current model object is returned.
        '''
        if not self.is_on_gpu:
            return self
        
        assert gpu_support, "GPU Support is not available, try installing cupy..."

        if self._alt_model is None:
            log('Sending Model to CPU...')
            start = datetime.now()

            # Setting all the arguments of the new class and convert to numpy if cupy array
            new_model = super().__new__(self.__class__)
            for arg, val in self.__dict__.items():
                new_model.__setattr__(arg, cp.asnumpy(val) if isinstance(val, cp.ndarray) else val)
            
            # GPU/CPU variables
            new_model.is_on_gpu = False
            new_model._alt_model = self
            self._alt_model = new_model
            
            duration = (datetime.now() - start).total_seconds()
            log(f'    > Done in {duration:.3f}s')

        return self._alt_model


class AlphaVector:
    '''
    A class to represent an Alpha Vector, a vector representing a plane in |S| dimension for POMDP models.

    ...

    Parameters
    ----------
    values: np.ndarray
        The actual vector with the value for each state.
    action: int
        The action associated with the vector.
    '''
    def __init__(self, values:np.ndarray, action:int) -> None:
        self.values = values
        self.action = action


class ValueFunction:
    '''
    Class representing a set of AlphaVectors. One such set approximates the value function of the MDP model.

    ...

    Parameters
    ----------
    model: (mdp.Model)
        The model the value function is associated with.
    alpha_vectors: (list[AlphaVector] | np.ndarray) (Optional)
        The alpha vectors composing the value function, if none are provided, it will be empty to start with and AlphaVectors can be appended.
    action_list: (list[int])
        The actions associated with alpha vectors in the case the alpha vectors are provided as an numpy array.
    
    # TODO: Add list of attributes
    Methods # TODO: update list of functions
    -------
    prune(level:int=1):
        Provides a ValueFunction where the alpha vector set is pruned with a certain level of pruning.
    plot(size:int=5, state_list:list[str], action_list:list[str], belief_set):
        Function to plot the value function for 2- or 3-state models.
    '''
    def __init__(self, model:Model, alpha_vectors:Union[list[AlphaVector], np.ndarray]=[], action_list:list[int]=[]):
        self.model = model

        self._vector_list = None
        self._vector_array = None
        self._actions = None

        self.is_on_gpu = False

        # List of alpha vectors
        if isinstance(alpha_vectors, list):
            assert all(v.values.shape[0] == model.state_count for v in alpha_vectors), f"Some or all alpha vectors in the list provided dont have the right size, they should be of shape: {model.state_count}"
            self._vector_list = alpha_vectors
            
            # Check if on gpu and make sure all vectors are also on the gpu
            if (len(alpha_vectors) > 0) and gpu_support and cp.get_array_module(alpha_vectors[0].values) == cp:
                assert all(cp.get_array_module(v.values) == cp for v in alpha_vectors), "Either all or none of the alpha vectors should be on the GPU, not just some."
                self.is_on_gpu = True
        
        # As numpy array
        else:
            av_shape = alpha_vectors.shape
            exp_shape = (len(action_list), model.state_count)
            assert av_shape == exp_shape, f"Alpha vector array does not have the right shape (received: {av_shape}; expected: {exp_shape})"

            self._vector_array = alpha_vectors
            self._actions = action_list

            # Check if array is on gpu
            if gpu_support and cp.get_array_module(alpha_vectors) == cp:
                self.is_on_gpu = True


    @property
    def alpha_vector_list(self) -> list[AlphaVector]:
        '''
        A list of AlphaVector objects. If the value function is defined as an matrix of vectors and a list of actions, the list of AlphaVectors will be generated from them.
        '''
        if self._vector_list is None:
            self._vector_list = []
            for alpha_vect, action in zip(self._vector_array, self._actions):
                self._vector_list.append(AlphaVector(alpha_vect, action))
        return self._vector_list
    

    @property
    def alpha_vector_array(self) -> np.ndarray:
        '''
        A matrix of size N x S, containing all the alpha vectors making up the value function. (N is the number of alpha vectors and S the amount of states in the model)
        If the value function is defined as a list of AlphaVector objects, the matrix will the generated from them.
        '''
        xp = cp if (gpu_support and self.is_on_gpu) else np

        if self._vector_array is None:
            self._vector_array = xp.array([v.values for v in self._vector_list])
            self._actions = [v.action for v in self._vector_list]
        return self._vector_array
    

    @property
    def actions(self) -> list[int]:
        '''
        A list of N actions corresponding to the N alpha vectors making up the value function.
        If the value function is defined as a list of AlphaVector objects, the list will the generated from the actions of those alpha vector objects.
        '''
        xp = cp if (gpu_support and self.is_on_gpu) else np

        if self._actions is None:
            self._vector_array = xp.array(self._vector_list)
            self._actions = [v.action for v in self._vector_list]
        return self._actions
    

    def __len__(self) -> int:
        return len(self._vector_list) if self._vector_list is not None else self._vector_array.shape[0]
    

    def append(self, alpha_vector:AlphaVector) -> None:
        '''
        Function to add an alpha vector to the value function.
        '''
        # Make sure size is correct
        assert alpha_vector.values.shape[0] == self.model.state_count, f"Vector to add to value function doesn't have the right size (received: {alpha_vector.values.shape[0]}, expected: {self.model.state_count})"
        
        # GPU support check
        xp = cp if (gpu_support and self.is_on_gpu) else np
        assert gpu_support and cp.get_array_module(alpha_vector.values) == xp, f"Vector is{' not' if self.is_on_gpu else ''} on GPU while value function is{'' if self.is_on_gpu else ' not'}."

        if self._vector_array is not None:
            self._vector_array = xp.append(self._vector_array, alpha_vector[None,:], axis=0)
            self._actions.append(alpha_vector.action)
        
        if self._vector_list is not None:
            self._vector_list.append(alpha_vector)


    def to_gpu(self) -> Self:
        '''
        Function returning an equivalent value function object with the arrays stored on GPU instead of CPU.

                Returns:
                        gpu_value_function (ValueFunction): A new value function with arrays on GPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        gpu_model = self.model.gpu_model

        gpu_value_function = None
        if self._vector_array is not None:
            gpu_vector_array = cp.array(self._vector_array)
            gpu_actions = self._actions if isinstance(self._actions, list) else cp.array(self._actions)
            gpu_value_function = ValueFunction(gpu_model, gpu_vector_array, gpu_actions)
        
        else:
            gpu_alpha_vectors = [AlphaVector(cp.array(av.values), av.action) for av in self._vector_list]
            gpu_value_function = ValueFunction(gpu_model, gpu_alpha_vectors)

        return gpu_value_function
    

    def to_cpu(self) -> Self:
        '''
        Function returning an equivalent value function object with the arrays stored on CPU instead of GPU.

                Returns:
                        cpu_value_function (ValueFunction): A new value function with arrays on CPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        cpu_model = self.model.cpu_model

        cpu_value_function = None
        if self._vector_array is not None:
            cpu_vector_array = cp.asnumpy(self._vector_array)
            cpu_actions = self._actions if isinstance(self._actions, list) else cp.asnumpy(self._actions)
            cpu_value_function = ValueFunction(cpu_model, cpu_vector_array, cpu_actions)
        
        else:
            cpu_alpha_vectors = [AlphaVector(cp.asnumpy(av.values), av.action) for av in self._vector_list]
            cpu_value_function = ValueFunction(cpu_model, cpu_alpha_vectors)

        return cpu_value_function


    def prune(self, level:int=1) -> Self:
        '''
        Function returning a new value function with the set of alpha vector composing it being it pruned.
        The pruning is as thorough as the level:
            - 0: No pruning, returns a value function with the alpha vector set being an exact copy of the current one.
            - 1: Simple deduplication of the alpha vectors.
            - 2: 1+ Check of absolute domination (check if dominated at each state).
            - 3: 2+ Solves Linear Programming problem for each alpha vector to see if it is dominated by combinations of other vectors.
        
        Note that the higher the level, the heavier the time impact will be.

                Parameters:
                        level (int): Between 0 and 3, how thorough the alpha vector pruning should be.
                
                Returns:
                        new_value_function (ValueFunction): A new value function with a pruned set of alpha vectors.
        '''
        # GPU support check
        xp = cp if (gpu_support and self.is_on_gpu) else np

        if level < 1:
            return ValueFunction(self.model, xp.copy(self))
        
        # Level 1 pruning: Check for duplicates - works equally for cupy array (on gpu)
        L = {alpha_vector.values.tobytes(): alpha_vector for alpha_vector in self.alpha_vector_list}
        pruned_alpha_set = ValueFunction(self.model, list(L.values()))

        # Level 2 pruning: Check for absolute domination
        if level >= 2:
            # Beyond this point, gpu can't be used due to the functions used so if on gpu, converting it back to cpu
            if pruned_alpha_set.is_on_gpu:
                pruned_alpha_set = pruned_alpha_set.to_cpu()

            alpha_vector_array = pruned_alpha_set.alpha_vector_array
            X = cdist(alpha_vector_array, alpha_vector_array, metric=(lambda a,b:(a <= b).all() and not (a == b).all())).astype(bool)
            non_dominated_vector_indices = np.invert(X).all(axis=1)

            non_dominated_vectors = alpha_vector_array[non_dominated_vector_indices]
            non_dominated_actions = np.array(pruned_alpha_set.actions)[non_dominated_vector_indices].tolist()

            pruned_alpha_set = ValueFunction(self.model, non_dominated_vectors, non_dominated_actions)

        # Level 3 pruning: LP to check for more complex domination
        if level >= 3:
            alpha_set = pruned_alpha_set.alpha_vector_list
            pruned_alpha_set = ValueFunction(self.model)

            for i, alpha_vect in enumerate(alpha_set):
                other_alphas = alpha_set[:i] + alpha_set[(i+1):]

                # Objective function
                c = np.concatenate([np.array([1]), -1*alpha_vect])

                # Alpha vector contraints
                other_count = len(other_alphas)
                A = np.c_[np.ones(other_count), np.multiply(np.array(other_alphas), -1)]
                alpha_constraints = LinearConstraint(A, 0, np.inf)

                # Constraints that sum of beliefs is 1
                belief_constraint = LinearConstraint(np.array([0] + ([1]*self.model.state_count)), 1, 1)

                # Solve problem
                res = milp(c=c, constraints=[alpha_constraints, belief_constraint])

                # Check if dominated
                is_dominated = (res.x[0] - np.dot(res.x[1:], alpha_vect)) >= 0
                if is_dominated:
                    print(alpha_vect)
                    print(' -> Dominated\n')
                else:
                    pruned_alpha_set.append(alpha_vect)
        
        # If initial value function was on gpu, and intermediate array was converted to cpu, convert it back to gpu
        if self.is_on_gpu and not pruned_alpha_set.is_on_gpu:
            pruned_alpha_set = pruned_alpha_set.to_cpu()

        return pruned_alpha_set
    

    def save(self, path:str='./ValueFunctions', file_name:Union[str,None]=None) -> None:
        '''
        Function to save the save function in a file at a given path. If no path is provided, it will be saved in a subfolder (ValueFunctions) inside the current working directory.
        If no file_name is provided, it be saved as '<current_timestamp>_value_function.csv'.

                Parameters:
                        path (str): The path at which the csv will be saved.
                        file_name (str): The file name used to save in.
        '''
        if not os.path.exists(path):
            print('Folder does not exist yet, creating it...')
            os.makedirs(path)
            
        if file_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = timestamp + '_value_function.csv'

        vector_array = self.alpha_vector_array
        actions = self.actions

        # Convert arrays to numpy if on gpu
        if self.is_on_gpu:
            vector_array = cp.asnumpy(vector_array)
            actions = cp.asnumpy(actions)

        data = np.concatenate((np.array(self.actions)[:,None], self.alpha_vector_array), axis=1)
        columns = ['action', *self.model.state_labels]

        df = pd.DataFrame(data)
        df.to_csv(path + '/' + file_name, index=False, header=columns)


    @classmethod
    def load_from_file(cls, file:str, model:Model) -> Self:
        '''
        Function to load the value function from a csv file.

                Parameters:
                        file (str): The path and file_name of the value function to be loaded.
                        model (mdp.Model): The model the value function is linked to.
                
                Returns:
                        loaded_value_function (ValueFunction): The loaded value function.
        '''
        df = pd.read_csv(file, header=0, index_col=False)
        alpha_vectors = df.to_numpy()

        return ValueFunction(model, alpha_vectors[:,1:], alpha_vectors[:,0].astype(int))


    def plot(self,
             as_grid:bool=False,
             size:int=5,
             belief_set=None
             ) -> None:
        '''
        Function to plot out the value function in 2 or 3 dimensions.

                Parameters:
                        size (int): Default:5, The actual plot scale.
                        belief_set (list[Belief]): Optional, a set of belief to plot the belief points that were explored.
        '''
        assert len(self) > 0, "Value function is empty, plotting is impossible..."
        
        # If on GPU, convert to CPU and plot that one
        if self.is_on_gpu:
            print('[Warning] Value function on GPU, converting to numpy before plotting...')
            cpu_value_function = self.to_cpu()
            cpu_value_function.plot(as_grid, size, belief_set)
            return

        func = None
        if as_grid:
            func = self._plot_grid
        elif self.model.state_count == 2:
            func = self._plot_2D
        elif self.model.state_count == 3:
            func = self._plot_3D
        else:
            print('[Warning] \'as_grid\' parameter set to False but state count is >3 so it will be plotted as a grid')
            func = self._plot_grid

        func(size, belief_set)


    def _plot_2D(self, size, belief_set=None):
        x = np.linspace(0, 1, 100)

        plt.figure(figsize=(int(size*1.5),size))
        grid_spec = {'height_ratios': ([1] if belief_set is None else [19,1])}
        _, ax = plt.subplots((2 if belief_set is not None else 1),1,sharex=True,gridspec_kw=grid_spec)

        # Vector plotting
        alpha_vects = self.alpha_vector_array

        m = alpha_vects[:,1] - alpha_vects[:,0] # type: ignore
        m = m.reshape(m.shape[0],1)

        x = x.reshape((1,x.shape[0])).repeat(m.shape[0],axis=0)
        y = (m*x) + alpha_vects[:,0].reshape(m.shape[0],1)

        ax1 = ax[0] if belief_set is not None else ax
        for i, alpha in enumerate(self.alpha_vector_list):
            ax1.plot(x[i,:], y[i,:], color=COLOR_LIST[alpha.action]['id']) # type: ignore

        # X-axis setting
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]

        ax1.set_xticks(ticks, x_ticks) # type: ignore

        # Action legend
        proxy = [patches.Rectangle((0,0),1,1,fc = COLOR_LIST[a]['id']) for a in self.model.actions]
        ax1.legend(proxy, self.model.action_labels) # type: ignore

        # Belief plotting
        if belief_set is not None:
            beliefs_x = belief_set.belief_array[:,1]
            ax[1].scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c='red')
            ax[1].get_yaxis().set_visible(False)
            ax[1].axhline(0, color='black')

        plt.show()


    def _plot_3D(self, size, belief_set=None):

        def get_alpha_vect_z(xx, yy, alpha_vect):
            x0, y0, z0 = [0, 0, alpha_vect[0]]
            x1, y1, z1 = [1, 0, alpha_vect[1]]
            x2, y2, z2 = [0, 1, alpha_vect[2]]

            ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
            vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

            u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

            point  = np.array([0, 0, alpha_vect[0]])
            normal = np.array(u_cross_v)

            d = -point.dot(normal)

            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            
            return z

        def get_plane_gradient(alpha_vect):
        
            x0, y0, z0 = [0, 0, alpha_vect[0]]
            x1, y1, z1 = [1, 0, alpha_vect[1]]
            x2, y2, z2 = [0, 1, alpha_vect[2]]

            ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
            vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

            u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
            
            normal_vector = np.array(u_cross_v)
            normal_vector_norm = float(np.linalg.norm(normal_vector))
            normal_vector = np.divide(normal_vector, normal_vector_norm)
            normal_vector[2] = 0
            
            return np.linalg.norm(normal_vector)

        # Actual plotting
        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1, 1000)

        xx, yy = np.meshgrid(x, y)

        max_z = np.zeros((xx.shape[0], yy.shape[0]))
        best_a = (np.zeros((xx.shape[0], yy.shape[0])))
        plane = (np.zeros((xx.shape[0], yy.shape[0])))
        gradients = (np.zeros((xx.shape[0], yy.shape[0])))

        for alpha in self.alpha_vector_list:

            z = get_alpha_vect_z(xx, yy, alpha)

            # Action array update
            new_a_mask = np.argmax(np.array([max_z, z]), axis=0)

            best_a[new_a_mask == 1] = alpha.action
            
            plane[new_a_mask == 1] = random.randrange(100)
            
            alpha_gradient = get_plane_gradient(alpha)
            gradients[new_a_mask == 1] = alpha_gradient

            # Max z update
            max_z = np.max(np.array([max_z, z]), axis=0)
            
        for x_i, x_val in enumerate(x):
            for y_i, y_val in enumerate(y):
                if (x_val+y_val) > 1:
                    max_z[x_i, y_i] = np.nan
                    plane[x_i, y_i] = np.nan
                    gradients[x_i, y_i] = np.nan
                    best_a[x_i, y_i] = np.nan

        belief_points = None
        if belief_set is not None:
            belief_points = np.array(belief_set)[:,1:]
                    
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(size*4,size*3.5), sharex=True, sharey=True)

        # Set ticks
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]
        
        y_ticks = [str(t) for t in ticks]
        y_ticks[0] = self.model.state_labels[0]
        y_ticks[-1] = self.model.state_labels[2]

        plt.setp([ax1,ax2,ax3,ax4], xticks=ticks, xticklabels=x_ticks, yticks=ticks, yticklabels=y_ticks)

        # Value function ax
        ax1.set_title("Value function")
        ax1_plot = ax1.contourf(x, y, max_z, 100, cmap="viridis")
        plt.colorbar(ax1_plot, ax=ax1)

        # Alpha planes ax
        ax2.set_title("Alpha planes")
        ax2_plot = ax2.contourf(x, y, plane, 100, cmap="viridis")
        plt.colorbar(ax2_plot, ax=ax2)
        
        # Gradient of planes ax
        ax3.set_title("Gradients of planes")
        ax3_plot = ax3.contourf(x, y, gradients, 100, cmap="Blues")
        plt.colorbar(ax3_plot, ax=ax3)

        # Action policy ax
        ax4.set_title("Action policy")
        ax4.contourf(x, y, best_a, 1, colors=[c['id'] for c in COLOR_LIST])
        proxy = [patches.Rectangle((0,0),1,1,fc = COLOR_LIST[int(a)]['id']) for a in self.model.actions]
        ax4.legend(proxy, self.model.action_labels)

        if belief_points is not None:
            for ax in [ax1,ax2,ax3,ax4]:
                ax.scatter(belief_points[:,0], belief_points[:,1], s=1, c='black')

        plt.show()


    def _plot_grid(self, size=5, belief_set=None):
        value_table = np.max(self.alpha_vector_array, axis=0)[self.model.state_grid]
        best_action_table = np.array(self.actions)[np.argmax(self.alpha_vector_array, axis=0)][self.model.state_grid]
        best_action_colors = COLOR_ARRAY[best_action_table]

        dimensions = self.model.state_grid.shape

        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(size*2, size), width_ratios=(0.55,0.45))

        ax1.set_title('Value function')
        ax1_plot = ax1.imshow(value_table)
        plt.colorbar(ax1_plot, ax=ax1)
        ax1.set_xticks([i for i in range(dimensions[1])])
        ax1.set_yticks([i for i in range(dimensions[0])])

        ax2.set_title('Action policy')
        ax2.imshow(best_action_colors)
        p = [ patches.Patch(color=COLOR_LIST[int(i)]['id'], label=str(self.model.action_labels[int(i)])) for i in self.model.actions]
        ax2.legend(handles=p, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax2.set_xticks([i for i in range(dimensions[1])])
        ax2.set_yticks([i for i in range(dimensions[0])])

        plt.show()


class SolverHistory:
    '''
    Class to represent the solving history of a solver.
    The purpose of this class is to allow plotting of the solution and plotting the evolution of the value function over the training process.
    This class is not meant to be instanciated manually, it meant to be used when returned by the solve() method of a Solver object.

    ...

    Parameters
    ----------
    tracking_level: int
        The tracking level of the solver.
    model: mdp.Model
        The model that has been solved by the Solver.
    gamma: float
        The gamma parameter used by the solver (learning rate).
    eps: float
        The epsilon parameter used by the solver (covergence bound).
    initial_value_function: ValueFunction
        The initial value function the solver will use to start the solving process.
    
    # TODO: Add list of attributes
    Methods
    -------
    add(iteration_time, value_function_change, value_function:ValueFunction):
        Function to add an iteration to the solving process with the various information that will be recorded.
    plot_changes():
        Function to plot the change between value functions over the solving process.
    '''
    def __init__(self,
                 tracking_level:int,
                 model:Model,
                 gamma:float,
                 eps:float,
                 initial_value_function:Union[ValueFunction,None]=None
                 ):
        self.tracking_level = tracking_level
        self.model = model
        self.gamma = gamma
        self.eps = eps
        self.run_ts = datetime.now()

        # Tracking metrics
        self.iteration_times = []
        self.value_function_changes = []

        self.value_functions = []
        if self.tracking_level >= 2:
            self.value_functions.append(initial_value_function)


    @property
    def solution(self) -> ValueFunction:
        '''
        The last value function of the solving process.
        '''
        assert self.tracking_level >= 2, "Tracking level is set too low, increase it to 2 if you want to have value function tracking as well."
        return self.value_functions[-1]
    

    def add(self,
            iteration_time:float,
            value_function_change:float,
            value_function:ValueFunction
            ) -> None:
        '''
        Function to add a step in the simulation history.

                Parameters:
                        iteration_time (float): The time it took to run the iteration.
                        value_function_change (float): The change between the value function of this iteration and of the previous iteration.
                        value_function (ValueFunction): The value function resulting after a step of the solving process.
        '''
        if self.tracking_level >= 1:
            self.iteration_times.append(float(iteration_time))
            self.value_function_changes.append(float(value_function_change))

        if self.tracking_level >= 2:
            self.value_functions.append(value_function)
    

    @property
    def summary(self) -> str:
        '''
        A summary as a string of the information recorded.
        '''
        summary_str =  f'Summary of Value Iteration run'
        summary_str += f'\n  - Model: {self.model.state_count}-state, {self.model.action_count}-action'
        summary_str += f'\n  - Converged in {len(self.iteration_times)} iterations and {sum(self.iteration_times):.4f} seconds'
        
        if self.tracking_level >= 1:
            summary_str += f'\n  - Took on average {sum(self.iteration_times) / len(self.iteration_times):.4f}s per iteration'
        
        return summary_str
    

    def plot_changes(self) -> None:
        '''
        Function to plot the value function changes over the solving process.
        '''
        assert self.tracking_level >= 1, "To plot the change of the value function over time, use tracking level 1 or higher."
        plt.plot(np.arange(len(self.value_function_changes)), self.value_function_changes)
        plt.show()


class Solver:
    '''
    MDP Model Solver - Abstract class.
    '''
    def __init__(self) -> None:
        raise Exception("Not an implementable class, please use a subclass...")
    
    def solve(self, model: Model) -> tuple[ValueFunction, SolverHistory]:
        raise Exception("Method has to be implemented by subclass...")


class VI_Solver(Solver):
    '''
    Solver for MDP Models. This solver implements Value Iteration.
    It works by iteratively updating the value function that maps states to actions.
    
    ...

    Parameters
    ----------
    horizon: int
        Controls for how many epochs the learning can run for (works as an infinite loop safety).
    gamma: float
        Controls the learning rate, how fast the rewards are discounted at each epoch.
    eps: float
        Controls the threshold to determine whether the value functions has settled. If the max change of value for a state is lower than eps, then it has converged.

    Methods
    -------
    solve(model: mdp.Model, initial_value_function(ValueFunction), use_gpu(bool), history_tracking_level(int), print_progress(bool)):
        The method to run the solving step that returns a value function.
    '''
    def __init__(self, horizon:int=10000, gamma:float=0.99, eps:float=0.001):
        self.horizon = horizon
        self.gamma = gamma
        self.eps = eps


    def solve(self, 
              model: Model,
              initial_value_function:Union[ValueFunction,None]=None,
              use_gpu:bool=False,
              history_tracking_level:int=1,
              print_progress:bool=True
              ) -> tuple[ValueFunction, SolverHistory]:
        '''
        Function to solve an MDP model using Value Iteration.
        If an initial value function is not provided, the value function will be initiated with the expected rewards.

                Parameters:
                        model (mdp.Model): The model on which to run value iteration.
                        initial_value_function (ValueFunction): An optional initial value function to kick-start the value iteration process. (Optional)
                        use_gpu (bool): Whether to use the GPU with cupy array to accelerate solving. (Default: False)
                        history_tracking_level (int): How thorough the tracking of the solving process should be. (0: Nothing; 1: Times and sizes of belief sets and value function; 2: The actual value functions and beliefs sets) (Default: 1)
                        print_progress (bool): Whether or not to print out the progress of the value iteration process. (Default: True)

                Returns:
                        value_function (ValueFunction): The resulting value function solution to the model.
                        history (SolverHistory): The tracking of the solution over time.
        '''
        # numpy or cupy module
        xp = np

        # If GPU usage
        if use_gpu:
            assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."
            model = model.gpu_model

            # Replace numpy module by cupy for computations
            xp = cp

        # Value function initialization
        if initial_value_function is None:
            V = ValueFunction(model, model.expected_rewards_table.T, model.actions)
        else:
            V = initial_value_function.to_gpu() if use_gpu else initial_value_function
        V_opt = xp.max(V.alpha_vector_array, axis=0)

        # History tracking setup
        solve_history = SolverHistory(tracking_level=history_tracking_level,
                                      model=model,
                                      gamma=self.gamma,
                                      eps=self.eps,
                                      initial_value_function=V)

        # Computing max allowed change from epsilon and gamma parameters
        max_allowed_change = self.eps * (self.gamma / (1-self.gamma))

        for _ in trange(self.horizon) if print_progress else range(self.horizon):
            old_V_opt = V_opt
            
            start = datetime.now()

            # Computing the new alpha vectors
            alpha_vectors = model.expected_rewards_table.T + (self.gamma * xp.einsum('sar,sar->as', model.reachable_probabilities, V_opt[model.reachable_states]))
            V = ValueFunction(model, alpha_vectors, model.actions)

            V_opt = xp.max(V.alpha_vector_array, axis=0)
            
            # Change computation
            max_change = xp.max(xp.abs(V_opt - old_V_opt))

            # Tracking the history
            iteration_time = (datetime.now() - start).total_seconds()
            solve_history.add(iteration_time=iteration_time,
                              value_function_change=max_change,
                              value_function=V)

            # Convergence check
            if max_change < max_allowed_change:
                break

        return V, solve_history


class RewardSet(list):
    '''
    Class to represent a list of rewards with some functionality to plot them. Plotting options:
        - Totals: to plot a graph of the accumulated rewards over time.
        - Moving average: to plot the moving average of the rewards received over time.
        - Histogram: to plot a histogram of the various rewards received.

    ...

    Parameters
    ----------
    items: list
        The rewards in the set.

    Methods
    -------
    plot(type:str, size:int=5, max_reward=None, compare_with:Union[Self, list[Self]]=[], graph_names:list[str]=[]):
        Function to summarize the rewards with a plot of one of ('Total', 'Moving Average' or 'Histogram')
    '''
    def __init__(self, items:list=[]):
        self.extend(items)


    def plot(self, type:str='total', size:int=5, max_reward=None, compare_with:Union[Self, list[Self]]=[], graph_names:list[str]=[]) -> None:
        '''
        The method to plot summaries of the rewards received over time.
        The plots available:
            - Total ('total' or 't'): to plot the total reward as a cummulative sum over time.
            - Moving average ('moving_average' or 'ma'): to plot the moving average of the rewards
            - Hisotgram ('histogram' or 'h'): to plot the various reward in bins to plot a histogram of what was received

                Parameters:
                        type (str): The type of plot to generate.
                        size (int): The plot scale. (Default: 5)
                        max_reward: An upper bound to rewards that can be received at each timestep. (Optional)
                        compare_with (RewardSet | list[RewardSet]): One or more RewardSets to plot onlonside this one for comparison. (Optional)
                        graph_name (list[str]): A list of the names of the comparison graphs.
        '''
        plt.figure(figsize=(size*2,size))

        # Histories
        reward_sets = [self]
        if isinstance(compare_with, RewardSet):
            reward_sets.append(compare_with)
        else:
            reward_sets += compare_with
        
        assert len(reward_sets) < len(COLOR_LIST), "Not enough colors to plot all the comparisson graphs"

        # Names
        names = []
        if len(graph_names) == 0:
            names.append('Main graph')
            for i in range(1, len(reward_sets)):
                names.append(f'Comparisson {i}')
        else:
            assert len(graph_names) == len(reward_sets), "Names for the graphs are provided but not enough"
            names = copy.deepcopy(graph_names)

        # Actual plot
        if type in ['total', 't']:
            plt.title('Cummulative reward received of time')
            self._plot_total(reward_sets, names, max_reward)

        elif type in ['moving_average', 'ma']:
            plt.title('Average rewards received of time')
            self._plot_moving_average(reward_sets, names, max_reward)

        elif type in ['histogram', 'h']:
            plt.title('Histogram of rewards received')
            self._plot_histogram(reward_sets, names, max_reward)

        # Finalization
        plt.legend(loc='upper left')
        plt.show()


    def _plot_total(self, reward_sets, names, max_reward=None):
        x = np.arange(len(reward_sets[0]))

        # If given plot upper bound
        if max_reward is not None:
            y_best = max_reward * x
            plt.plot(x, y_best, color='red', linestyle='--', label='Max rewards')

        # Plot rewards
        for i, (rh, name) in enumerate(zip(reward_sets, names)):
            cum_rewards = np.cumsum([r for r in rh])
            plt.plot(x, cum_rewards, label=name, c=COLOR_LIST[i]['id'])
    

    def _plot_moving_average(self, reward_sets, names, max_reward=None):
        x = np.arange(len(reward_sets[0]))

        # If given plot upper bound
        if max_reward is not None:
            y_best = np.ones(len(reward_sets[0])) * max_reward
            plt.plot(x, y_best, color='red', linestyle='--', label='Max rewards')

        # Plot rewards
        for i, (rh, name) in enumerate(zip(reward_sets, names)):
            moving_avg = np.divide(np.cumsum([r for r in rh]), (x+1))
            plt.plot(x, moving_avg, label=name, c=COLOR_LIST[i]['id'])


    def _plot_histogram(self, reward_sets, names, max_rewards=None):
        max_unique = -np.inf
        for rh in reward_sets:
            unique_count = np.unique([r for r in rh]).shape[0]
            if max_unique < unique_count:
                max_unique = unique_count

        bin_count = int(max_unique) if max_unique < 10 else 10

        # Plot rewards
        for i, (rh, name) in enumerate(zip(reward_sets, names)):
            plt.hist([r for r in rh], bin_count, label=name, color=COLOR_LIST[i]['id'])


class SimulationHistory:
    '''
    Class to represent a list of the steps that happened during a Simulation with:
        - the state the agent passes by ('state')
        - the action the agent takes ('action')
        - the state the agent lands in ('next_state)
        - the reward received ('reward')

    ...

    Parameters
    ----------
    model: mdp.Model
        The model on which the simulation happened on.
    start_state: int
        The initial state in the simulation.

    Methods
    -------
    plot_simulation_steps(size:int=5):
        Function to plot the final state of the simulation will all states the agent passed through.
    save_simulation_video(custom_name:Union[str,None]=None, fps:int=1):
        Function to save a video of the simulation history with all the states it passes through.
    add(action:int, reward, next_state:int):
        Function to add a step in the simulation history.
    '''

    def __init__(self, model:Model, start_state:int):
        self.model = model

        self.states = [start_state]
        self.grid_point_sequence = [[i[0] for i in np.where(self.model.state_grid == start_state)]]
        self.actions = []
        self.rewards = RewardSet()


    def add(self, action:int, reward, next_state:int) -> None:
        '''
        Function to add a step in the simulation history

                Parameters:
                        action (int): The action that was taken by the agent.
                        reward: The reward received by the agent after having taken action.
                        next_state: The state that was reached by the agent after having taken action.
        '''
        self.actions.append(action)
        self.rewards.append(reward)
        self.states.append(next_state)
        self.grid_point_sequence.append([i[0] for i in np.where(self.model.state_grid == next_state)])
    

    def plot_simulation_steps(self, size:int=5):
        '''
        Plotting the path that was taken during the simulation.

                Parameters:
                        size (int): The scale of the plot.
        '''
        plt.figure(figsize=(size,size))

        # Ticks
        dimensions = self.model.state_grid.shape
        plt.xticks([i for i in range(dimensions[1])])
        plt.yticks([i for i in range(dimensions[0])])

        ax = plt.gca()
        ax.invert_yaxis()

        # Actual plotting
        data = np.array(self.grid_point_sequence)
        plt.plot(data[:,1], data[:,0], color='red')
        plt.scatter(data[:,1], data[:,0], color='red')
        plt.show()


    def _plot_to_frame_on_ax(self, frame_i, ax):
        # Data
        data = np.array(self.grid_point_sequence)[:(frame_i+1),:]

        # Ticks
        dimensions = self.model.state_grid.shape
        x_ticks = [i for i in range(dimensions[1])]
        y_ticks = [i for i in range(dimensions[0])]

        # Plotting
        ax.clear()
        ax.set_title(f'Simulation (Frame {frame_i})')

        ax.plot(data[:,1], data[:,0], color='red')
        ax.scatter(data[:,1], data[:,0], color='red')

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.invert_yaxis()


    def save_simulation_video(self, custom_name:Union[str,None]=None, fps:int=1) -> None:
        '''
        Function to save a video of the simulation history with all the states it passes through.

                Parameters:
                        custom_name (str): Optional, the name of the file it will be saved under. By default it is will be a combination of the state count, the action count and the run timestamp.
                        fps (int): The amount of steps per second appearing in the video.
        '''
        fig = plt.figure()
        ax = plt.gca()
        steps = len(self.states)

        ani = animation.FuncAnimation(fig, (lambda frame_i: self._plot_to_frame_on_ax(frame_i, ax)), frames=steps, interval=500, repeat=False)
        
        # File Title
        solved_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        video_title = f'{custom_name}-' if custom_name is not None else '' # Base
        video_title += f's{self.model.state_count}-a{self.model.action_count}-' # Model params
        video_title += f'{solved_time}.mp4'

        # Video saving
        if not os.path.exists('./Sim Videos'):
            print('Folder does not exist yet, creating it...')
            os.makedirs('./Sim Videos')

        writervideo = animation.FFMpegWriter(fps=fps)
        ani.save('./Sim Videos/' + video_title, writer=writervideo)
        print(f'Video saved at \'Sim Videos/{video_title}\'...')
        plt.close()


class Simulation:
    '''
    Class to reprensent a simulation process for a POMDP model.
    An initial random state is given and action can be applied to the model that impact the actual state of the agent along with returning a reward and an observation.

    Can be overwritten to be fit simulation needs of particular problems.

    ...
    Parameters
    ----------
    model: pomdp.Model
        The POMDP model the simulation will be applied on.

    Methods
    -------
    initialize_simulation():
        The function to initialize the simulation with a random state for the agent.
    run_action(a:int):
        Runs the action a on the current state of the agent.
    '''
    def __init__(self, model:Model) -> None:
        self.model = model
        self.initialize_simulation()


    def initialize_simulation(self, start_state:int=-1) -> int:
        '''
        Function to initialize the simulation by setting a random start state (according to the start probabilities) to the agent.

                Parameters:
                        start_state (int): The state the agent should start in. (Default: randomly over model's start probabilities)

                Returns:
                        state (int): The state the agent will start in.
        '''
        if start_state < 0:
            self.agent_state = int(np.random.choice(a=self.model.states, size=1, p=self.model.start_probabilities)[0])
        else:
            self.agent_state = start_state
        
        self.is_done = False
        return self.agent_state

    
    def run_action(self, a:int) -> Tuple[Union[int, float], int]:
        '''
        Run one step of simulation with action a.

                Parameters:
                        a (int): the action to take in the simulation.

                Returns:
                        r (int, float): the reward given when doing action a in state s and landing in state s_p. (s and s_p are hidden from agent)
        '''
        assert not self.is_done, "Action run when simulation is done."

        s = self.agent_state
        s_p = self.model.transition(s,a)
        r = self.model.reward(s,a,s_p)

        # Update agent state
        self.agent_state = s_p

        # State Done check
        if s_p in self.model.end_states:
            self.is_done = True

        # Action Done check
        if a in self.model.end_actions:
            self.is_done = True

        return r, s_p


class Agent:
    '''
    The class of an Agent running on a MDP model.
    It has the ability to train using a given mdp solver.
    Then, once trained, it can simulate actions with a given Simulation,
    either for a given amount of steps or until a single reward is received.

    ...

    Parameters
    ----------
    model: mdp.Model
        The model in which the agent can run
    
    Methods
    -------
    train(solver: Solver):
        Runs the solver on the agent's model in order to retrieve a value function.
    get_best_action(state:int):
        Retrieves the best action from the value function given a state.
    simulate(simulator:Simulation, max_steps:int=1000):
        Simulate the process on the Agent's model with the given simulator for up to max_steps iterations.
    run_n_simulations(simulator:Simulation, n:int):
        Runs n times the simulate() function.
    '''
    def __init__(self, model:Model) -> None:
        super().__init__()

        self.model = model
        self.value_function = None


    def train(self, solver:Solver) -> None:
        '''
        Method to train the agent using a given solver.
        The solver will provide a value function that will map states to actions.

                Parameters:
                        solver (Solver): The solver to run.
        '''
        self.value_function, solve_history = solver.solve(self.model)


    def get_best_action(self, state:int) -> int:
        '''
        Function to retrieve the best action for a given state based on the value function retrieved from the training.

                Parameters:
                        state (int): The state to get the best action with.
                
                Returns:
                        best_action (int): The best action found.
        '''
        assert self.value_function is not None, "No value function, training probably has to be run..."

        best_vector = np.argmax(self.value_function.alpha_vector_array[:,state])
        best_action = self.value_function.actions[best_vector]

        return best_action


    def simulate(self,
                 simulator:Union[Simulation,None]=None,
                 max_steps:int=1000,
                 start_state:int=-1,
                 print_progress:bool=True,
                 print_stats:bool=True
                 ) -> SimulationHistory:
        '''
        Function to run a simulation with the current agent for up to 'max_steps' amount of steps using a Simulation simulator.

                Parameters:
                        simulator (mdp.Simulation): The simulation that will be used by the agent. If not provided, the default MDP simulator will be used. (Optional)
                        max_steps (int): The max amount of steps the simulation can run for. (Default: 1000)
                        start_state (int): The state the agent should start in, if not provided, will be set at random based on start probabilities of the model (Default: random)
                        print_progress (bool): Whether or not to print out the progress of the simulation. (Default: True)
                        print_stats (bool): Whether or not to print simulation statistics at the end of the simulation (Default: True)

                Returns:
                        history (SimulationHistory): A step by step history of the simulation with additional functionality to plot rewards for example.
        '''
        if simulator is None:
            simulator = Simulation(self.model)

        s = simulator.initialize_simulation(start_state=start_state)

        history = SimulationHistory(self.model, s)

        sim_start_ts = datetime.now()

        # Simulation loop
        for _ in (trange(max_steps) if print_progress else range(max_steps)):
            # Play best action
            a = self.get_best_action(s)
            r, s_p = simulator.run_action(a)

            # Track progress
            history.add(action=a, next_state=s_p, reward=r)

            # Update current state
            s = s_p

            # If simulation is considered done, the rewards are simply returned
            if simulator.is_done:
                break

        if print_stats:
            sim_end_ts = datetime.now()
            print('Simulation done:')
            print(f'\t- Runtime (s): {(sim_end_ts - sim_start_ts).total_seconds()}')
            print(f'\t- Steps: {len(history.states)}')
            print(f'\t- Total rewards: {sum(history.rewards)}')
            print(f'\t- End state: {self.model.state_labels[history.states[-1]]}')
            
        return history


    def run_n_simulations(self,
                          simulator:Union[Simulation,None]=None,
                          n:int=1000,
                          max_steps:int=1000,
                          start_state:int=-1,
                          print_progress:bool=True,
                          print_stats:bool=True
                          ) -> RewardSet:
        '''
        Function to run a set of simulations in a row.
        This is useful when the simulation has a 'done' condition.
        In this case, the rewards of individual simulations are summed together under a single number.

        Not implemented:
            - Overal simulation stats

                Parameters:
                        simulator (mdp.Simulation): The simulation that will be used by the agent. If not provided, the default MDP simulator will be used. (Optional)
                        n (int): the amount of simulations to run. (Default: 1000)
                        max_steps (int): the max_steps to run per simulation. (Default: 1000)
                        start_state (int): The state the agent should start in, if not provided, will be set at random based on start probabilities of the model (Default: random)
                        print_progress (bool): Whether or not to print out the progress of the simulation. (Default: True)
                        print_stats (bool): Whether or not to print simulation statistics at the end of the simulation (Default: True)

                Returns:
                        all_histories (RewardSet): A list of the final rewards after each simulation.
        '''
        if simulator is None:
            simulator = Simulation(self.model)

        sim_start_ts = datetime.now()

        all_final_rewards = RewardSet()
        all_sim_length = []
        for _ in (trange(n) if print_progress else range(n)):
            sim_history = self.simulate(simulator, max_steps, start_state, False, False)
            all_final_rewards.append(sum(sim_history.rewards))
            all_sim_length.append(len(sim_history.states))

        if print_stats:
            sim_end_ts = datetime.now()
            print(f'All {n} simulations done:')
            print(f'\t- Average runtime (s): {((sim_end_ts - sim_start_ts).total_seconds() / n)}')
            print(f'\t- Average step count: {(sum(all_sim_length) / n)}')
            print(f'\t- Average total rewards: {(sum(all_final_rewards) / n)}')
        
        return all_final_rewards