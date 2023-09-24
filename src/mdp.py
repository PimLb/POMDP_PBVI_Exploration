from datetime import datetime
from inspect import signature
from matplotlib import animation, colors, patches
from matplotlib import pyplot as plt
from scipy.optimize import milp, LinearConstraint
from tqdm.auto import trange
from typing import Self, Union, Tuple

import copy
import json
# import numpy as np
import cupy as np
import os
import pandas as pd
import random


COLOR_LIST = [{
    'name': item.replace('tab:',''),
    'id': item,
    'hex': value,
    'rgb': tuple(int(value.lstrip('#')[i:i + (len(value)-1) // 3], 16) for i in range(0, (len(value)-1), (len(value)-1) // 3))
    } for item, value in colors.TABLEAU_COLORS.items()] # type: ignore


def log(content:str) -> None:
    '''
    Function to print a log line with a timestamp 
    '''
    print(f'[{datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}] ' + content)


class Model:
    '''
    MDP Model class.

    ...

    Attributes
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
    grid_states: list[list]]
        Optional, if provided, the model will be converted to a grid model. Allows for 'None' states if there is a gaps in the grid.
    start_probabilities: list
        Optional, the distribution of chances to start in each state. If not provided, there will be an uniform chance for each state.
    end_states: list
        Optional, entering either state in the list during a simulation will end the simulation.
    end_action: list
        Optional, playing action of the list during a simulation will end the simulation.
        
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
                 grid_states:Union[None,list[list[Union[str,None]]]]=None,
                 start_probabilities:Union[list,None]=None,
                 end_states:list[int]=[],
                 end_actions:list[int]=[]
                 ):
        
        log('Instantiation of MDP Model:')
        
        # ------------------------- States -------------------------
        if isinstance(states, int): # State count
            self.state_labels = [f's_{i}' for i in range(states)]
            self.grid_states = [self.state_labels]

        elif isinstance(states, list) and all(isinstance(item, list) for item in states): # 2D list of states
            dim1 = len(states)
            dim2 = len(states[0])
            assert all(len(state_dim) == dim2 for state_dim in states), "All sublists of states must be of equal size"
            
            self.state_labels = []
            for state_dim in states:
                for state in state_dim:
                    self.state_labels.append(state)

            self.grid_states = states

        else: # Default: single of list of string items
            self.state_labels = [item for item in states if isinstance(item, str)]
            self.grid_states = [self.state_labels]

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
            self.max_reachable_states = self.reachable_states.shape[2]

            log(f'- At most {self.max_reachable_states} reachable states per state-action pair')

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
                print(f'[Warning] No transition matrix or function provided but reachable states are, so probability to reach any reachable states will "1 / reachable state count" so here: {1/self.max_reachable_states:.3f}.')

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

        # ------------------------- Rewards are probabilistic -------------------------
        self.probabilistic_rewards = rewards_are_probabilistic

        # Convert to grid if grid_states is provided
        if grid_states is not None:
            self.convert_to_grid(grid_states)

        # TODO: Rework this
        # self.map_states_to_grid_points()

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
            # TODO Optimize reachable state computation
            log('- Starting computation of reachable states from transition data')
            start_ts = datetime.now()

            self.reachable_states = []
            self.max_reachable_states = 0
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
                    
                    if len(reachable_list) > self.max_reachable_states:
                        self.max_reachable_states = len(reachable_list)

                self.reachable_states.append(reachable_states_for_action)

            # In case some state-action pairs lead to more states than other, we fill with the 1st non states not used
            for s in self.states:
                for a in self.actions:
                    to_add = 0
                    while len(self.reachable_states[s][a]) < self.max_reachable_states:
                        if to_add not in self.reachable_states[s][a]:
                            self.reachable_states[s][a].append(to_add)
                        to_add += 1

            # Converting to ndarray
            self.reachable_states = np.array(self.reachable_states, dtype=int)

            duration = (datetime.now() - start_ts).total_seconds()
            log(f'    > Done in {duration:.3f}s')
            log(f'- At most {self.max_reachable_states} reachable states per state-action pair')

        # ------------------------- Reachable state probabilities -------------------------
        log('- Starting computation of reachable state probabilities from transition data')
        start_ts = datetime.now()

        if self.transition_function is None and self.transition_table is None:
            self.reachable_probabilities = np.full(self.reachable_states.shape, 1/self.max_reachable_states)
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
        ri = int(np.argmax(np.random.multinomial(n=1, pvals=self.reachable_probabilities[s,a,:])))
        s_p = int(self.reachable_states[s,a,ri])
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
        if self.probabilistic_rewards:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward
        

    def convert_to_grid(self, state_grid:list[list]) -> None:
        '''
        Function to define the grid structure of the the MDP model.

                Parameters:
                        state_grid (list[list]): A matrix of states (as their labels), None are allowed, it will just be a gap in the grid.
        '''
        assert all(len(state_dim) == len(state_grid[0]) for state_dim in state_grid), "All sublists of states must be of equal size"

        states_covered = 0
        for dim1_states in state_grid:
            for state in dim1_states:
                if state is None:
                    continue

                if state in self.state_labels:
                    states_covered += 1
                elif not (state in self.state_labels):
                    raise Exception(f'Countains a state (\'{state}\') not in the list of states...')

        assert states_covered == self.state_count, "Some states of the state list are missing..."
        self.grid_states = state_grid
        self.map_states_to_grid_points()


    def map_states_to_grid_points(self) -> None:
        '''
        Function to map states to grid points.
        '''
        self.state_grid_points = []

        for state in self.state_labels:
            for y, y_state_list in enumerate(self.grid_states):
                for x, x_state in enumerate(y_state_list):
                    if x_state == state:
                        self.state_grid_points.append([x,y])


    # TODO: Fix this
    # def to_dict(self) -> dict:
    #     '''
    #     Function to return a python dictionary with all the information of the model.

    #             Returns:
    #                     model_dict (dict): The representation of the model in a dictionary format.
    #     '''
    #     model_dict = {
    #         'states': self.state_labels,
    #         'actions': self.action_labels,
    #         'transition_table': self.transition_table.tolist(),
    #         'immediate_reward_table': self.immediate_reward_table.tolist(),
    #         'probabilistic_rewards': self.probabilistic_rewards,
    #         'grid_states': self.grid_states
    #     }

    #     return model_dict
    

    # def save(self, file_name:str, path:str='./Models') -> None:
    #     '''
    #     Function to save the current model in a json file.
    #     By default, the model will be saved in 'Models' directory in the current working directory but this can be changed using the 'path' parameter.

    #             Parameters:
    #                     file_name (str): The name of the json file the model will be saved in.
    #                     path (str): The path at which the model will be saved. (Default: './Models')
    #     '''
    #     if not os.path.exists(path):
    #         print('Folder does not exist yet, creating it...')
    #         os.makedirs(path)

    #     if not file_name.endswith('.json'):
    #         file_name += '.json'

    #     model_dict = self.to_dict()
    #     json_object = json.dumps(model_dict, indent=4)
    #     with open(path + '/' + file_name, 'w') as outfile:
    #         outfile.write(json_object)


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

        loaded_model = Model(**json_model)

        if 'grid_states' in json_model:
            loaded_model.convert_to_grid(json_model['grid_states'])

        return loaded_model


class AlphaVector(np.ndarray):
    '''
    A class to represent an Alpha Vector, a vector representing a plane in |S| dimension for POMDP models.

    ...

    Attributes
    ----------
    input_array: 
        The actual vector with the value for each state.
    action: int
        The action associated with the vector.
    '''
    def __new__(cls, input_array, action:int):
        obj = np.asarray(input_array).view(cls)
        obj._action = action
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._action = getattr(obj, '_action', None)

    @property
    def action(self) -> int:
        assert self._action is not None
        return self._action


class ValueFunction:
    '''
    Class representing a set of AlphaVectors. One such set approximates the value function of the MDP model.

    ...

    Attributes
    ----------
    model: (mdp.Model)
        The model the value function is associated with.
    alpha_vectors: (list[AlphaVector] | np.ndarray) (Optional)
        The alpha vectors composing the value function, if none are provided, it will be empty to start with and AlphaVectors can be appended.
    action_list: (list[int])
        The actions associated with alpha vectors in the case the alpha vectors are provided as an numpy array.

    Methods
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

        # As numpy array
        if isinstance(alpha_vectors, np.ndarray):
            av_shape = alpha_vectors.shape
            exp_shape = (len(action_list), model.state_count)
            assert av_shape == exp_shape, f"Alpha vector array does not have the right shape (received: {av_shape}; expected: {exp_shape})"

            self._vector_array = alpha_vectors
            self._actions = action_list

        # List of alpha vectors
        else:
            self._vector_list = alpha_vectors


    @property
    def alpha_vector_list(self) -> AlphaVector:
        if self._vector_list is None:
            self._vector_list = []
            for alpha_vect, action in zip(self._vector_array, self._actions):
                self._vector_list.append(AlphaVector(alpha_vect, action))
        return self._vector_list
    

    @property
    def alpha_vector_array(self) -> np.ndarray:
        if self._vector_array is None:
            self._vector_array = np.array(self._vector_list)
            self._actions = [v.action for v in self._vector_list]
        return self._vector_array
    

    @property
    def actions(self) -> int:
        if self._actions is None:
            self._vector_array = np.array(self._vector_list)
            self._actions = [v.action for v in self._vector_list]
        return self._actions
    

    def __len__(self) -> int:
        return len(self._vector_list) if self._vector_list is not None else self._vector_array.shape[0]
    

    def append(self, alpha_vector:AlphaVector):
        '''
        Function to add an alpha vector to the value function.
        '''
        if self._vector_array is not None:
            self._vector_array = np.append(self._vector_array, alpha_vector[None,:], axis=0)
            self._actions.append(alpha_vector.action)
        
        if self._vector_list is not None:
            self._vector_list.append(alpha_vector)


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
        if level < 1:
            return copy.deepcopy(self)
        
        # Level 1 pruning: Check for duplicates
        L = {array.tobytes(): array for array in self.alpha_vector_list}
        pruned_alpha_set = ValueFunction(self.model, list(L.values()))

        # Level 2 pruning: Check for absolute domination
        if level >= 2:
            alpha_set = pruned_alpha_set
            pruned_alpha_set = ValueFunction(self.model)
            for alpha_vector in alpha_set:
                dominated = False
                for compare_alpha_vector in alpha_set:
                    if all(alpha_vector < compare_alpha_vector):
                        dominated = True
                if not dominated:
                    pruned_alpha_set.append(alpha_vector)

        # Level 3 pruning: LP to check for more complex domination
        if level >= 3:
            alpha_set = pruned_alpha_set
            pruned_alpha_set = ValueFunction(self.model)

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

        data = np.array([[alpha.action, *alpha] for alpha in self.alpha_vector_list])
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

        vector_count = alpha_vectors.shape[0]

        vector_list = []
        for i in range(vector_count):
            vector_list.append(AlphaVector(alpha_vectors[i,1:], action=int(alpha_vectors[i,0])))

        return ValueFunction(model, vector_list)
    

    def plot(self,
             as_grid:bool=False,
             size:int=5,
             belief_set=None
             ):
        '''
        Function to plot out the value function in 2 or 3 dimensions.

                Parameters:
                        size (int): Default:5, The actual plot scale.
                        belief_set (list[Belief]): Optional, a set of belief to plot the belief points that were explored.
        '''
        assert len(self) > 0, "Value function is empty, plotting is impossible..."

        func = None
        if as_grid:
            func = self._plot_grid
        elif self.model.state_count == 2:
            func = self._plot_2D
        elif self.model.state_count == 3:
            func = self._plot_3D
        else:
            print('Warning: as_grid parameter set to False but state count is >3 so it will be plotted as a grid')
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
        proxy = [patches.Rectangle((0,0),1,1,fc = COLOR_LIST[a]['id']) for a in self.model.actions]
        ax4.legend(proxy, self.model.action_labels)

        if belief_points is not None:
            for ax in [ax1,ax2,ax3,ax4]:
                ax.scatter(belief_points[:,0], belief_points[:,1], s=1, c='black')

        plt.show()


    def _plot_grid(self, size=5, belief_set=None):
        assert self.model.grid_states is not None, "Model is not in grid format"

        dimensions = (len(self.model.grid_states), len(self.model.grid_states[0]))

        value_table = np.full(dimensions, np.nan)
        best_action_table = np.full([*dimensions,3],0)

        for x in range(value_table.shape[0]):
            for y in range(value_table.shape[1]):
                state_label = self.model.grid_states[x][y]
                if state_label in self.model.state_labels:
                    state_id = self.model.state_labels.index(state_label)

                    best_vector = np.argmax(self.alpha_vector_array[:,state_id])
                    value_table[x,y] = self.alpha_vector_array[best_vector, state_id]
                    best_action_table[x,y,:] = COLOR_LIST[self.actions[best_vector]]['rgb']

        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(size*2, size), width_ratios=(0.55,0.45))

        ax1.set_title('Value function')
        ax1_plot = ax1.imshow(value_table)
        plt.colorbar(ax1_plot, ax=ax1)
        ax1.set_xticks([i for i in range(dimensions[1])])
        ax1.set_yticks([i for i in range(dimensions[0])])

        ax2.set_title('Action policy')
        ax2.imshow(best_action_table)
        p = [ patches.Patch(color=COLOR_LIST[i]['id'], label=self.model.action_labels[i]) for i in self.model.actions]
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

    Attributes
    ----------
    model: mdp.Model
        The model that has been solved by the Solver
    initial_value_function: ValueFunction
        The initial value function the solver will use to start the solving process.
    params: dict
        Additional Solver parameters used to make better visualizations
    '''
    def __init__(self,
                 model:Model,
                 initial_value_function:ValueFunction,
                 gamma:float,
                 eps:float):
        self.model = model
        self.value_functions = [initial_value_function]
        self.gamma = gamma
        self.eps = eps
        self.run_ts = datetime.now()


    @property
    def solution(self) -> ValueFunction:
        '''
        The last Value function of the solving process
        '''
        return self.value_functions[-1]
    

    def add(self, value_function:ValueFunction) -> None:
        '''
        Function to add a step in the simulation history.

                Parameters:
                        value_function (ValueFunction): The value function resulting after a step of the solving process.
        '''
        self.value_functions.append(value_function)


    def __len__(self):
        return len(self.value_functions)


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

    Attributes
    ----------
    horizon: int
        Controls for how many epochs the learning can run for (works as an infinite loop safety).
    gamma: float
        Controls the learning rate, how fast the rewards are discounted at each epoch.
    eps: float
        Controls the threshold to determine whether the value functions has settled. If the max change of value for a state is lower than eps, then it has converged.

    Methods
    -------
    solve(model: mdp.Model):
        The method to run the solving step that returns a value function.
    '''
    def __init__(self, horizon:int=10000, gamma:float=0.99, eps:float=0.001):
        self.horizon = horizon
        self.gamma = gamma
        self.eps = eps


    def solve(self, 
              model: Model,
              initial_value_function:Union[ValueFunction,None]=None,
              print_progress:bool=True
              ) -> tuple[ValueFunction, SolverHistory]:
        '''
        Function to solve an MDP model using Value Iteration.
        If an initial value function is not provided, the value function will be initiated with the expected rewards.

                Parameters:
                        model (mdp.Model): The model on which to run value iteration.
                        initial_value_function (ValueFunction): An optional initial value function to kick-start the value iteration process. (Optional)
                        print_progress (bool): Whether or not to print out the progress of the value iteration process. (Default: True)

                Returns:
                        value_function (ValueFunction): The resulting value function solution to the model.
                        history (SolverHistory): The tracking of the solution over time.
        '''
        if not initial_value_function:
            V = ValueFunction(model, model.expected_rewards_table.T, model.actions)
        else:
            V = copy.deepcopy(initial_value_function)
        V_opt = V.alpha_vector_list[0]

        solve_history = SolverHistory(model, V, self.gamma, self.eps)

        max_allowed_change = self.eps * (self.gamma / (1-self.gamma))

        for _ in trange(self.horizon) if print_progress else range(self.horizon):
            old_V_opt = V_opt
            
            # Computing the new alpha vectors
            alpha_vectors = model.expected_rewards_table + (self.gamma * np.einsum('sar,sar->sa', model.reachable_probabilities, V_opt.take(model.reachable_states)))
            V = ValueFunction(model, alpha_vectors, model.actions)

            V_opt = np.max(V.alpha_vector_array, axis=0)

            # Tracking the history
            solve_history.add(V)
                
            max_change = np.max(np.abs(V_opt - old_V_opt))
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

    Attributes
    ----------
    items: list
        The rewards in the set.

    Methods
    -------
    plot_rewards(type:str, size:int=5, max_reward=None, compare_with:Union[Self, list[Self]]=[], graph_names:list[str]=[]):
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

    Attributes
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
        self.grid_point_sequence = [self.model.state_grid_points[start_state]]
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
        self.grid_point_sequence.append(self.model.state_grid_points[next_state])
    

    def plot_simulation_steps(self, size:int=5):
        '''
        Plotting the path that was taken during the simulation.

                Parameters:
                        size (int): The scale of the plot.
        '''
        plt.figure(figsize=(size,size))

        # Ticks
        dimensions = (len(self.model.grid_states), len(self.model.grid_states[0]))
        plt.xticks([i for i in range(dimensions[1])])
        plt.yticks([i for i in range(dimensions[0])])

        ax = plt.gca()
        ax.invert_yaxis()

        # Actual plotting
        data = np.array(self.grid_point_sequence)
        plt.plot(data[:, 0], data[:, 1], color='red')
        plt.scatter(data[:, 0], data[:, 1], color='red')
        plt.show()


    def _plot_to_frame_on_ax(self, frame_i, ax):
        # Data
        data = np.array(self.grid_point_sequence)[:(frame_i+1),:]

        # Ticks
        dimensions = (len(self.model.grid_states), len(self.model.grid_states[0]))
        x_ticks = [i for i in range(dimensions[1])]
        y_ticks = [i for i in range(dimensions[0])]

        # Plotting
        ax.clear()
        ax.set_title(f'Simulation (Frame {frame_i})')

        ax.plot(data[:, 0], data[:, 1], color='red')
        ax.scatter(data[:, 0], data[:, 1], color='red')

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
    Attributes
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
            self.agent_state = int(np.argmax(np.random.multinomial(n=1, pvals=self.model.start_probabilities)))
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

    Attributes
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