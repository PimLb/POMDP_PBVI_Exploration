from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.patches import Rectangle
from scipy.optimize import milp, LinearConstraint
from tqdm import tqdm, trange
from typing import Self, Union

import copy
import datetime
import json
import numpy as np
import os
import pandas as pd
import random


COLOR_LIST = []
for item, value in colors.TABLEAU_COLORS.items(): # type: ignore
    value = value.lstrip('#')
    lv = len(value)
    rgb_value = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    COLOR_LIST.append({
        'name': item.replace('tab:',''),
        'id': item,
        'hex': value,
        'rgb': rgb_value
    })


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
    transitions_table:
        The transition matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.
    immediate_reward_table:
        The reward matrix, has to be |S| x |A| x |S|. If provided, it will be use in combination with the transition matrix to fill to expected rewards.
    probabilistic_rewards: bool
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.
    grid_states: list[list[Union[str,None]]]
        Optional, if provided, the model will be converted to a grid model. Allows for 'None' states if there is a gaps in the grid.

    Methods
    -------
    transition(s:int, a:int):
        Returns a random state given a prior state and an action.
    '''
    def __init__(self,
                 states:Union[int, list[str], list[list[str]]],
                 actions:Union[int, list],
                 transition_table=None,
                 immediate_reward_table=None,
                 probabilistic_rewards:bool=False,
                 grid_states:Union[None,list[list[Union[str,None]]]]=None
                 ):
        
        # States
        self.is_grid = False
        self.grid_dimensions = None
        self.grid_states = None
        if isinstance(states, int):
            self.state_labels = [f's_{i}' for i in range(states)]
        elif isinstance(states, list) and all(isinstance(item, list) for item in states):
            dim1 = len(states)
            dim2 = len(states[0])
            assert all(len(state_dim) == dim2 for state_dim in states), "All sublists of states must be of equal size"
            
            self.state_labels = []
            for state_dim in states:
                for state in state_dim:
                    self.state_labels.append(state)

            self.is_grid = True
            self.grid_dimensions = (dim1,dim2)
            self.grid_states = states
        else:
            self.state_labels = [item for item in states if isinstance(item, str)]

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
        if transition_table is None:
            # If no transitiong matrix given, generate random one
            random_probs = np.random.rand(self.state_count, self.action_count, self.state_count)
            # Normalization to have s_p probabilies summing to 1
            self.transition_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.transition_table = np.array(transition_table)
            assert self.transition_table.shape == (self.state_count, self.action_count, self.state_count), "transitions table doesnt have the right shape, it should be SxAxS"

        # Rewards
        if immediate_reward_table is None:
            # If no reward matrix given, generate random one
            self.immediate_reward_table = np.random.rand(self.state_count, self.action_count, self.state_count)
        else:
            self.immediate_reward_table = np.array(immediate_reward_table)
            assert self.immediate_reward_table.shape == (self.state_count, self.action_count, self.state_count), "rewards table doesnt have the right shape, it should be SxAxS"

        # Expected rewards
        self.expected_rewards_table = np.zeros((self.state_count, self.action_count))
        for s in self.states:
            for a in self.actions:
                self.expected_rewards_table[s,a] = np.dot(self.transition_table[s,a,:], self.immediate_reward_table[s,a,:])

        # Rewards are probabilistic
        self.probabilistic_rewards = probabilistic_rewards

        # Convert to grid if grid_states is provided
        if grid_states is not None:
            self.convert_to_grid(grid_states)

    
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
        reward = self.immediate_reward_table[s,a,s_p]
        if self.probabilistic_rewards:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward
        

    def convert_to_grid(self, state_grid:list[list[Union[str,None]]]) -> None:
        '''
        Function to define the grid structure of the the MDP model.

                Parameters:
                        state_grid (list[list[Union[str,None]]]): A matrix of states (as their labels), None are allowed, it will just be a gap in the grid.
        '''
        dim1 = len(state_grid)
        dim2 = len(state_grid[0])
        assert all(len(state_dim) == dim2 for state_dim in state_grid), "All sublists of states must be of equal size"

        states_convered = 0
        for dim1_states in state_grid:
            for state in dim1_states:
                if state is None:
                    continue

                if state in self.state_labels:
                    states_convered += 1
                elif not (state in self.state_labels):
                    raise Exception(f'Countains a state (\'{state}\') not in the list of states...')

        assert states_convered == self.state_count, "Some states of the state list are missing..."

        self.is_grid = True
        self.grid_dimensions = (dim1, dim2)
        self.grid_states = state_grid


    def to_dict(self) -> dict:
        '''
        Function to return a python dictionary with all the information of the model.

                Returns:
                        model_dict (dict): The representation of the model in a dictionary format.
        '''
        model_dict = {
            'states': self.state_labels,
            'actions': self.action_labels,
            'transition_table': self.transition_table.tolist(),
            'immediate_reward_table': self.immediate_reward_table.tolist(),
            'probabilistic_rewards': self.probabilistic_rewards
        }

        if self.is_grid:
            model_dict['grid_states'] = self.grid_states

        return model_dict
    

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

        model_dict = self.to_dict()
        json_object = json.dumps(model_dict, indent=4)
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


class ValueFunction(list[AlphaVector]):
    '''
    Class representing a set of AlphaVectors. One such set approximates the value function of the MDP model.

    ...

    Methods
    -------
    prune(level:int=1):
        Provides a ValueFunction where the alpha vector set is pruned with a certain level of pruning.
    plot(size:int=5, state_list:list[str], action_list:list[str], belief_set):
        Function to plot the value function for 2- or 3-state models.
    '''
    def __init__(self, model:Model, alpha_vectors:list[AlphaVector]=[]):
        self.model = model
        for vector in alpha_vectors:
            self.append(vector)


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
        
        pruned_alpha_set = ValueFunction(self.model)

        # Level 1 pruning: Check for duplicates
        if level >= 1:
            L = {array.tobytes(): array for array in self}
            pruned_alpha_set.extend(list(L.values()))
        
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
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = timestamp + '_value_function.csv'

        data = np.array([[alpha.action, *alpha] for alpha in self])
        columns = ['action', *[f'state_{i}' for i in range(len(self[0]))]]

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
        if self.model.state_count == 2:
            func = self._plot_2D
        elif self.model.state_count == 3:
            func = self._plot_3D
        elif self.model.is_grid:
            func = self._plot_grid
        else:
            raise Exception("Value function plotting only available for MDP's of 2 or 3 states or in grid configuration.")

        func(size, belief_set)


    def _plot_2D(self, size, belief_set=None):
        x = np.linspace(0, 1, 100)

        plt.figure(figsize=(int(size*1.5),size))
        grid_spec = {'height_ratios': ([1] if belief_set is None else [19,1])}
        _, ax = plt.subplots((2 if belief_set is not None else 1),1,sharex=True,gridspec_kw=grid_spec)

        # Vector plotting
        alpha_vects = np.array(self)

        m = alpha_vects[:,1] - alpha_vects[:,0] # type: ignore
        m = m.reshape(m.shape[0],1)

        x = x.reshape((1,x.shape[0])).repeat(m.shape[0],axis=0)
        y = (m*x) + alpha_vects[:,0].reshape(m.shape[0],1)

        ax1 = ax[0] if belief_set is not None else ax
        for i, alpha in enumerate(self):
            ax1.plot(x[i,:], y[i,:], color=COLOR_LIST[alpha.action]['id']) # type: ignore

        # X-axis setting
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]

        ax1.set_xticks(ticks, x_ticks) # type: ignore

        # Action legend
        proxy = [Rectangle((0,0),1,1,fc = COLOR_LIST[a]['id']) for a in self.model.actions]
        ax1.legend(proxy, self.model.action_labels) # type: ignore

        # Belief plotting
        if belief_set is not None:
            beliefs_x = np.array(belief_set)[:,1]
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

        for alpha in self:

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
        proxy = [Rectangle((0,0),1,1,fc = COLOR_LIST[a]['id']) for a in self.model.actions]
        ax4.legend(proxy, self.model.action_labels)

        if belief_points is not None:
            for ax in [ax1,ax2,ax3,ax4]:
                ax.scatter(belief_points[:,0], belief_points[:,1], s=1, c='black')

        plt.show()


    def _plot_grid(self, size=5, belief_set=None):
        assert self.model.grid_dimensions is not None, "Model is not in grid format"
        assert self.model.grid_states is not None, "Model is not in grid format"

        value_table = np.full(self.model.grid_dimensions, np.nan)
        best_action_table = np.full([*self.model.grid_dimensions,3],0)

        for x in range(value_table.shape[0]):
            for y in range(value_table.shape[1]):
                state_label = self.model.grid_states[x][y]
                if state_label in self.model.state_labels:
                    state_id = self.model.state_labels.index(state_label)

                    best_vector = np.argmax(np.array(self)[:,state_id])
                    value_table[x,y] = self[best_vector][state_id]
                    best_action_table[x,y,:] = COLOR_LIST[self[best_vector].action]['rgb']

        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(size*2, size), width_ratios=(0.55,0.45))

        ax1.set_title('Value function')
        ax1_plot = ax1.imshow(value_table)
        plt.colorbar(ax1_plot, ax=ax1)
        ax1.set_xticks([i for i in range(self.model.grid_dimensions[1])])
        ax1.set_yticks([i for i in range(self.model.grid_dimensions[0])])

        ax2.set_title('Action policy')
        ax2.imshow(best_action_table)
        p = [ patches.Patch(color=COLOR_LIST[i]['id'], label=self.model.action_labels[i]) for i in self.model.actions]
        ax2.legend(handles=p, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax2.set_xticks([i for i in range(self.model.grid_dimensions[1])])
        ax2.set_yticks([i for i in range(self.model.grid_dimensions[0])])

        plt.show()


class SolverHistory(list[dict]):
    '''
    Class to represent the solving history of a solver.
    The purpose of this class is to allow plotting of the solution and plotting the evolution of the value function over the training process.
    This class is not meant to be instanciated manually, it meant to be used when returned by the solve() method of a Solver object.

    ...

    Attributes
    ----------
    model: mdp.Model
        The model that has been solved by the Solver
    params: dict
        Additional Solver parameters used to make better visualizations
    '''
    def __init__(self, model, **params):
        self.model = model
        self.params = params
        self.run_ts = datetime.datetime.now()


    @property
    def solution(self) -> ValueFunction:
        return self[-1]['value_function']


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
            V = ValueFunction(model, [AlphaVector(model.expected_rewards_table[:,a], a) for a in model.actions])
        else:
            V = copy.deepcopy(initial_value_function)
        V_opt = V[0]

        solve_history = SolverHistory(model)
        solve_history.append({'value_function': V})

        max_allowed_change = self.eps * (self.gamma / (1-self.gamma))

        for _ in trange(self.horizon) if print_progress else range(self.horizon):
            old_V_opt = copy.deepcopy(V_opt)

            V = ValueFunction(model)
            for a in model.actions:
                alpha_vect = []
                for s in model.states:
                    summer = sum(model.transition_table[s, a, s_p] * old_V_opt[s_p] for s_p in model.states)
                    alpha_vect.append(model.expected_rewards_table[s,a] + (self.gamma * summer))

                V.append(AlphaVector(alpha_vect, a))

            V_opt = np.max(np.array(V), axis=1)

            solve_history.append({'value_function': V})
                
            max_change = np.max(np.abs(V_opt - old_V_opt))
            if max_change < max_allowed_change:
                break

        return V, solve_history


class Simulation:
    def __init__(self, model:Model, done_on_reward:bool=False, done_on_state:Union[int,list[int]]=[],done_on_action:Union[int,list[int]]=[]) -> None:
        self.model = model
        self.done_on_reward = done_on_reward
        self.done_on_state = done_on_state if isinstance(done_on_state, list) else [done_on_state]
        self.done_on_action = done_on_action if isinstance(done_on_action, list) else [done_on_action]

        self.initialize_simulation()


    def initialize_simulation(self) -> None:
        '''
        Function to initialize the simulation by setting a random start state to the agent.
        '''
        self.agent_state = random.choice(self.model.states)
        self.is_done = False

    
    def run_action(self, a:int) -> Union[int, float]:
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

        # Reward Done check
        if self.done_on_reward and (r != 0):
            self.is_done = True

        # State Done check
        if s_p in self.done_on_state:
            self.is_done = True

        # Action Done check
        if a in self.done_on_action:
            self.is_done = True

        return r
    

class Agent:
    # TODO
    pass