from datetime import datetime
from inspect import signature
from matplotlib import animation, cm, colors, ticker
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tqdm.auto import trange
from typing import Self, Tuple, Union

import copy
import json
# import numpy as np
import cupy as np
import math
import os
import random

from src.mdp import log
from src.mdp import AlphaVector, ValueFunction
from src.mdp import Model as MDP_Model
from src.mdp import RewardSet
from src.mdp import SimulationHistory as MDP_SimulationHistory
from src.mdp import SolverHistory as MDP_SolverHistory
from src.mdp import Solver as MDP_Solver
from src.mdp import Simulation as MDP_Simulation


COLOR_LIST = [{
    'name': item.replace('tab:',''),
    'id': item,
    'hex': value,
    'rgb': tuple(int(value.lstrip('#')[i:i + (len(value)-1) // 3], 16) for i in range(0, (len(value)-1), (len(value)-1) // 3))
    } for item, value in colors.TABLEAU_COLORS.items()] # type: ignore


class Model(MDP_Model):
    '''
    POMDP Model class. Partially Observable Markov Decision Process Model.

    ...

    Attributes
    ----------
    states: int | list[str] | list[list[str]]
        A list of state labels or an amount of states to be used. Also allows to provide a matrix of states to define a grid model.
    actions: int|list
        A list of action labels or an amount of actions to be used.
    observations:
        A list of observation labels or an amount of observations to be used
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
    observation_table:
        The observation matrix, has to be |S| x |A| x |O|. If none is provided, it will be randomly generated.
    rewards_are_probabilistic: bool
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.
    grid_states: list[list[Union[str,None]]]
        Optional, if provided, the model will be converted to a grid model. Allows for 'None' states if there is a gaps in the grid.
    start_probabilities: list
        Optional, the distribution of chances to start in each state. If not provided, there will be an uniform chance for each state. It is also used to represent a belief of complete uncertainty.
    end_states: list
        Optional, entering either state in the list during a simulation will end the simulation.
    end_action: list
        Optional, playing action of the list during a simulation will end the simulation.

    Methods
    -------
    transition(s:int, a:int):
        Returns a random state given a prior state and an action.
    observe(s_p:int, a:int):
        Returns a random observation based on the posterior state and the action that was taken.
    '''
    def __init__(self,
                 states:Union[int, list[str], list[list[str]]],
                 actions:Union[int, list],
                 observations:Union[int, list],
                 transitions=None,
                 reachable_states=None,
                 rewards=None,
                 observation_table=None,
                 rewards_are_probabilistic:bool=False,
                 grid_states:Union[None,list[list[Union[str,None]]]]=None,
                 start_probabilities:Union[list,None]=None,
                 end_states:list[int]=[],
                 end_actions:list[int]=[]
                 ):
        
        super().__init__(states=states,
                         actions=actions,
                         transitions=transitions,
                         reachable_states=reachable_states,
                         rewards=-1, # Defined here lower since immediate reward table has different shape for MDP is different than for POMDP
                         rewards_are_probabilistic=rewards_are_probabilistic,
                         grid_states=grid_states,
                         start_probabilities=start_probabilities,
                         end_states=end_states,
                         end_actions=end_actions)

        log('POMDP particular parameters:')

        # ------------------------- Observations -------------------------
        if isinstance(observations, int):
            self.observation_labels = [f'o_{i}' for i in range(observations)]
        else:
            self.observation_labels = observations
        self.observation_count = len(self.observation_labels)
        self.observations = np.arange(self.observation_count)

        if observation_table is None:
            # If no observation matrix given, generate random one
            random_probs = np.random.rand(self.state_count, self.action_count, self.observation_count)
            # Normalization to have s_p probabilies summing to 1
            self.observation_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.observation_table = np.array(observation_table)
            o_shape = self.observation_table.shape
            exp_shape = (self.state_count, self.action_count, self.observation_count)
            assert o_shape == exp_shape, f"Observations table doesnt have the right shape, it should be SxAxO (expected: {exp_shape}, received: {o_shape})."

        log(f'- {self.observation_count} observations')

        # ------------------------- Reachable transitional observation probabilities -------------------------
        log('- Starting of transitional observations for reachable states table')
        start_ts = datetime.now()

        reachable_observations = self.observation_table[self.reachable_states[:,:,None,:], self.actions[None,:,None,None], self.observations[None,None,:,None]] # SAOR
        self.reachable_transitional_observation_table = np.einsum('sar,saor->saor', self.reachable_probabilities, reachable_observations)
        
        duration = (datetime.now() - start_ts).total_seconds()
        log(f'    > Done in {duration:.3f}s')

        # ------------------------- Rewards -------------------------
        self.immediate_reward_table = None
        self.immediate_reward_function = None
        
        if rewards is None:
            # If no reward matrix given, generate random one
            self.immediate_reward_table = np.random.rand(self.state_count, self.action_count, self.state_count, self.observation_count)
        elif callable(rewards):
            # Rewards is a function
            self.immediate_reward_function = rewards
            assert len(signature(rewards).parameters) == 4, "Reward function should accept 4 parameters: s, a, sn, o..."
        else:
            # Array like
            self.immediate_reward_table = np.array(rewards)
            r_shape = self.immediate_reward_table.shape
            exp_shape = (self.state_count, self.action_count, self.state_count, self.observation_count)
            assert r_shape == exp_shape, f"Rewards table doesnt have the right shape, it should be SxAxSxO (expected: {exp_shape}, received {r_shape})"
        
        # ------------------------- Expected rewards -------------------------
        log('- Starting generation of expected rewards table')
        start_ts = datetime.now()

        reachable_rewards = None
        if self.immediate_reward_table is not None:
            reachable_rewards = rewards[self.states[:,None,None,None], self.actions[None,:,None,None], self.reachable_states[:,:,:,None], self.observations[None,None,None,:]]
        else:
            def reach_reward_func(s,a,ri,o):
                s = s.astype(int)
                a = a.astype(int)
                ri = ri.astype(int)
                o = o.astype(int)
                return self.immediate_reward_function(s,a,self.reachable_states[s,a,ri],o)
            
            reachable_rewards = np.fromfunction(reach_reward_func, (*self.reachable_states.shape, self.observation_count))

        self.expected_rewards_table = np.einsum('saor,saro->sa', self.reachable_transitional_observation_table, reachable_rewards)

        duration = (datetime.now() - start_ts).total_seconds()
        log(f'    > Done in {duration:.3f}s')


    def reward(self, s:int, a:int, s_p:int, o:int) -> Union[int,float]:
        '''
        Returns the rewards of playing action a when in state s and landing in state s_p.
        If the rewards are probabilistic, it will return 0 or 1.

                Parameters:
                        s (int): The current state
                        a (int): The action taking in state s
                        s_p (int): The state landing in after taking action a in state s
                        o (int): The observation that is done after having played action a in state s and landing in s_p

                Returns:
                        reward (int, float): The reward received.
        '''
        reward = self.immediate_reward_table[s,a,s_p,o] if self.immediate_reward_table is not None else self.immediate_reward_function(s,a,s_p,o)
        if self.rewards_are_probabilistic:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward
    

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
    

# TODO: Update this
    # def to_dict(self) -> dict:
    #     '''
    #     Function to return a python dictionary with all the information of the model.

    #             Returns:
    #                     model_dict (dict): The representation of the model in a dictionary format.
    #     '''
    #     model_dict = super().to_dict()
    #     model_dict['observations'] = self.observation_labels
    #     model_dict['observation_table'] = self.observation_table.tolist()

    #     return model_dict
    
    
    # @classmethod
    # def load_from_json(cls, file:str) -> Self:
    #     '''
    #     Function to load a POMDP model from a json file. The json structure must contain the same items as in the constructor of this class.

    #             Parameters:
    #                     file (str): The file and path of the model to be loaded.
    #             Returns:
    #                     loaded_model (pomdp.Model): An instance of the loaded model.
    #     '''
    #     with open(file, 'r') as openfile:
    #         json_model = json.load(openfile)

    #     loaded_model = Model(**json_model)

    #     if 'grid_states' in json_model:
    #         loaded_model.convert_to_grid(json_model['grid_states'])

    #     return loaded_model


class Belief:
    '''
    A class representing a belief in the space of a given model. It is the belief to be in any combination of states:
    eg:
        - In a 2 state POMDP: a belief of (0.5, 0.5) represent the complete ignorance of which state we are in. Where a (1.0, 0.0) belief is the certainty to be in state 0.

    The belief update function has been implemented based on the belief update define in the paper of J. Pineau, G. Gordon, and S. Thrun, 'Point-based approximations for fast POMDP solving'

    ...

    Attributes
    ----------
    model: Model
        The model on which the belief applies on.
    state_probabilities: np.ndarray|None
        A vector of the probabilities to be in each state of the model. The sum of the probabilities must sum to 1. If not specifies it will be set as the start probabilities of the model.

    Methods
    -------
    update(a:int, o:int):
        Function to provide a new Belief object updated using an action 'a' and observation 'o'.
    random_state():
        Function to give a random state based with the belief as the probability distribution.
    plot(size:int=5):
        Function to plot a value function as a grid heatmap.
    '''
    def __init__(self, model:Model, values:Union[np.ndarray,None]=None):
        assert model is not None
        self.model = model

        if values is not None:
            assert values.shape[0] == model.state_count, "Belief must contain be of dimension |S|"
            prob_sum = np.round(np.sum(values), decimals=3)
            assert prob_sum == 1, f"States probabilities in belief must sum to 1 (found: {prob_sum})"
            self._values = values
        else:
            self._values = model.start_probabilities

        self._grid_values = None

    
    @property
    def values(self) -> np.ndarray:
        return self._values
    

    def update(self, a:int, o:int) -> Self:
        '''
        Returns a new belief based on this current belief, the most recent action (a) and the most recent observation (o).

                Parameters:
                        a (int): The most recent action
                        o (int): The most recent observation

                Returns:
                        new_belief (Belief): An updated belief

        '''
        new_state_probabilities = np.einsum('sr,sr->s', self.model.reachable_transitional_observation_table[:,a,o,:], self.values.take(self.model.reachable_states[:,a,:]))
        
        # Normalization
        prob_sum = np.sum(new_state_probabilities)
        if prob_sum != 1.0:
            new_state_probabilities /= prob_sum
        new_belief = Belief(self.model, new_state_probabilities)
        return new_belief
    

    def random_state(self) -> int:
        '''
        Returns a random state of the model weighted by the belief probabily.

                Returns:
                        rand_s (int): A random state
        '''
        rand_s = int(np.argmax(np.random.multinomial(n=1, pvals=self.values)))
        return rand_s
    

    @property
    def grid_values(self) -> np.ndarray:
        if self._grid_values is None:
            dimensions = (len(self.model.grid_states), len(self.model.grid_states[0]))

            self._grid_values = np.zeros(dimensions)

            for x in range(dimensions[0]):
                for y in range(dimensions[1]):
                    state_label = self.model.grid_states[x][y]
                    
                    if state_label is not None:
                        s = self.model.state_labels.index(state_label) # type: ignore
                        self._grid_values[x,y] = self._values[s]

        return self._grid_values
    

    def plot(self, size:int=5) -> None:
        '''
        Function to plot a heatmap of the belief distribution if the belief is of a grid model.

                Parameters:
                        size (int): The scale of the plot. (Default: 5)
        '''
        # Plot setup
        plt.figure(figsize=(size*1.2,size))

        # Ticks
        dimensions = (len(self.model.grid_states), len(self.model.grid_states[0]))
        plt.xticks([i for i in range(dimensions[1])])
        plt.yticks([i for i in range(dimensions[0])])

        # Actual plot
        plt.imshow(self.grid_values,cmap='Blues')
        plt.colorbar()
        plt.show()


class BeliefSet:
    '''
    Class to represent a set of beliefs with regard to a POMDP model.
    It has the purpose to store the beliefs in numpy array format and be able to conver it to a list of Belief class objects.
    This class also provides the option to display the beliefs when operating on a 2 or 3d space with the plot() function.
    
    ...

    Attributes
    ----------
    model: pomdp.Model
        The model on which the beliefs apply.
    beliefs: list[Belief] | np.ndarray
        The actual set of beliefs.

    Methods
    -------
    plot(size=15):
        A function to plot the beliefs in belief space when in 2- or 3-states models.
    '''
    def __init__(self, model:Model, beliefs:Union[list[Belief],np.ndarray]) -> None:
        self.model = model

        self._belief_list = None
        self._belief_array = None

        if isinstance(beliefs, np.ndarray):
            assert beliefs.shape[1] == model.state_count, f"Belief array provided doesnt have the right shape (expected (-,{model.state_count}), received {beliefs.shape})"
            self._belief_array = beliefs
        else:
            assert all(len(b.values) == model.state_count for b in beliefs), f"Beliefs in belief list provided dont all have shape ({model.state_count},)"
            self._belief_list = beliefs


    @property
    def belief_array(self) -> np.ndarray:
        if self._belief_array is None:
            self._belief_array = np.array([b.values for b in self._belief_list])
        return self._belief_array
    

    @property
    def belief_list(self) -> list[Belief]:
        if self._belief_list is None:
            self._belief_list = [Belief(self.model, belief_vector) for belief_vector in self._belief_array]
        return self._belief_list
    
    
    def plot(self, size:int=15):
        '''
        Function to plot the beliefs in the belief set.
        Note: Only works for 2-state and 3-state believes.

                Parameters:
                        size (int): The figure size and general scaling factor
        '''
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3"
        
        if self.model.state_count == 2:
            self._plot_2D(size)
        elif self.model.state_count == 3:
            self._plot_3D(size)


    def _plot_2D(self, size=15):
        beliefs_x = self.belief_array[:,1]

        plt.figure(figsize=(size, max([int(size/7),1])))
        plt.scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c=list(range(beliefs_x.shape[0])), cmap='Blues')
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)

        # X-axis setting
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]

        plt.xticks(ticks, x_ticks)
        plt.show()


    def _plot_3D(self, size=15):
        # Function to project points to a simplex triangle
        def projectSimplex(points):
            """ 
            Project probabilities on the 3-simplex to a 2D triangle
            
            N points are given as N x 3 array
            """
            # Convert points one at a time
            tripts = np.zeros((points.shape[0],2))
            for idx in range(points.shape[0]):
                # Init to triangle centroid
                x = 1.0 / 2
                y = 1.0 / (2 * np.sqrt(3))
                # Vector 1 - bisect out of lower left vertex 
                p1 = points[idx, 0]
                x = x - (1.0 / np.sqrt(3)) * p1 * np.cos(np.pi / 6)
                y = y - (1.0 / np.sqrt(3)) * p1 * np.sin(np.pi / 6)
                # Vector 2 - bisect out of lower right vertex  
                p2 = points[idx, 1]  
                x = x + (1.0 / np.sqrt(3)) * p2 * np.cos(np.pi / 6)
                y = y - (1.0 / np.sqrt(3)) * p2 * np.sin(np.pi / 6)        
                # Vector 3 - bisect out of top vertex
                p3 = points[idx, 2]
                y = y + (1.0 / np.sqrt(3) * p3)
            
                tripts[idx,:] = (x,y)

            return tripts
        
        # Plotting the simplex 
        def plotSimplex(points,
                        fig=None,
                        vertexlabels=['s_0','s_1','s_2'],
                        **kwargs):
            """
            Plot Nx3 points array on the 3-simplex 
            (with optionally labeled vertices) 
            
            kwargs will be passed along directly to matplotlib.pyplot.scatter    
            Returns Figure, caller must .show()
            """
            if(fig == None):        
                fig = plt.figure()
            # Draw the triangle
            l1 = Line2D([0, 0.5, 1.0, 0], # xcoords
                        [0, np.sqrt(3) / 2, 0, 0], # ycoords
                        color='k')
            fig.gca().add_line(l1)
            fig.gca().xaxis.set_major_locator(ticker.NullLocator())
            fig.gca().yaxis.set_major_locator(ticker.NullLocator())
            # Draw vertex labels
            fig.gca().text(-0.05, -0.05, vertexlabels[0])
            fig.gca().text(1.05, -0.05, vertexlabels[1])
            fig.gca().text(0.5, np.sqrt(3) / 2 + 0.05, vertexlabels[2])
            # Project and draw the actual points
            projected = projectSimplex(points)
            plt.scatter(projected[:,0], projected[:,1], **kwargs)              
            # Leave some buffer around the triangle for vertex labels
            fig.gca().set_xlim(-0.2, 1.2)
            fig.gca().set_ylim(-0.2, 1.2)

            return fig

        # Actual plot
        fig = plt.figure(figsize=(size,size))

        cmap = cm.get_cmap('Blues')
        norm = colors.Normalize(vmin=0, vmax=self.belief_array.shape[0])
        c = range(self.belief_array.shape[0])
        # Do scatter plot
        fig = plotSimplex(self.belief_array, fig=fig, vertexlabels=self.model.state_labels, s=size, c=c, cmap=cmap, norm=norm)

        plt.show()


class SolverHistory(MDP_SolverHistory):
    '''
    Class to represent the history of a solver for a POMDP solver.
    It has mainly the purpose to have visualizations for the solution, belief set and the whole solving history.
    The visualizations available are:
        - Belief set plot
        - Solution plot
        - Video of value function and belief set evolution over training.

    ...

    Attributes
    ----------
    model: pomdp.Model
        The model the solver has solved
    initial_value_function: ValueFunction
        The initial value function the solver will use to start the solving process.
    initial_belief_set: BeliefSet
        The initial belief set the solver will use to start the solving process.
    gamma: float
        The gamma parameter used by the solver (learning rate).
    eps: float
        The epsilon parameter used by the solver (covergence bound).
    expand_function: str
        The expand (exploration) function used by the solver.

    Methods
    -------
    plot_belief_set(size:int=15):
        Once solve() has been run, the explored beliefs can be plot for 2- and 3- state models.
    plot_solution(size:int=5, plot_belief:bool=True):
        Once solve() has been run, the value function solution can be plot for 2- and 3- state models.
    save_history_video(custom_name:Union[str,None]=None, compare_with:Union[list, ValueFunction, Self]=[]):
        Once the solve has been run, we can save a video of the history of the solving process.
    '''
    def __init__(self,
                 model:Model,
                 initial_value_function:ValueFunction,
                 initial_belief_set:BeliefSet,
                 gamma:float,
                 eps:float,
                 expand_function:str
                 ):
        super().__init__(model,
                         initial_value_function,
                         gamma,
                         eps)
        self.belief_sets = [initial_belief_set]
        self.expand_function = expand_function


    @property
    def explored_beliefs(self) -> BeliefSet:
        '''
        The final set of beliefs explored during the solving.
        '''
        return self.belief_sets[-1]
    

    def add(self, value_function:ValueFunction, belief_set:list[Belief]) -> None:
        '''
        Function to add a step in the simulation history by recording the value function and the explored belief set.

                Parameters:
                        value_function (ValueFunction): The value function resulting after a step of the solving process.
                        belief_set (list[Belief]): The belief set used for the Update step of the solving process.
        '''
        self.value_functions.append(value_function)
        self.belief_sets.append(belief_set)


    def plot_belief_set(self, size:int=15):
        '''
        Function to plot the last belief set explored during the solving process.

                Parameters:
                        size (int): The scale of the plot
        '''
        self.explored_beliefs.plot(size=size)


    def plot_solution(self, size:int=5, plot_belief:bool=True):
        '''
        Function to plot the value function of the solution.
        Note: only works for 2 and 3 states models

                Parameters:
                        size (int): The figure size and general scaling factor.
                        plot_belief (bool): Whether to plot the belief set along with the value function.
        '''
        self.solution.plot(size=size, belief_set=(self.explored_beliefs if plot_belief else None))


    def save_history_video(self,
                           custom_name:Union[str,None]=None,
                           compare_with:Union[list, ValueFunction, MDP_SolverHistory]=[],
                           graph_names:list[str]=[],
                           fps:int=10
                           ) -> None:
        '''
        Function to generate a video of the training history. Another solved solver or list of solvers can be put in the 'compare_with' parameter.
        These other solver's value function will be overlapped with the 1st value function.
        The explored beliefs of the main solver are also mapped out. (other solvers's explored beliefs will not be plotted)
        Also, a single value function or list of value functions can be sent in but they will be fixed in the video.

        Note: only works for 2-state models.

                Parameters:
                        custom_name (str): Optional, the name the video will be saved with.
                        compare_with (PBVI, ValueFunction, list): Optional, value functions or other solvers to plot against the current solver's history
                        graph_names (list[str]): Optional, names of the graphs for the legend of which graph is being plot.
                        fps (int): How many frames per second should the saved video have. (Default: 10)
        '''
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3"
        if self.model.state_count == 2:
            self._save_history_video_2D(custom_name, compare_with, copy.copy(graph_names), fps)
        elif self.model.state_count == 3:
            print('Not implemented...')


    def _save_history_video_2D(self, custom_name=None, compare_with=[], graph_names=[], fps=10):
        # Figure definition
        grid_spec = {'height_ratios': [19,1]}
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=grid_spec)

        # Figure title
        fig.suptitle(f"{self.model.state_count}-s {self.model.action_count}-a {self.model.observation_count}-o POMDP model solve history", fontsize=16)
        title = f'{self.expand_function} expand strat, {self.gamma}-gamma, {self.eps}-eps '

        # X-axis setting
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]

        # Colors and lines
        line_types = ['-', '--', '-.', ':']

        proxy = [Rectangle((0,0),1,1,fc = COLOR_LIST[a]['id']) for a in self.model.actions]

        # Solver list
        if isinstance(compare_with, ValueFunction) or isinstance(compare_with, MDP_SolverHistory):
            compare_with_list = [compare_with] # Single item
        else:
            compare_with_list = compare_with # Already group of items
        solver_histories = [self] + compare_with_list
        
        assert len(solver_histories) <= len(line_types), f"Plotting can only happen for up to {len(line_types)} solvers..."
        line_types = line_types[:len(solver_histories)]

        assert len(graph_names) in [0, len(solver_histories)], "Not enough graph names provided"
        if len(graph_names) == 0:
            graph_names.append('Main graph')
            for i in range(1,len(solver_histories)):
                graph_names.append(f'Comparison {i}')

        def plot_on_ax(history:Union[ValueFunction,Self], frame_i:int, ax, line_type:str):
            if isinstance(history, ValueFunction):
                value_function = history
            else:
                frame_i = frame_i if frame_i < len(history) else (len(history) - 1)
                value_function = history.value_functions[frame_i]

            alpha_vects = value_function.alpha_vector_array
            m = np.subtract(alpha_vects[:,1], alpha_vects[:,0])
            m = m.reshape(m.shape[0],1)

            x = np.linspace(0, 1, 100)
            x = x.reshape((1,x.shape[0])).repeat(m.shape[0],axis=0)
            y = np.add((m*x), alpha_vects[:,0].reshape(m.shape[0],1))

            for i, alpha in enumerate(value_function.alpha_vector_list):
                ax.plot(x[i,:], y[i,:], line_type, color=COLOR_LIST[alpha.action]['id'])

        def plt_frame(frame_i):
            ax1.clear()
            ax2.clear()

            self_frame_i = frame_i if frame_i < len(self) else (len(self) - 1)

            # Subtitle
            ax1.set_title(title + f'(Frame {frame_i})')

            # Color legend
            leg1 = ax1.legend(proxy, self.model.action_labels, loc='upper center')
            ax1.set_xticks(ticks, x_ticks)
            ax1.add_artist(leg1)

            # Line legend
            lines = []
            point = self.value_functions[self_frame_i].alpha_vector_list[0][0]
            for l in line_types:
                lines.append(Line2D([0,point],[0,point],linestyle=l))
            ax1.legend(lines, graph_names, loc='lower center')

            # Alpha vector plotting
            for history, line_type in zip(solver_histories, line_types):
                plot_on_ax(history, frame_i, ax1, line_type)

            # Belief plotting
            beliefs_x = self.belief_sets[frame_i].belief_array[:,1]
            ax2.scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c='red')
            ax2.get_yaxis().set_visible(False)
            ax2.axhline(0, color='black')

        max_steps = max([len(history) for history in solver_histories if not isinstance(history,ValueFunction)])
        ani = animation.FuncAnimation(fig, plt_frame, frames=max_steps, repeat=False)
        
        # File Title
        solved_time = self.run_ts.strftime('%Y%m%d_%H%M%S')

        video_title = f'{custom_name}-' if custom_name is not None else '' # Base
        video_title += f's{self.model.state_count}-a{self.model.action_count}-' # Model params
        video_title += f'{self.expand_function}-' # Expand function used
        video_title += f'g{self.gamma}-e{self.eps}-' # Solving params
        video_title += f'{solved_time}.mp4'

        # Video saving
        if not os.path.exists('./Results'):
            print('Folder does not exist yet, creating it...')
            os.makedirs('./Results')

        writervideo = animation.FFMpegWriter(fps=fps)
        ani.save('./Results/' + video_title, writer=writervideo)
        print(f'Video saved at \'Results/{video_title}\'...')
        plt.close()


class Solver(MDP_Solver):
    '''
    POMDP Model Solver - abstract class
    '''
    def solve(self, model: Model) -> tuple[ValueFunction, SolverHistory]:
        raise Exception("Method has to be implemented by subclass...")


class PBVI_Solver(Solver):
    '''
    The Point-Based Value Iteration solver for POMDP Models. It works in two steps, first the backup step that updates the alpha vector set that approximates the value function.
    Then, the expand function that expands the belief set.

    The various expand functions and the backup function have been implemented based on the pseudocodes found the paper from J. Pineau, G. Gordon, and S. Thrun, 'Point-based approximations for fast POMDP solving'

    ...
    Attributes
    ----------
    gamma: float (default 0.9)
        The learning rate, used to control how fast the value function will change after the each iterations.
    eps: float (default 0.001)
        The treshold for convergence. If the max change between value function is lower that eps, the algorithm is considered to have converged.
    expand_function: str (default 'ssea')
        The type of expand strategy to use to expand the belief set.
    expand_function_params: dict (Optional)
        Other required parameters to be sent to the expand function.

    Methods
    -------
    backup(model:pomdp.Model, belief_set:list[Belief], alpha_set:list[AlphaVector], discount_factor:float=0.9):
        The backup function, responsible to update the alpha vector set.
    expand_ssra(model:pomdp.Model, belief_set:list[Belief]):
        Random action, belief expansion strategy function.
    expand_ssga(model:pomdp.Model, belief_set:list[Belief], alpha_set:list[AlphaVector], eps:float=0.1):
        Expsilon greedy action, belief expansion strategy function.
    expand_ssea(model:pomdp.Model, belief_set:list[Belief], alpha_set:list[AlphaVector]):
        Exploratory action, belief expansion strategy function.
    expand_ger(model:pomdp.Model, belief_set, alpha_set):
        Greedy error reduction, belief expansion strategy function.
    expand():
        The general expand function, used to call the other expand_* functions.
    solve(model:pomdp.Model, expansions:int, horizon:int, initial_belief:Union[list[Belief], Belief, None]=None, initial_value_function:Union[ValueFunction,None]=None):
        The general solving function that will call iteratively the expand and the backup function.
    '''
    def __init__(self,
                 gamma:float=0.9,
                 eps:float=0.001,
                 expand_function:str='ssea',
                 expand_function_params:dict={}):
        self.gamma = gamma
        self.eps = eps
        self.expand_function = expand_function
        self.expand_function_params = expand_function_params


    def backup(self, model:Model, belief_set:BeliefSet, value_function:ValueFunction) -> ValueFunction:
        '''
        This function has purpose to update the set of alpha vectors. It does so in 3 steps:
        1. It creates projections from each alpha vector for each possible action and each possible observation
        2. It collapses this set of generated alpha vectors by taking the weighted sum of the alpha vectors weighted by the observation probability and this for each action and for each belief.
        3. Then it further collapses the set to take the best alpha vector and action per belief
        In the end we have a set of alpha vectors as large as the amount of beliefs.

        The alpha vectors are also pruned to avoid duplicates and remove dominated ones.

                Parameters:
                        model (POMDP): The model on which to run the backup method on.
                        belief_set (BeliefSet): The belief set to use to generate the new alpha vectors with.
                        alpha_set (ValueFunction): The alpha vectors to generate the new set from.
                        gamma (float): The discount factor used for training, default: 0.9.

                Returns:
                        new_alpha_set (ValueFunction): A list of updated alpha vectors.
        '''
        
        # Step 1
        vector_array = value_function.alpha_vector_array
        vectors_array_reachable_states = vector_array[np.arange(vector_array.shape[0])[:,None,None,None], model.reachable_states[None,:,:,:]]
        
        gamma_a_o_t = self.gamma * np.einsum('saor,vsar->aovs', model.reachable_transitional_observation_table, vectors_array_reachable_states)

        # Step 2
        belief_array = belief_set.belief_array
        best_alpha_ind = np.argmax(np.tensordot(belief_array, gamma_a_o_t, (1,3)), axis=3)

        best_alphas_per_o = gamma_a_o_t[model.actions[None,:,None,None], model.observations[None,None,:,None], best_alpha_ind[:,:,:,None], model.states[None,None,None,:]]

        alpha_a = np.sum(best_alphas_per_o, axis=2)
        alpha_a += model.expected_rewards_table.T

        # Step 3
        alpha_vectors = np.zeros(belief_array.shape)
        best_actions = []
        for i, b in enumerate(belief_array):
            best_ind = np.argmax(np.dot(alpha_a[i,:,:], b))
            alpha_vectors[i,:] = alpha_a[i, best_ind,:]
            best_actions.append(best_ind)

        new_value_function = ValueFunction(model, alpha_vectors, best_actions)

        # Pruning
        new_value_function = new_value_function.prune(level=1) # Just check for duplicates
                
        return new_value_function
    
    
    def expand_ssra(self, model:Model, belief_set:BeliefSet) -> BeliefSet:
        '''
        Stochastic Simulation with Random Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state (weighted by the belief) and taking a random action leading to a state s_p and a observation o.
        From this action a and observation o we can update our belief. 

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (BeliefSet): list of beliefs to expand on.

                Returns:
                        belief_set_new (BeliefSet): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        old_shape = belief_set.belief_array.shape

        new_belief_array = np.empty((old_shape[0] * 2, old_shape[1]))
        new_belief_array[:old_shape[0]] = belief_set.belief_array

        for i, belief_vector in enumerate(belief_set.belief_array):
            b = Belief(model, belief_vector)
            s = b.random_state()
            a = random.choice(model.actions)
            s_p = model.transition(s, a)
            o = model.observe(s_p, a)
            b_new = b.update(a, o)
            
            new_belief_array[i+old_shape[0]] = b_new.values
            
        return BeliefSet(model, new_belief_array)
    

    def expand_ssga(self, model:Model, belief_set:BeliefSet, value_function:ValueFunction, eps:float=0.1) -> BeliefSet:
        '''
        Stochastic Simulation with Greedy Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state s (weighted by the belief),
         then taking the best action a based on the belief with probability 'epsilon'.
        These lead to a new state s_p and a observation o.
        From this action a and observation o we can update our belief. 

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (BeliefSet): list of beliefs to expand on.
                        value_function (ValueFunction): Used to find the best action knowing the belief.
                        eps (float): Parameter tuning how often we take a greedy approach and how often we move randomly.

                Returns:
                        belief_set_new (BeliefSet): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        old_shape = belief_set.belief_array.shape

        new_belief_array = np.empty((old_shape[0] * 2, old_shape[1]))
        new_belief_array[:old_shape[0]] = belief_set.belief_array
                
        for i, belief_vector in enumerate(belief_set.belief_array):
            b = Belief(model, belief_vector)
            s = b.random_state()
            
            if random.random() < eps:
                a = random.choice(model.actions)
            else:
                best_alpha_index = np.argmax(np.dot(value_function.alpha_vector_array, b.values))
                a = value_function.actions[best_alpha_index]
            
            s_p = model.transition(s, a)
            o = model.observe(s_p, a)
            b_new = b.update(a, o)
            
            new_belief_array[i+old_shape[0]] = b_new.values
            
        return BeliefSet(model, new_belief_array)
    

    def expand_ssea(self, model:Model, belief_set:BeliefSet) -> BeliefSet:
        '''
        Stochastic Simulation with Exploratory Action.
        Simulates running steps forward for each possible action knowing we are a state s, chosen randomly with according to the belief probability.
        These lead to a new state s_p and a observation o for each action.
        From all these and observation o we can generate updated beliefs. 
        Then it takes the belief that is furthest away from other beliefs, meaning it explores the most the belief space.

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (BeliefSet): list of beliefs to expand on.

                Returns:
                        belief_set_new (BeliefSet): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        old_shape = belief_set.belief_array.shape

        new_belief_array = np.empty((old_shape[0] * 2, old_shape[1]))
        new_belief_array[:old_shape[0]] = belief_set.belief_array
        
        for i, belief_vector in enumerate(belief_set.belief_array):
            b = Belief(model, belief_vector)
            best_b = None
            max_dist = -math.inf
            
            for a in model.actions:
                s = b.random_state()
                s_p = model.transition(s, a)
                o = model.observe(s_p, a)
                b_a = b.update(a, o)
                
                # Check distance with other beliefs
                min_dist = min(float(np.linalg.norm(b_p - b_a.values)) for b_p in new_belief_array)
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_b = b_a
            
            assert best_b is not None
            new_belief_array[i+old_shape[0]] = best_b.values
        
        return BeliefSet(model, new_belief_array)
    

    def expand_ger(self, model:Model, belief_set:BeliefSet, value_function:ValueFunction) -> BeliefSet:
        '''
        Greedy Error Reduction

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (BeliefSet): list of beliefs to expand on.
                        value_function (ValueFunction): Used to find the best action knowing the belief.

                Returns:
                        belief_set_new (BeliefSet): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        print('Not implemented')
        return []


    def expand(self, model:Model, belief_set:BeliefSet, **function_specific_parameters) -> BeliefSet:
        '''
        Central method to call one of the functions for a particular expansion strategy:
            - Stochastic Simulation with Random Action (ssra)
            - Stochastic Simulation with Greedy Action (ssga)
            - Stochastic Simulation with Exploratory Action (ssea)
            - Greedy Error Reduction (ger) - not implemented
                
                Parameters:
                        model (pomdp.Model): The model on which to run the belief expansion on.
                        belief_set (BeliefSet): The set of beliefs to expand.
                        function_specific_parameters: Potential additional parameters necessary for the specific expand function.

                Returns:
                        belief_set_new (BeliefSet): The belief set the expansion function returns. 
        '''
        if self.expand_function in 'expand_ssra':
            return self.expand_ssra(model=model, belief_set=belief_set)
        
        elif self.expand_function in 'expand_ssga':
            args = {arg: function_specific_parameters[arg] for arg in ['value_function', 'eps'] if arg in function_specific_parameters}
            return self.expand_ssga(model=model, belief_set=belief_set, **args)
        
        elif self.expand_function in 'expand_ssea':
            return self.expand_ssea(model=model, belief_set=belief_set)
        
        elif self.expand_function in 'expand_ger':
            args = {arg: function_specific_parameters[arg] for arg in ['value_function'] if arg in function_specific_parameters}
            return self.expand_ger(model=model, belief_set=belief_set, **args)
        
        return []


    def solve(self,
              model:Model,
              expansions:int,
              horizon:int,
              initial_belief:Union[BeliefSet, Belief, None]=None,
              initial_value_function:Union[ValueFunction,None]=None,
              expand_prune_level:Union[int,None]=None,
              print_progress:bool=True
              ) -> tuple[ValueFunction, SolverHistory]:
        '''
        Main loop of the Point-Based Value Iteration algorithm.
        It consists in 2 steps, Backup and Expand.
        1. Expand: Expands the belief set base with a expansion strategy given by the parameter expand_function
        2. Backup: Updates the alpha vectors based on the current belief set

        Depending on the expand strategy chosen, various extra parameters are needed. List of the available expand strategies and their extra required parameters:
            - ssra: Stochastic Simulation with Random Action. Extra params: /
            - ssga: Stochastic Simulation with Greedy Action. Extra params: epsilon (float)
            - ssea: Stochastic Simulation with Exploratory Action. Extra params: /
            - ger: Greedy Error Reduction. Extra params: /

                Parameters:
                        model (pomdp.Model) - The model to solve.
                        expansions (int) - How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
                        horizon (int) - How many times the alpha vector set must be updated every time the belief set is expanded.
                        initial_belief (BeliefSet, Belief) - Optional: An initial list of beliefs to start with.
                        initial_value_function (ValueFunction) - Optional: An initial value function to start the solving process with.
                        expand_prune_level (int) - Optional: Parameter to prune the value function further before the expand function.
                        print_progress (bool): Whether or not to print out the progress of the value iteration process. (Default: True)

                Returns:
                        value_function (ValueFunction): The alpha vectors approximating the value function.
        '''

        # Initial belief #TODO Modify this
        if isinstance(initial_belief, BeliefSet):
            belief_set = initial_belief
        elif isinstance(initial_belief, Belief):
            belief_set = BeliefSet(model, [initial_belief])
        else:
            belief_set = BeliefSet(model, [Belief(model)])
        
        # Initial value function
        if initial_value_function is None:
            value_function = ValueFunction(model, model.expected_rewards_table.T, model.actions)
        else:
            value_function = initial_value_function

        # Convergence check boundary
        max_allowed_change = self.eps * (self.gamma / (1-self.gamma))

        # History tracking
        solver_history = SolverHistory(model=model,
                                       initial_value_function=value_function,
                                       initial_belief_set=belief_set,
                                       expand_function=self.expand_function,
                                       gamma=self.gamma,
                                       eps=self.eps)

        # Loop
        for expansion_i in range(expansions) if not print_progress else trange(expansions, desc='Expansions'):

            # 0: Prune value function at a higher level between expansions
            if expand_prune_level is not None:
                value_function = value_function.prune(expand_prune_level)

            # 1: Expand belief set
            belief_set = self.expand(model=model, belief_set=belief_set, value_function=value_function, **self.expand_function_params)

            old_max_val_per_belief = None

            # 2: Backup, update value function (alpha vector set)
            for _ in range(horizon) if not print_progress else trange(horizon, desc=f'Backups {expansion_i}'):
                value_function = self.backup(model, belief_set, value_function)

                # History tracking
                solver_history.add(value_function, belief_set)

                # convergence check
                max_val_per_belief = np.max(np.matmul(belief_set.belief_array, value_function.alpha_vector_array.T), axis=1)
                if old_max_val_per_belief is not None:
                    max_change = np.max(np.abs(max_val_per_belief - old_max_val_per_belief))
                    if max_change < max_allowed_change:
                        print('Converged early...')
                        return value_function, solver_history
                old_max_val_per_belief = max_val_per_belief

        return value_function, solver_history


class SimulationHistory(MDP_SimulationHistory):
    '''
    Class to represent a list of rewards received during a Simulation.
    The main purpose of the class is to provide a set of visualization options of the rewards received.

    Multiple types of plots can be done:
        - Totals: to plot a graph of the accumulated rewards over time.
        - Moving average: to plot the moving average of the rewards received over time.
        - Histogram: to plot a histogram of the various rewards received.

    ...

    Attributes
    ------------
    model: mdp.Model
        The model on which the simulation happened on.
    start_state: int
        The initial state in the simulation.
    start_belief: Belief
        The initial belief the agent starts with during the simulation.

    Methods
    -------
    plot_simulation_steps(size:int=5):
        Function to plot the final state of the simulation will all states the agent passed through.
    save_simulation_video(custom_name:Union[str,None]=None, fps:int=1):
        Function to save a video of the simulation history with all the states it passes through and the believes it explores.
    add(action:int, reward, next_state:int, next_belief:Belief, observation:int):
        Function to add a step in the simulation history.
    '''
    def __init__(self, model:Model, start_state:int, start_belief:Belief):
        super().__init__(model, start_state)
        self.beliefs = [start_belief]
        self.observations = []


    def add(self, action:int, reward, next_state:int, next_belief:Belief, observation:int) -> None:
        super().add(action, reward, next_state)
        self.beliefs.append(next_belief)
        self.observations.append(observation)


    # Overwritten
    def _plot_to_frame_on_ax(self, frame_i, ax):
        # Data
        data = np.array(self.grid_point_sequence)[:(frame_i+1),:]
        belief = self.beliefs[frame_i]

        # Ticks
        dimensions = (len(self.model.grid_states), len(self.model.grid_states[0]))
        x_ticks = [i for i in range(dimensions[1])]
        y_ticks = [i for i in range(dimensions[0])]

        # Plotting
        ax.clear()
        ax.set_title(f'Simulation (Frame {frame_i})')

        ax.imshow(belief.grid_values,cmap='Blues')
        ax.plot(data[:, 0], data[:, 1], color='red')
        ax.scatter(data[:, 0], data[:, 1], color='red')

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)


class Simulation(MDP_Simulation):
    '''
    Class to reprensent a simulation process for a POMDP model.
    An initial random state is given and action can be applied to the model that impact the actual state of the agent along with returning a reward and an observation.

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
        super().__init__(model)
        self.model = model


    def run_action(self, a:int) -> tuple[Union[int,float], int]:
        '''
        Run one step of simulation with action a.

                Parameters:
                        a (int): the action to take in the simulation.

                Returns:
                        r (int, float): the reward given when doing action a in state s and landing in state s_p. (s and s_p are hidden from agent)
                        o (int): the observation following the action applied on the previous state
        '''
        assert not self.is_done, "Action run when simulation is done."

        s = self.agent_state
        s_p = self.model.transition(s, a)
        o = self.model.observe(s_p, a)
        r = self.model.reward(s, a, s_p, o)

        # Update agent state
        self.agent_state = s_p

        # State Done check
        if s_p in self.model.end_states:
            self.is_done = True

        # Action Done check
        if a in self.model.end_actions:
            self.is_done = True

        return (r, o)


class Agent:
    '''
    The class of an Agent running on a POMDP model.
    It has the ability to train using a given solver (here PBVI_Solver).
    Then, once trained, it can simulate actions with a given Simulation,
    either for a given amount of steps or until a single reward is received.

    ...

    Attributes
    ----------
    model: pomdp.Model
        The model in which the agent can run
    
    Methods
    -------
    train(solver: PBVI_Solver):
        Runs the solver on the agent's model in order to retrieve a value function.
    get_best_action(belief:Belief):
        Retrieves the best action from the value function given a belief (the belief of the agent being in a certain state).
    simulate(simulator:Simulation, max_steps:int=1000):
        Simulate the process on the Agent's model with the given simulator for up to max_steps iterations.
    run_n_simulations(simulator:Simulation, n:int):
        Runs n times the simulate() function.
    '''
    def __init__(self, model:Model) -> None:
        super().__init__()

        self.model = model
        self.value_function = None


    def train(self, solver:PBVI_Solver, expansions:int, horizon:int) -> None:
        '''
        Method to train the agent using a given solver.
        The solver will provide a value function that will map beliefs in belief space to actions.

                Parameters:
                        solver (PBVI_Solver): The solver to run.
                        expansions (int): How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
                        horizon (int): How many times the alpha vector set must be updated every time the belief set is expanded.
        '''
        self.value_function, solve_history = solver.solve(self.model, expansions, horizon)


    def get_best_action(self, belief:Belief) -> int:
        '''
        Function to retrieve the best action for a given belief based on the value function retrieved from the training.

                Parameters:
                        belief (Belief): The belief to get the best action with.
                
                Returns:
                        best_action (int): The best action found.
        '''
        assert self.value_function is not None, "No value function, training probably has to be run..."

        best_vector = np.argmax(np.dot(self.value_function.alpha_vector_array, belief.values))
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
                        max_steps (int): the max amount of steps the simulation can run for.
                        start_state (int): The state the agent should start in, if not provided, will be set at random based on start probabilities of the model (Default: random)
                        print_progress (bool): Whether or not to print out the progress of the simulation. (Default: True)
                        print_stats (bool): Whether or not to print simulation statistics at the end of the simulation (Default: True)

                Returns:
                        history (SimulationHistory): A list of rewards with the additional functionality that the can be plot with the plot() function.
        '''
        if simulator is None:
            simulator = Simulation(self.model)

        s = simulator.initialize_simulation(start_state=start_state) # s is only used for the simulation history
        belief = Belief(self.model)

        history = SimulationHistory(self.model, start_state=s, start_belief=belief)

        sim_start_ts = datetime.now()

        # Simulation loop
        for _ in (trange(max_steps) if print_progress else range(max_steps)):
            # Play best action
            a = self.get_best_action(belief)
            r,o = simulator.run_action(a)

            # Update the belief
            new_belief = belief.update(a, o)

            # Post action history recording
            history.add(action=a, next_state=simulator.agent_state, next_belief=new_belief, reward=r, observation=o)

            # Replace belief
            belief = new_belief

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
            history = self.simulate(simulator, max_steps, start_state, False, False)
            all_final_rewards.append(np.sum(history.rewards))
            all_sim_length.append(len(history.states))

        if print_stats:
            sim_end_ts = datetime.now()
            print(f'All {n} simulations done:')
            print(f'\t- Average runtime (s): {((sim_end_ts - sim_start_ts).total_seconds() / n)}')
            print(f'\t- Average step count: {(sum(all_sim_length) / n)}')
            print(f'\t- Average total rewards: {(sum(all_final_rewards) / n)}')

        return all_final_rewards
    

def load_POMDP_file(file_name:str) -> Tuple[Model, PBVI_Solver]:
    '''
    Function to load files of .POMDP format.
    This file format was implemented by Cassandra and the specifications of the format can be found here: https://pomdp.org/code/pomdp-file-spec.html
     
    Then, example models can be found on the following page: https://pomdp.org/examples/

            Parameters:
                    file_name (str): The name and path of the file to be loaded.

            Returns:
                    loaded_model (pomdp.Model): A POMDP model with the parameters found in the POMDP file. 
                    loaded_solver (PBVI_Solver): A solver with the gamma parameter from the POMDP file.
    '''
    # Working params
    gamma_param = 1.0
    values_param = '' # To investigate
    state_count = -1
    action_count = -1
    observation_count = -1

    model_params = {}
    reading:str = ''
    read_lines = 0

    with open(file_name) as file:
        for line in file:
            if line.startswith(('#', '\n')):
                continue

            # Split line
            line_items = line.replace('\n','').strip().split()

            # Discount factor
            if line.startswith('discount'):
                gamma_param = float(line_items[-1])
            
            # Value (either reward or cost)
            elif line.startswith('values'):
                values_param = line_items[-1] # To investigate

            # States
            elif line.startswith('states'):
                if line_items[-1].isnumeric():
                    state_count = int(line_items[-1])
                    model_params['states'] = [f's{i}' for i in range(state_count)]
                else:
                    model_params['states'] = line_items[1:]
                    state_count = len(model_params['states'])

            # Actions
            elif line.startswith('actions'):
                if line_items[-1].isnumeric():
                    action_count = int(line_items[-1])
                    model_params['actions'] = [f'a{i}' for i in range(action_count)]
                else:
                    model_params['actions'] = line_items[1:]
                    action_count = len(model_params['actions'])

            # Observations
            elif line.startswith('observations'):
                if line_items[-1].isnumeric():
                    observation_count = int(line_items[-1])
                    model_params['observations'] = [f'o{i}' for i in range(observation_count)]
                else:
                    model_params['observations'] = line_items[1:]
                    observation_count = len(model_params['observations'])
            
            # Start
            elif line.startswith('start'):
                if len(line_items) == 1:
                    reading = 'start'
                else:
                    assert len(line_items[1:]) == state_count, 'Not enough states in initial belief'
                    model_params['start_probabilities'] = np.array([float(item) for item in line_items[1:]])
            elif reading == 'start':
                assert len(line_items) == state_count, 'Not enough states in initial belief'
                model_params['start_probabilities'] = np.array([float(item) for item in line_items])
                reading = 'None'

            # ----------------------------------------------------------------------------------------------
            # Transition table
            # ----------------------------------------------------------------------------------------------
            if ('states' in model_params) and ('actions' in model_params) and ('observations' in model_params) and not ('transition_table' in model_params):
                model_params['transition_table'] = np.full((state_count, action_count, state_count), 0.0)
            
            if line.startswith('T'):
                transition_params = line.replace(':',' ').split()[1:]
                transition_params = transition_params[:-1] if (line.count(':') == 3) else transition_params

                ids = []
                for i, param in enumerate(transition_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                    elif i in [1,2]:
                        ids.append(np.arange(state_count) if param == '*' else [model_params['states'].index(param)])
                    else:
                        raise Exception('Cant load more than 3 parameters for transitions')

                # single item
                if len(transition_params) == 3:
                    for s in ids[1]:
                        for a in ids[0]:
                            for s_p in ids[2]:
                                model_params['transition_table'][s, a, s_p] = float(line_items[-1])
                
                # More items
                else:
                    reading = f'T{len(transition_params)} ' + ' '.join(transition_params)
                    
            # Reading action-state line
            elif reading.startswith('T2'):
                transition_params = reading.split()[1:]

                ids = []
                for i, param in enumerate(transition_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                    else:
                        ids.append(np.arange(state_count) if param == '*' else [model_params['states'].index(param)])

                for a in ids[0]:
                    for s in ids[1]:
                        # Uniform
                        if 'uniform' in line_items:
                            model_params['transition_table'][s, a, :] = np.ones(state_count) / state_count
                            continue

                        for s_p, item in enumerate(line_items):
                            model_params['transition_table'][s, a, s_p] = float(item)

                reading = ''

            # Reading action matrix
            elif reading.startswith('T1'):
                s = read_lines

                transition_params = reading.split()[1:]

                ids = []
                for i, param in enumerate(transition_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                
                for a in ids[0]:
                    # Uniform
                    if 'uniform' in line_items:
                        model_params['transition_table'][:, a, :] = np.ones((state_count, state_count)) / state_count
                        reading = ''
                        continue
                    # Identity
                    if 'identity' in line_items:
                        model_params['transition_table'][:, a, :] = np.eye(state_count)
                        reading = ''
                        continue

                    for s_p, item in enumerate(line_items):
                        model_params['transition_table'][s, a, s_p] = float(item)

                if ('uniform' not in line_items) and ('identity' not in line_items):
                    read_lines += 1
                
                if read_lines == state_count:
                    reading = ''
                    read_lines = 0


            # ----------------------------------------------------------------------------------------------
            # Observation table
            # ----------------------------------------------------------------------------------------------
            if ('states' in model_params) and ('actions' in model_params) and ('observations' in model_params) and not ('observation_table' in model_params):
                model_params['observation_table'] = np.full((state_count, action_count, observation_count), 0.0)

            if line.startswith('O'):
                observation_params = line.replace(':',' ').split()[1:]
                observation_params = observation_params[:-1] if (line.count(':') == 3) else observation_params

                ids = []
                for i, param in enumerate(observation_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                    elif i == 1:
                        ids.append(np.arange(state_count) if param == '*' else [model_params['states'].index(param)])
                    elif i == 2:
                        ids.append(np.arange(observation_count) if param == '*' else [model_params['observations'].index(param)])
                    else:
                        raise Exception('Cant load more than 3 parameters for observations')

                # single item
                if len(observation_params) == 3:
                    for a in ids[0]:
                        for s_p in ids[1]:
                            for o in ids[2]:
                                model_params['observation_table'][s_p, a, o] = float(line_items[-1])
                
                # More items
                else:
                    reading = f'O{len(observation_params)} ' + ' '.join(observation_params)
                    
            # Reading action-state line
            elif reading.startswith('O2'):
                observation_params = reading.split()[1:]

                ids = []
                for i, param in enumerate(observation_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                    else:
                        ids.append(np.arange(state_count) if param == '*' else [model_params['states'].index(param)])

                for a in ids[0]:
                    for s_p in ids[1]:
                        # Uniform
                        if 'uniform' in line_items:
                            model_params['observation_table'][s_p, a, :] = np.ones(observation_count) / observation_count
                            continue

                        for o, item in enumerate(line_items):
                            model_params['observation_table'][s_p, a, o] = float(item)

                reading = ''

            # Reading action matrix
            elif reading.startswith('O1'):
                s_p = read_lines

                observation_params = reading.split()[1:]
                ids = []
                for i, param in enumerate(observation_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    else:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])

                for a in ids[0]:
                    # Uniform
                    if 'uniform' in line_items:
                        model_params['observation_table'][:, a, :] = np.ones((state_count, observation_count)) / observation_count
                        reading = ''
                        continue

                    for o, item in enumerate(line_items):
                        model_params['observation_table'][s_p, a, o] = float(item)

                if 'uniform' not in line_items:
                    read_lines += 1
                
                if read_lines == state_count:
                    reading = ''
                    read_lines = 0


            # ----------------------------------------------------------------------------------------------
            # Rewards table
            # ----------------------------------------------------------------------------------------------
            if ('states' in model_params) and ('actions' in model_params) and ('observations' in model_params) and not ('immediate_reward_table' in model_params):
                model_params['immediate_reward_table'] = np.full((state_count, action_count, state_count, observation_count), 0.0)

            if line.startswith('R'):
                reward_params = line.replace(':',' ').split()[1:]
                reward_params = reward_params[:-1] if (line.count(':') == 4) else reward_params

                ids = []
                for i, param in enumerate(reward_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                    elif i in [1,2]:
                        ids.append(np.arange(state_count) if param == '*' else [model_params['states'].index(param)])
                    elif i == 3:
                        ids.append(np.arange(observation_count) if param == '*' else [model_params['observations'].index(param)])
                    else:
                        raise Exception('Cant load more than 4 parameters for rewards')

                # single item
                if len(reward_params) == 4:
                    for a in ids[0]:
                        for s in ids[1]:
                            for s_p in ids[2]:
                                for o in ids[3]:
                                    model_params['immediate_reward_table'][s, a, s_p, o] = float(line_items[-1])
                
                elif len(reward_params) == 1:
                    raise Exception('Need more than 1 parameter for rewards')

                # More items
                else:
                    reading = f'R{len(reward_params)} ' + ' '.join(reward_params)
                    
            # Reading action-state line
            elif reading.startswith('R3'):
                reward_params = reading.split()[1:]
                
                ids = []
                for i, param in enumerate(reward_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                    else:
                        ids.append(np.arange(state_count) if param == '*' else [model_params['states'].index(param)])

                for a in ids[0]:
                    for s in ids[1]:
                        for s_p in ids[2]:
                            for o, item in enumerate(line_items):
                                model_params['immediate_reward_table'][s, a, s_p, o] = float(item)

                reading = ''

            # Reading action matrix
            elif reading.startswith('R2'):
                s_p = read_lines

                reward_params = reading.split()[1:]
                ids = []
                for i, param in enumerate(reward_params):
                    if param.isnumeric():
                        ids.append([int(param)])
                    elif i == 0:
                        ids.append(np.arange(action_count) if param == '*' else [model_params['actions'].index(param)])
                    else:
                        ids.append(np.arange(state_count) if param == '*' else [model_params['states'].index(param)])

                for a in ids[0]:
                    for s in ids[1]:
                        for o, item in enumerate(line_items):
                            model_params['immediate_reward_table'][s, a, s_p, o] = float(item)

                read_lines += 1
                if read_lines == state_count:
                    reading = ''
                    read_lines = 0

    # Generation of output
    loaded_model = Model(**model_params)
    loaded_solver = PBVI_Solver(gamma=gamma_param)

    return (loaded_model, loaded_solver)