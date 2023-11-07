from datetime import datetime
from inspect import signature
from matplotlib import animation, cm, colors, ticker, patches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tqdm.auto import trange
from typing import Tuple, Union

import copy
import math
import os
import random

import numpy as np
gpu_support = False
try:
    import cupy as cp
    gpu_support = True
except:
    print('[Warning] Cupy could not be loaded: GPU support is not available.')

from src.mdp import log
from src.mdp import ValueFunction
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
    'rgb': [int(value.lstrip('#')[i:i + (len(value)-1) // 3], 16) for i in range(0, (len(value)-1), (len(value)-1) // 3)]
    } for item, value in colors.TABLEAU_COLORS.items()] # type: ignore

COLOR_ARRAY = np.array([c['rgb'] for c in COLOR_LIST])


class Model(MDP_Model):
    '''
    POMDP Model class. Partially Observable Markov Decision Process Model.

    ...

    Parameters
    ----------
    states : int or list[str] or list[list[str]]
        A list of state labels or an amount of states to be used. Also allows to provide a matrix of states to define a grid model.
    actions : int or list
        A list of action labels or an amount of actions to be used.
    observations : int or list
        A list of observation labels or an amount of observations to be used
    transitions : array-like or function, optional
        The transitions between states, an array can be provided and has to be |S| x |A| x |S| or a function can be provided. 
        If a function is provided, it has be able to deal with np.array arguments.
        If none is provided, it will be randomly generated.
    reachable_states : array-like, optional
        A list of states that can be reached from each state and actions. It must be a matrix of size |S| x |A| x |R| where |R| is the max amount of states reachable from any given state and action pair.
        It is optional but useful for speedup purposes.
    rewards : array-like or function, optional
        The reward matrix, has to be |S| x |A| x |S|.
        A function can also be provided here but it has to be able to deal with np.array arguments.
        If provided, it will be use in combination with the transition matrix to fill to expected rewards.
    observation_table : array-like or function, optional
        The observation matrix, has to be |S| x |A| x |O|. If none is provided, it will be randomly generated.
    rewards_are_probabilistic: bool, default=False
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.
    state_grid : array-like, optional
        If provided, the model will be converted to a grid model.
    start_probabilities : list, optional
        The distribution of chances to start in each state. If not provided, there will be an uniform chance for each state. It is also used to represent a belief of complete uncertainty.
    end_states : list, optional
        Entering either state in the list during a simulation will end the simulation.
    end_action : list, optional
        Playing action of the list during a simulation will end the simulation.

    Attributes
    ----------
    states : np.ndarray
        A 1D array of states indices. Used to loop over states.
    state_labels : list[str]
        A list of state labels. (To be mainly used for plotting)
    state_count : int
        How many states are in the Model.
    state_grid : np.ndarray
        The state indices organized as a 2D grid. (Used for plotting purposes)
    actions : np.ndarry
        A 1D array of action indices. Used to loop over actions.
    action_labels : list[str]
        A list of action labels. (To be mainly used for plotting)
    action_count : int
        How many action are in the Model.
    observations : np.ndarray
        A 1D array of observation indices. Used to loop over obervations.
    observation_labels : list[str]
        A list of observation labels. (To be mainly used for plotting)
    observation_count : int
        How many observations can be made in the Model.
    transition_table : np.ndarray
        A 3D matrix of the transition probabilities.
        Can be None in the case a transition function is provided instead.
        Note: When possible, use reachable states and reachable probabilities instead.
    transition_function : function
        A callable function taking 3 arguments: s, a, s_p; and returning a float between 0.0 and 1.0.
        Can be None in the case a transition table is provided instead.
        Note: When possible, use reachable states and reachable probabilities instead.
    observation_table : np.ndarray
        A 3D matrix of shape S x A x O representing the probabilies of obsevating o when taking action a and leading to state s_p.
    reachable_states : np.ndarray
        A 3D array of the shape S x A x R, where R is max amount to states that can be reached from any state-action pair.
    reachable_probabilities : np.ndarray
        A 3D array of the same shape as reachable_states, the array represent the probability of reaching the state pointed by the reachable_states matrix.
    reachable_state_count : int
        The maximum of states that can be reached from any state-action combination.
    reachable_transitional_observation_table : np.ndarray
        A 4D array of shape S x A x O x R, representing the probabiliies of landing if each reachable state r, while observing o after having taken action a from state s.
        Mainly used to speedup repeated operations in solver.
    immediate_reward_table : np.ndarray
        A 3D matrix of shape S x A x S x O of the reward that will received when taking action a, in state s, landing in state s_p, and observing o.
        Can be None in the case an immediate rewards function is provided instead.
    immediate_reward_function : function
        A callable function taking 4 argments: s, a, s_p, o and returning the immediate reward the agent will receive.
        Can be None in the case an immediate rewards function is provided instead.
    expected_reward_table : np.ndarray
        A 2D array of shape S x A. It represents the rewards that is expected to be received when taking action a from state s.
        It is made by taking the weighted average of immediate rewards with the transitions and the observation probabilities.
    start_probabilities : np.ndarray
        A 1D array of length |S| containing the probility distribution of the agent starting in each state.
    rewards_are_probabilisitic : bool
        Whether the immediate rewards are probabilitic, ie: returning a 0 or 1 based on the reward that is considered to be a probability.
    end_states : list[int]
        A list of states that, when reached, terminate a simulation.
    end_actions : list[int]
        A list of actions that, when taken, terminate a simulation.
    is_on_gpu : bool
        Whether the numpy array of the model are stored on the gpu or not.
    gpu_model : mdp.Model
        An equivalent model with the np.ndarray objects on GPU. (If already on GPU, returns self)
    cpu_model : mdp.Model
        An equivalent model with the np.ndarray objects on CPU. (If already on CPU, returns self)
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
                 state_grid=None,
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
                         state_grid=state_grid,
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

        self._min_reward = float(np.min(reachable_rewards))
        self._max_reward = float(np.max(reachable_rewards))

        self.expected_rewards_table = np.einsum('saor,saro->sa', self.reachable_transitional_observation_table, reachable_rewards)

        duration = (datetime.now() - start_ts).total_seconds()
        log(f'    > Done in {duration:.3f}s')


    def reward(self, s:int, a:int, s_p:int, o:int) -> Union[int,float]:
        '''
        Returns the rewards of playing action a when in state s and landing in state s_p.
        If the rewards are probabilistic, it will return 0 or 1.

        Parameters
        ----------
        s : int
            The current state.
        a : int
            The action taking in state s.
        s_p : int
            The state landing in after taking action a in state s
        o : int
            The observation that is done after having played action a in state s and landing in s_p

        Returns
        -------
        reward : int or float
            The reward received.
        '''
        reward = float(self.immediate_reward_table[s,a,s_p,o] if self.immediate_reward_table is not None else self.immediate_reward_function(s,a,s_p,o))
        if self.rewards_are_probabilistic:
            rnd = random.random()
            return 1 if rnd < reward else 0
        else:
            return reward
    

    def observe(self, s_p:int, a:int) -> int:
        '''
        Returns a random observation knowing action a is taken from state s, it is weighted by the observation probabilities.

        Parameters
        ----------
        s_p : int
            The state landed on after having done action a.
        a : int
            The action to take.

        Returns
        -------
        o : int
            A random observation.
        '''
        xp = cp if self.is_on_gpu else np
        o = int(xp.random.choice(a=self.observations, size=1, p=self.observation_table[s_p,a])[0])
        return o


class Belief:
    '''
    A class representing a belief in the space of a given model. It is the belief to be in any combination of states:
    eg:
        - In a 2 state POMDP: a belief of (0.5, 0.5) represent the complete ignorance of which state we are in. Where a (1.0, 0.0) belief is the certainty to be in state 0.

    The belief update function has been implemented based on the belief update define in the paper of J. Pineau, G. Gordon, and S. Thrun, 'Point-based approximations for fast POMDP solving'

    ...

    Parameters
    ----------
    model : pomdp.Model
        The model on which the belief applies on.
    values : np.ndarray, optional
        A vector of the probabilities to be in each state of the model. The sum of the probabilities must sum to 1.
        If not specified, it will be set as the start probabilities of the model.

    Attributes
    ----------
    model : pomdp.Model
    values : np.ndarray
    '''
    def __init__(self, model:Model, values:Union[np.ndarray,None]=None):
        assert model is not None
        self.model = model

        if values is not None:
            assert values.shape[0] == model.state_count, "Belief must contain be of dimension |S|"

            xp = np if not gpu_support else cp.get_array_module(values)

            prob_sum = xp.sum(values)
            rounded_sum = xp.round(prob_sum, decimals=3)
            assert rounded_sum == 1.0, f"States probabilities in belief must sum to 1 (found: {prob_sum}; rounded {rounded_sum})"

            self._values = values
        else:
            self._values = model.start_probabilities

    
    @property
    def values(self) -> np.ndarray:
        '''
        An array of the probability distribution to be in each state.
        '''
        return self._values
    

    def update(self, a:int, o:int) -> 'Belief':
        '''
        Returns a new belief based on this current belief, the most recent action (a) and the most recent observation (o).

        Parameters
        ----------
        a : int
            The most recent action.
        o : int
            The most recent observation.

        Returns
        -------
        new_belief : Belief
            An updated belief
        '''
        xp = np if not gpu_support else cp.get_array_module(self._values)

        reachable_state_probabilities = self.model.reachable_transitional_observation_table[:,a,o,:] * self.values[:,None]
        new_state_probabilities = xp.bincount(self.model.reachable_states[:,a,:].flatten(), weights=reachable_state_probabilities.flatten(), minlength=self.model.state_count)
        
        # Normalization
        new_state_probabilities /= xp.sum(new_state_probabilities)

        # Generation of new belief from new state probabilities
        new_belief = super().__new__(self.__class__)
        new_belief.model = self.model
        new_belief._values = new_state_probabilities

        return new_belief
    

    def generate_successors(self) -> list['Belief']:
        '''
        Function to generate a set of belief that can be reached for each actions and observations available in the model.

        Returns
        -------
        successor_beliefs : list[Belief]
            The successor beliefs.
        '''
        successor_beliefs = []
        for a in self.model.actions:
            for o in self.model.observations:
                successor_beliefs.append(self.update(a,o))

        return successor_beliefs


    def random_state(self) -> int:
        '''
        Returns a random state of the model weighted by the belief probabily.

        Returns
        -------
        rand_s : int
            A random state.
        '''
        xp = np if not gpu_support else cp.get_array_module(self._values)

        rand_s = int(xp.random.choice(a=self.model.states, size=1, p=self._values)[0])
        return rand_s
    

    def plot(self, size:int=5) -> None:
        '''
        Function to plot a heatmap of the belief distribution if the belief is of a grid model.

        Parameters
        ----------
        size : int, default=5
            The scale of the plot.
        '''
        # Plot setup
        plt.figure(figsize=(size*1.2,size))

        # Ticks
        dimensions = self.model.state_grid.shape
        x_ticks = np.arange(0, dimensions[1], (1 if dimensions[1] < 10 else int(dimensions[1] / 10)))
        y_ticks = np.arange(0, dimensions[0], (1 if dimensions[0] < 5 else int(dimensions[0] / 5)))

        plt.xticks(x_ticks)
        plt.yticks(y_ticks)

        # Title
        plt.title(f'Belief (probability distribution over states)')

        # Actual plot
        grid_values = self._values[self.model.state_grid]
        plt.imshow(grid_values,cmap='Blues')
        plt.colorbar()
        plt.show()


class BeliefSet:
    '''
    Class to represent a set of beliefs with regard to a POMDP model.
    It has the purpose to store the beliefs in numpy array format and be able to conver it to a list of Belief class objects.
    This class also provides the option to display the beliefs when operating on a 2 or 3d space with the plot() function.
    
    ...

    Parameters
    ----------
    model : pomdp.Model
        The model on which the beliefs apply.
    beliefs : list[Belief] | np.ndarray
        The actual set of beliefs.

    Attributes
    ----------
    model : pomdp.Model
    belief_array : np.ndarray
        A 2D array of shape N x S of N belief vectors.
    belief_list : list[Belief]
        A list of N Belief object.
    '''
    def __init__(self, model:Model, beliefs:Union[list[Belief],np.ndarray]) -> None:
        self.model = model

        self._belief_list = None
        self._belief_array = None

        self.is_on_gpu = False

        if isinstance(beliefs, list):
            assert all(len(b.values) == model.state_count for b in beliefs), f"Beliefs in belief list provided dont all have shape ({model.state_count},)"
            self._belief_list = beliefs

            # Check if on gpu and make sure all beliefs are also on the gpu
            if (len(beliefs) > 0) and gpu_support and cp.get_array_module(beliefs[0].values) == cp:
                assert all(cp.get_array_module(b.values) == cp for b in beliefs), "Either all or none of the alpha vectors should be on the GPU, not just some."
                self.is_on_gpu = True
        else:
            assert beliefs.shape[1] == model.state_count, f"Belief array provided doesnt have the right shape (expected (-,{model.state_count}), received {beliefs.shape})"
            
            self._belief_list = []
            for belief_values in beliefs:
                self._belief_list.append(Belief(model, belief_values))

            # Check if array is on gpu
            if gpu_support and cp.get_array_module(beliefs) == cp:
                self.is_on_gpu = True

        # # Deduplication
        # self._uniqueness_dict = {belief.values.tobytes(): belief for belief in self._belief_list}
        # self._belief_list = list(self._uniqueness_dict.values())


    @property
    def belief_array(self) -> np.ndarray:
        '''
        A matrix of size N x S containing N belief vectors. If belief set is stored as a list of Belief objects, the matrix of beliefs will be generated from them.
        '''
        xp = cp if (gpu_support and self.is_on_gpu) else np

        if self._belief_array is None:
            self._belief_array = xp.array([b.values for b in self._belief_list])
        return self._belief_array
    

    @property
    def belief_list(self) -> list[Belief]:
        '''
        A list of Belief objects. If the belief set is represented as a matrix of Belief vectors, the list of Belief objects will be generated from it.
        '''
        if self._belief_list is None:
            self._belief_list = [Belief(self.model, belief_vector) for belief_vector in self._belief_array]
        return self._belief_list
    

    def generate_all_successors(self) -> 'BeliefSet':
        '''
        Function to generate the successors beliefs of all the beliefs in the belief set.

        Returns
        -------
        all_successors : BeliefSet
            All successors of all beliefs in the belief set.
        '''
        all_successors = []
        for belief in self.belief_list:
            all_successors.extend(belief.generate_successors())
        return BeliefSet(self.model, all_successors)
    

    def __len__(self) -> int:
        return len(self._belief_list) if self._belief_list is not None else self._belief_array.shape[0]
    

    def to_gpu(self) -> 'BeliefSet':
        '''
        Function returning an equivalent belief set object with the array of values stored on GPU instead of CPU.

        Returns
        -------
        gpu_belief_set : BeliefSet
            A new belief set with array on GPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        gpu_model = self.model.gpu_model

        gpu_belief_set = None
        if self._belief_array is not None:
            gpu_belief_array = cp.array(self._belief_array)
            gpu_belief_set = BeliefSet(gpu_model, gpu_belief_array)
        else:
            gpu_belief_list = [Belief(gpu_model, cp.array(b.values)) for b in self._belief_list]
            gpu_belief_set = BeliefSet(gpu_model, gpu_belief_list)

        return gpu_belief_set
    

    def to_cpu(self) -> 'BeliefSet':
        '''
        Function returning an equivalent belief set object with the array of values stored on CPU instead of GPU.

        Returns
        -------
        cpu_belief_set : BeliefSet
            A new belief set with array on CPU.
        '''
        assert gpu_support, "GPU support is not enabled, unable to execute this function"

        cpu_model = self.model.cpu_model

        cpu_belief_set = None
        if self._belief_array is not None:
            cpu_belief_array = cp.asnumpy(self._belief_array)
            cpu_belief_set = BeliefSet(cpu_model, cpu_belief_array)
        
        else:
            cpu_belief_list = [Belief(cpu_model, cp.asnumpy(b.values)) for b in self._belief_list]
            cpu_belief_set = BeliefSet(cpu_model, cpu_belief_list)

        return cpu_belief_set
    
    
    def plot(self, size:int=15):
        '''
        Function to plot the beliefs in the belief set.
        Note: Only works for 2-state and 3-state believes.

        Parameters
        ----------
        size : int, default=15
            The figure size and general scaling factor
        '''
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3"

        # If on GPU, convert to CPU and plot that one
        if self.is_on_gpu:
            print('[Warning] Value function on GPU, converting to numpy before plotting...')
            cpu_belief_set = self.to_cpu()
            cpu_belief_set.plot(size)
            return

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

        # Set title and ax-label
        ax.set_title('Set of beliefs')
        ax.set_xlabel('Belief space')

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
        plt.title('Set of Beliefs')

        cmap = cm.get_cmap('Blues')
        norm = colors.Normalize(vmin=0, vmax=self.belief_array.shape[0])
        c = range(self.belief_array.shape[0])
        # Do scatter plot
        fig = plotSimplex(self.belief_array, fig=fig, vertexlabels=self.model.state_labels, s=size, c=c, cmap=cmap, norm=norm)

        plt.show()


class BeliefValueMapping:
    '''
    Alternate representation of a value function, particularly for pomdp models.
    It works by adding adding belief and associated value to the object.
    To evaluate this version of the value function the sawtooth algorithm is used (described in Shani G. et al., "A survey of point-based POMDP solvers")
    
    We can also compute the Q value for a particular belief b and action using the qva function.

    Parameters
    ----------
    model: pomdp.Model
        The model on which the value function applies on
    corner_belief_values: ValueFunction
        A general value function to define the value at corner points in belief space (ie: at certainty beliefs, or when beliefs have a probability of 1 for a given state).
        This is usually the solution of the MDP version of the problem.

    Attributes
    ----------
    model: pomdp.Model
    corner_belief_values: ValueFunction
    corner_values: np.ndarray
        Array of |S| shape, having the max value at each state based on the corner_belief_values.
    beliefs: list[Belief]
        List of beliefs points that have been added to the belief value mapping VF.
    values: list[float]
        List of values that have been to the value mapping VF.
    
    '''
    def __init__(self, model, corner_belief_values:ValueFunction) -> None:
        xp = np if not gpu_support else cp.get_array_module(corner_belief_values.alpha_vector_array)

        self.model = model
        self.corner_belief_values = corner_belief_values
        
        self.corner_values = xp.max(corner_belief_values.alpha_vector_array, axis=0)

        self.beliefs = []
        self.values = []

    
    def add(self, b:Belief, v:float) -> None:
        '''
        Function to a belief point and its associated value to the belief value mappings

        Parameters
        ----------
        b: Belief
        v: float
        '''
        self.beliefs.append(b)
        self.values.append(v)


    def evaluate(self, belief:Belief) -> float:
        '''
        Runs the sawtooth algorithm to find the value at a given belief point.

        Parameters
        ----------
        belief: Belief
        '''
        xp = np if not gpu_support else cp.get_array_module(belief.values)

        v0 = xp.dot(belief.values, self.corner_values)

        if len(self.beliefs) > 0:
            belief_array = xp.array([b.values for b in self.beliefs])
            value_array = xp.array(self.values)

            with np.errstate(divide='ignore', invalid='ignore'):
                vb = v0 + ((value_array - xp.dot(belief_array, self.corner_values)) * xp.min(belief.values / belief_array, axis=1))

            return min(v0, xp.min(vb))
        
        return v0
    

    def qva(self, belief:Belief, a:int, gamma:float) -> np.ndarray:
        '''
        Evaluate the value function at a given belief point with a given action a and a given gamma.

        Parameters
        ----------
        belief: Belief
        a: int
        gamma: float
        '''
        xp = np if not gpu_support else cp.get_array_module(belief.values)

        b_probs = xp.einsum('sor,s->o', self.model.reachable_transitional_observation_table[:,a,:,:], belief.values)
        successors_values = xp.array([self.evaluate(belief.update(a,o)) for o in self.model.observations])

        qba = xp.dot(self.model.expected_rewards_table[:,a], belief.values) + gamma * xp.dot(b_probs, successors_values)

        return qba


class SolverHistory:
    '''
    Class to represent the history of a solver for a POMDP solver.
    It has mainly the purpose to have visualizations for the solution, belief set and the whole solving history.
    The visualizations available are:
        - Belief set plot
        - Solution plot
        - Video of value function and belief set evolution over training.

    ...

    Parameters
    ----------
    tracking_level : int
        The tracking level of the solver.
    model : pomdp.Model
        The model the solver has solved.
    gamma : float
        The gamma parameter used by the solver (learning rate).
    eps : float
        The epsilon parameter used by the solver (covergence bound).
    expand_function : str
        The expand (exploration) function used by the solver.
    expand_append : bool
        Whether the expand function appends new belief points to the belief set of reloads it all.
    initial_value_function : ValueFunction
        The initial value function the solver will use to start the solving process.
    initial_belief_set : BeliefSet
        The initial belief set the solver will use to start the solving process.

    Attributes
    ----------
    tracking_level : int
    model : pomdp.Model
    gamma : float
    eps : float
    expand_function : str
    expand_append : bool
    run_ts : datetime
        The time at which the SolverHistory object was instantiated which is assumed to be the start of the solving run.
    expansion_times : list[float]
        A list of recorded times of the expand function.
    backup_times : list[float]
        A list of recorded times of the backup function.
    alpha_vector_counts : list[int]
        A list of recorded alpha vector count making up the value function over the solving process.
    beliefs_counts : list[int]
        A list of recorded belief count making up the belief set over the solving process.
    value_function_changes : list[float]
        A list of recorded value function changes (the maximum changed value between 2 value functions).
    value_functions : list[ValueFunction]
        A list of recorded value functions.
    belief_sets : list[BeliefSet]
        A list of recorded belief sets.
    solution : ValueFunction
    explored_beliefs : BeliefSet
    '''
    def __init__(self,
                 tracking_level:int,
                 model:Model,
                 gamma:float,
                 eps:float,
                 expand_function:str,
                 expand_append:bool,
                 initial_value_function:ValueFunction,
                 initial_belief_set:BeliefSet
                 ):
        
        self.tracking_level = tracking_level
        self.model = model
        self.gamma = gamma
        self.eps = eps
        self.run_ts = datetime.now()
        
        self.expand_function = expand_function
        self.expand_append = expand_append

        # Time tracking
        self.expansion_times = []
        self.backup_times = []
        self.pruning_times = []

        # Value function and belief set sizes tracking
        self.alpha_vector_counts = []
        self.beliefs_counts = []
        self.prune_counts = []

        if self.tracking_level >= 1:
            self.alpha_vector_counts.append(len(initial_value_function))
            self.beliefs_counts.append(len(initial_belief_set))

        # Value function and belief set tracking
        self.belief_sets = []
        self.value_functions = []
        self.value_function_changes = []

        if self.tracking_level >= 2:
            self.belief_sets.append(initial_belief_set)
            self.value_functions.append(initial_value_function)


    @property
    def solution(self) -> ValueFunction:
        '''
        The last value function of the solving process.
        '''
        assert self.tracking_level >= 2, "Tracking level is set too low, increase it to 2 if you want to have value function tracking as well."
        return self.value_functions[-1]
    

    @property
    def explored_beliefs(self) -> BeliefSet:
        '''
        The final set of beliefs explored during the solving.
        '''
        assert self.tracking_level >= 2, "Tracking level is set too low, increase it to 2 if you want to have belief sets tracking as well."
        return self.belief_sets[-1]
    

    def add_expand_step(self,
                        expansion_time:float,
                        belief_set:BeliefSet
                        ) -> None:
        '''
        Function to add an expansion step in the simulation history by the explored belief set the expand function generated.

        Parameters
        ----------
        expansion_time : float
            The time it took to run a step of expansion of the belief set. (Also known as the exploration step.)
        belief_set : BeliefSet
            The belief set used for the Update step of the solving process.
        '''
        if self.tracking_level >= 1:
            self.expansion_times.append(float(expansion_time))
            self.beliefs_counts.append(len(belief_set))

        if self.tracking_level >= 2:
            self.belief_sets.append(belief_set if not belief_set.is_on_gpu else belief_set.to_cpu())


    def add_backup_step(self,
                        backup_time:float,
                        value_function_change:float,
                        value_function:ValueFunction
                        ) -> None:
        '''
        Function to add a backup step in the simulation history by recording the value function the backup function generated.

        Parameters
        ----------
        backup_time : float
            The time it took to run a step of backup of the value function. (Also known as the value function update.)
        value_function_change : float
            The change between the value function of this iteration and of the previous iteration.
        value_function : ValueFunction
            The value function resulting after a step of the solving process.
        '''
        if self.tracking_level >= 1:
            self.backup_times.append(float(backup_time))
            self.alpha_vector_counts.append(len(value_function))
            self.value_function_changes.append(float(value_function_change))

        if self.tracking_level >= 2:
            self.value_functions.append(value_function if not value_function.is_on_gpu else value_function.to_cpu())


    def add_prune_step(self,
                       prune_time:float,
                       alpha_vectors_pruned:int
                       ) -> None:
        '''
        Function to add a prune step in the simulation history by recording the amount of alpha vectors that were pruned by the pruning function and how long it took.

        Parameters
        ----------
        prune_time : float
            The time it took to run the pruning step.
        alpha_vectors_pruned : int
            How many alpha vectors were pruned.
        '''
        if self.tracking_level >= 1:
            self.pruning_times.append(prune_time)
            self.prune_counts.append(alpha_vectors_pruned)


    @property
    def summary(self) -> str:
        '''
        A summary as a string of the information recorded.

        Returns
        -------
        summary_str : str
            The summary of the information.
        '''
        summary_str =  f'Summary of Value Iteration run'
        summary_str += f'\n  - Model: {self.model.state_count} state, {self.model.action_count} action, {self.model.observation_count} observations'
        summary_str += f'\n  - Converged or stopped after {len(self.expansion_times)} expansion steps and {len(self.backup_times)} backup steps.'

        if self.tracking_level >= 1:
            summary_str += f'\n  - Resulting value function has {self.alpha_vector_counts[-1]} alpha vectors.'
            summary_str += f'\n  - Converged in {(sum(self.expansion_times) + sum(self.backup_times)):.4f}s'
            summary_str += f'\n'

            summary_str += f'\n  - Expand function took on average {sum(self.expansion_times) / len(self.expansion_times):.4f}s '
            if self.expand_append:
                summary_str += f'and yielded on average {sum(np.diff(self.beliefs_counts)) / len(self.beliefs_counts[1:]):.2f} beliefs per iteration.'
            else:
                summary_str += f'and yielded on average {sum(self.beliefs_counts[1:]) / len(self.beliefs_counts[1:]):.2f} beliefs per iteration.'
            summary_str += f' ({np.sum(np.divide(self.expansion_times, self.beliefs_counts[1:])) / len(self.expansion_times):.4f}s/it/belief)'
            
            summary_str += f'\n  - Backup function took on average {sum(self.backup_times) /len(self.backup_times):.4f}s '
            summary_str += f'and yielded on average value functions of size {sum(self.alpha_vector_counts[1:]) / len(self.alpha_vector_counts[1:]):.2f} per iteration.'
            summary_str += f' ({np.sum(np.divide(self.backup_times, self.alpha_vector_counts[1:])) / len(self.backup_times):.4f}s/it/alpha)'

            summary_str += f'\n  - Pruning function took on average {sum(self.pruning_times) /len(self.pruning_times):.4f}s '
            summary_str += f'and yielded on average prunings of {sum(self.prune_counts) / len(self.prune_counts):.2f} alpha vectors per iteration.'
        
        return summary_str
    

    def plot_belief_set(self, size:int=15) -> None:
        '''
        Function to plot the last belief set explored during the solving process.

        Parameters
        ----------
        size : int, default=15
            The scale of the plot.
        '''
        self.explored_beliefs.plot(size=size)


    def plot_solution(self, size:int=5, plot_belief:bool=True) -> None:
        '''
        Function to plot the value function of the solution.
        Note: only works for 2 and 3 states models

        Parameters
        ----------
        size : int, default=5
            The figure size and general scaling factor.
        plot_belief : bool, default=True
            Whether to plot the belief set along with the value function.
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

        Parameters
        ----------
        custom_name : str, optional
            The name the video will be saved with.
        compare_with : PBVI or ValueFunction or list, default=[]
            Value functions or other solvers to plot against the current solver's history.
        graph_names : list[str], default=[]
            Names of the graphs for the legend of which graph is being plot.
        fps : int, default=10
            How many frames per second should the saved video have.
        '''
        assert self.tracking_level >= 2, "Tracking level is set too low, increase it to 2 if you want to have value function and belief sets tracking as well."
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3" # TODO Make support for gird videos

        if self.model.state_count == 2:
            self._save_history_video_2D(custom_name, compare_with, copy.copy(graph_names), fps)
        elif self.model.state_count == 3:
            raise Exception('Not implemented...')


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

        def plot_on_ax(history:Union[ValueFunction,'SolverHistory'], frame_i:int, ax, line_type:str):
            if isinstance(history, ValueFunction):
                value_function = history
            else:
                frame_i = frame_i if frame_i < len(history.value_functions) else (len(history.value_functions) - 1)
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

            # Axes labels
            ax1.set_ylabel('V(b)')
            ax2.set_xlabel('Belief space')

            self_frame_i = frame_i if frame_i < len(self.value_functions) else (len(self.value_functions) - 1)

            # Subtitle
            ax1.set_title(title + f'(Frame {frame_i})')

            # Color legend
            leg1 = ax1.legend(proxy, self.model.action_labels, loc='upper center')
            ax1.set_xticks(ticks, x_ticks)
            ax1.add_artist(leg1)

            # Line legend
            lines = []
            point = self.value_functions[self_frame_i].alpha_vector_array[0,0]
            for l in line_types:
                lines.append(Line2D([0,point],[0,point],linestyle=l))
            ax1.legend(lines, graph_names, loc='lower center')

            # Alpha vector plotting
            for history, line_type in zip(solver_histories, line_types):
                plot_on_ax(history, frame_i, ax1, line_type)

            # Belief plotting
            beliefs_x = self.belief_sets[frame_i if frame_i < len(self.belief_sets) else -1].belief_array[:,1]
            ax2.scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c='red')
            ax2.get_yaxis().set_visible(False)
            ax2.axhline(0, color='black')

        max_steps = max([len(history.value_functions) for history in solver_histories if not isinstance(history,ValueFunction)])
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
    Parameters
    ----------
    gamma : float, default=0.9
        The learning rate, used to control how fast the value function will change after the each iterations.
    eps : float, default=0.001
        The treshold for convergence. If the max change between value function is lower that eps, the algorithm is considered to have converged.
    expand_function : str, default='ssea'
        The type of expand strategy to use to expand the belief set.
    expand_function_params : dict, optional
        Other required parameters to be sent to the expand function.

    Attributes
    ----------
    gamma : float
    eps : float
    expand_function : str
    expand_function_params : dict
    '''
    def __init__(self,
                 gamma:float=0.9,
                 eps:float=0.001,
                 expand_function:str='ssea',
                 **expand_function_params):
        self.gamma = gamma
        self.eps = eps
        self.expand_function = expand_function
        self.expand_function_params = expand_function_params


    def test_n_simulations(self, model:Model, value_function:ValueFunction, n:int=1000, horizon:int=300, print_progress:bool=False):
        '''
        Function that tests a value function with n simulations. It returns the start states, the amount of steps in which the simulation reached an end state, the rewards received and the discounted rewards received.

        Parameters
        ----------
        model : pomdp.Model
            The model on which to run the simulations.
        value_function : ValueFunction
            The value function that will be evaluated.
        n : int, default=1000
            The amount of simulations to run.
        horizon : int, default=300
            The maximum amount of steps the simulation can run for.
        print_progress : bool, default=False
            Whether to display a progress bar of how many simulation steps have been run so far. 
        '''
        # GPU support
        xp = np if not value_function.is_on_gpu else cp
        model = model.cpu_model if not value_function.is_on_gpu else model.gpu_model

        # Genetion of an array of n beliefs
        initial_beliefs = xp.repeat(Belief(model).values[None,:], n, axis=0)

        # Generating n initial positions
        start_states = xp.random.choice(model.states, size=n, p=model.start_probabilities)
        
        # Belief and state arrays
        beliefs = initial_beliefs
        new_beliefs = None
        states = start_states
        next_states = None

        # Tracking what simulations are done
        sim_is_done = xp.zeros(n, dtype=bool)
        done_at_step = xp.full(n, -1)

        # Speedup item
        simulations = xp.arange(n)
        flatten_offset = (simulations[:,None] * model.state_count)
        flat_shape = (n, (model.state_count * model.reachable_state_count))

        # 2D bincount for belief set update
        def bincount2D_vectorized(a, w):    
            a_offs = a + flatten_offset
            return xp.bincount(a_offs.ravel(), weights=w.ravel(), minlength=a.shape[0]*model.state_count).reshape(-1,model.state_count)

        # Results
        discount = self.gamma
        rewards = []
        discounted_rewards = []
        
        iterator = trange(horizon) if print_progress else range(horizon)
        for i in iterator:
            # Retrieving the top vectors according to the value function
            best_vectors = xp.argmax(xp.matmul(beliefs, value_function.alpha_vector_array.T), axis=1)

            # Retrieving the actions associated with the vectors chosen
            best_actions = value_function.actions[best_vectors]

            # Get each reachable next states for each action
            reachable_state_per_actions = model.reachable_states[:, best_actions, :]

            # Gathering new states based on the transition function and the chosen actions
            next_state_potentials = reachable_state_per_actions[states, simulations]
            if model.reachable_state_count == 1:
                next_states = next_state_potentials[:,0]
            else:
                potential_probabilities = model.reachable_probabilities[states[:,None], best_actions[:,None]][:,0,:]
                chosen_indices = xp.apply_along_axis(lambda x: xp.random.choice(len(x), size=1, p=x), axis=1, arr=potential_probabilities)
                next_states = next_state_potentials[chosen_indices][:,0,0]

            # Making observations based on the states landed in and the action that was taken
            observation_probabilities = model.observation_table[next_states[:,None], best_actions[:,None]][:,0,:]
            observations = xp.sum(xp.random.random(n)[:,None] > xp.cumsum(observation_probabilities[:,:-1], axis=1), axis=1)

            # Belief set update
            reachable_probabilities = (model.reachable_transitional_observation_table[:,best_actions,observations,:] * beliefs.T[:,:,None])
            new_beliefs = bincount2D_vectorized(a=reachable_state_per_actions.swapaxes(0,1).reshape(flat_shape),
                                                w=reachable_probabilities.swapaxes(0,1).reshape(flat_shape))

            new_beliefs /= xp.sum(new_beliefs, axis=1)[:,None]

            # Rewards computation
            step_rewards = xp.array([model.immediate_reward_function(s,a,s_p,o) for s,a,s_p,o in zip(states, best_actions, next_states, observations)])
            rewards.append(xp.where(~sim_is_done, step_rewards, 0))
            discounted_rewards.append(xp.where(~sim_is_done, step_rewards * discount, 0))

            # Checking for done condition
            are_done = xp.isin(next_states, xp.array(model.end_states))
            done_at_step[sim_is_done ^ are_done] = i+1
            sim_is_done |= are_done

            # Update iterator postfix
            if print_progress:
                iterator.set_postfix({'done': xp.sum(sim_is_done)})

            # Replacing old with new
            states = next_states
            beliefs = new_beliefs
            discount *= self.gamma

            # Early stopping
            if xp.all(sim_is_done):
                break

        return start_states, done_at_step, rewards, discounted_rewards


    def backup(self,
               model:Model,
               belief_set:BeliefSet,
               value_function:ValueFunction,
               append:bool=False,
               belief_dominance_prune:bool=True
               ) -> ValueFunction:
        '''
        This function has purpose to update the set of alpha vectors. It does so in 3 steps:
        1. It creates projections from each alpha vector for each possible action and each possible observation
        2. It collapses this set of generated alpha vectors by taking the weighted sum of the alpha vectors weighted by the observation probability and this for each action and for each belief.
        3. Then it further collapses the set to take the best alpha vector and action per belief
        In the end we have a set of alpha vectors as large as the amount of beliefs.

        The alpha vectors are also pruned to avoid duplicates and remove dominated ones.

        Parameters
        ----------
        model : pomdp.Model
            The model on which to run the backup method on.
        belief_set : BeliefSet
            The belief set to use to generate the new alpha vectors with.
        value_function : ValueFunction
            The alpha vectors to generate the new set from.
        append : bool, default=False
            Whether to append the new alpha vectors generated to the old alpha vectors before pruning.
        belief_dominance_prune : bool, default=True
            Whether, before returning the new value function, checks what alpha vectors have a supperior 
            
        Returns
        -------
        new_alpha_set : ValueFunction
            A list of updated alpha vectors.
        '''
        # Get numpy corresponding to the arrays
        xp = np if not gpu_support else cp.get_array_module(value_function.alpha_vector_array)

        # Step 1
        vector_array = value_function.alpha_vector_array
        vectors_array_reachable_states = vector_array[xp.arange(vector_array.shape[0])[:,None,None,None], model.reachable_states[None,:,:,:]]
        
        gamma_a_o_t = self.gamma * xp.einsum('saor,vsar->aovs', model.reachable_transitional_observation_table, vectors_array_reachable_states)

        # Step 2
        belief_array = belief_set.belief_array # bs
        best_alpha_ind = xp.argmax(xp.tensordot(belief_array, gamma_a_o_t, (1,3)), axis=3) # argmax(bs,aovs->baov) -> bao

        best_alphas_per_o = gamma_a_o_t[model.actions[None,:,None,None], model.observations[None,None,:,None], best_alpha_ind[:,:,:,None], model.states[None,None,None,:]] # baos

        alpha_a = model.expected_rewards_table.T + xp.sum(best_alphas_per_o, axis=2) # as + bas

        # Step 3
        best_actions = xp.argmax(xp.einsum('bas,bs->ba', alpha_a, belief_array), axis=1)
        alpha_vectors = xp.take_along_axis(alpha_a, best_actions[:,None,None],axis=1)[:,0,:]

        # Belief domination
        if belief_dominance_prune:
            best_value_per_belief = xp.sum((belief_array * alpha_vectors), axis=1)
            old_best_value_per_belief = xp.max(xp.matmul(belief_array, vector_array.T), axis=1)
            dominating_vectors = best_value_per_belief > old_best_value_per_belief

            best_actions = best_actions[dominating_vectors]
            alpha_vectors = alpha_vectors[dominating_vectors]

        # Creation of value function
        new_value_function = ValueFunction(model, alpha_vectors, best_actions)

        # Union with previous value function
        if append:
            new_value_function.extend(value_function)
                
        return new_value_function
    

    def expand_ra(self, model:Model, belief_set:BeliefSet, max_generation:int=10) -> BeliefSet:
        '''
        This expansion technique relies only randomness and will generate at most 'max_generation' beliefs.

        Parameters
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.
        '''
        xp = np if not gpu_support else cp.get_array_module(belief_set.belief_array)

        # How many new beliefs to add
        generation_count = min(belief_set.belief_array.shape[0], max_generation)

        # Generation of the new beliefs at random
        new_beliefs = xp.random.random((generation_count, model.state_count))
        new_beliefs /= xp.sum(new_beliefs, axis=1)[:,None]

        # Combining with the initial belief set
        new_belief_set = np.vstack((belief_set.belief_array, new_beliefs))

        return BeliefSet(model, new_belief_set)

    
    def expand_ssra(self, model:Model, belief_set:BeliefSet, max_generation:int=10) -> BeliefSet:
        '''
        Stochastic Simulation with Random Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state (weighted by the belief) and taking a random action leading to a state s_p and a observation o.
        From this action a and observation o we can update our belief.

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.

        Returns
        -------
        belief_set_new : BeliefSet
            Union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        xp = np if not gpu_support else cp.get_array_module(belief_set.belief_array)

        old_shape = belief_set.belief_array.shape
        to_generate = min(max_generation, old_shape[0])

        new_belief_array = xp.empty((old_shape[0] + to_generate, old_shape[1]))
        new_belief_array[:old_shape[0]] = belief_set.belief_array

        # Random previous beliefs
        rand_ind = np.random.choice(np.arange(old_shape[0]), to_generate, replace=False)

        for i, belief_vector in enumerate(belief_set.belief_array[rand_ind]):
            b = Belief(model, belief_vector)
            s = b.random_state()
            a = random.choice(model.actions)
            s_p = model.transition(s, a)
            o = model.observe(s_p, a)
            b_new = b.update(a, o)
            
            new_belief_array[i+old_shape[0]] = b_new.values
            
        return BeliefSet(model, new_belief_array)
    

    def expand_ssga(self, model:Model, belief_set:BeliefSet, value_function:ValueFunction, epsilon:float=0.1, max_generation:int=10) -> BeliefSet:
        '''
        Stochastic Simulation with Greedy Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state s (weighted by the belief),
         then taking the best action a based on the belief with probability 'epsilon'.
        These lead to a new state s_p and a observation o.
        From this action a and observation o we can update our belief. 

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        value_function : ValueFunction
            Used to find the best action knowing the belief.
        eps : float
            Parameter tuning how often we take a greedy approach and how often we move randomly.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.

        Returns
        -------
        belief_set_new : BeliefSet
            Union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        xp = np if not gpu_support else cp.get_array_module(belief_set.belief_array)

        old_shape = belief_set.belief_array.shape
        to_generate = min(max_generation, old_shape[0])

        new_belief_array = xp.empty((old_shape[0] + to_generate, old_shape[1]))
        new_belief_array[:old_shape[0]] = belief_set.belief_array

        # Random previous beliefs
        rand_ind = np.random.choice(np.arange(old_shape[0]), to_generate, replace=False)

        for i, belief_vector in enumerate(belief_set.belief_array[rand_ind]):
            b = Belief(model, belief_vector)
            s = b.random_state()
            
            if random.random() < epsilon:
                a = random.choice(model.actions)
            else:
                best_alpha_index = xp.argmax(xp.dot(value_function.alpha_vector_array, b.values))
                a = value_function.actions[best_alpha_index]
            
            s_p = model.transition(s, a)
            o = model.observe(s_p, a)
            b_new = b.update(a, o)
            
            new_belief_array[i+old_shape[0]] = b_new.values
            
        return BeliefSet(model, new_belief_array)
    

    def expand_ssea(self, model:Model, belief_set:BeliefSet, max_generation:int=10) -> BeliefSet:
        '''
        Stochastic Simulation with Exploratory Action.
        Simulates running steps forward for each possible action knowing we are a state s, chosen randomly with according to the belief probability.
        These lead to a new state s_p and a observation o for each action.
        From all these and observation o we can generate updated beliefs. 
        Then it takes the belief that is furthest away from other beliefs, meaning it explores the most the belief space.

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.

        Returns
        -------
        belief_set_new : BeliefSet
            Union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        xp = np if not gpu_support else cp.get_array_module(belief_set.belief_array)

        old_shape = belief_set.belief_array.shape
        to_generate = min(max_generation, old_shape[0])

        # Generation of successors
        successor_beliefs = xp.array([[[b.update(a,o).values for o in model.observations] for a in model.actions] for b in belief_set.belief_list])
        
        # Compute the distances between each pair and of successor are source beliefs
        diff = (belief_set.belief_array[:, None,None,None, :] - successor_beliefs)
        dist = xp.sqrt(xp.einsum('bnaos,bnaos->bnao', diff, diff))

        # Taking the min distance for each belief
        belief_min_dists = xp.min(dist,axis=0)

        # Taking the max distanced successors
        b_star, a_star, o_star = xp.unravel_index(xp.argsort(belief_min_dists, axis=None)[::-1][:to_generate], successor_beliefs.shape[:-1])

        # Selecting successor beliefs
        selected_beliefs = successor_beliefs[b_star[:,None], a_star[:,None], o_star[:,None], model.states[None,:]]

        # Unioning with previous beliefs
        new_belief_array = xp.vstack((belief_set.belief_array, selected_beliefs))

        return BeliefSet(model, new_belief_array)
    

    def expand_ger(self, model:Model, belief_set:BeliefSet, value_function:ValueFunction, max_generation:int=10) -> BeliefSet:
        '''
        Greedy Error Reduction.
        It attempts to choose the believes that will maximize the improvement of the value function by minimizing the error.
        The error is computed by the sum of the change between two beliefs and their two corresponding alpha vectors.

        Parameters
        ----------
        model : pomdp.Model
            The POMDP model on which to expand the belief set on.
        belief_set : BeliefSet
            List of beliefs to expand on.
        value_function : ValueFunction
            Used to find the best action knowing the belief.
        max_generation : int, default=10
            The max amount of beliefs that can be added to the belief set at once.

        Returns
        -------
        belief_set_new : BeliefSet
            Union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        xp = np if not gpu_support else cp.get_array_module(belief_set.belief_array)

        old_shape = belief_set.belief_array.shape
        to_generate = min(max_generation, old_shape[0])

        new_belief_array = xp.empty((old_shape[0] + to_generate, old_shape[1]))
        new_belief_array[:old_shape[0]] = belief_set.belief_array

        # Finding the min and max rewards for computation of the epsilon
        r_min = model._min_reward / (1 - self.gamma)
        r_max = model._max_reward / (1 - self.gamma)

        # Generation of all potential successor beliefs
        successor_beliefs = xp.array([[[b.update(a,o).values for o in model.observations] for a in model.actions] for b in belief_set.belief_list])
        
        # Finding the alphas associated with each previous beliefs
        best_alpha = xp.argmax(xp.dot(belief_set.belief_array, value_function.alpha_vector_array.T), axis = 1)
        b_alphas = value_function.alpha_vector_array[best_alpha]

        # Difference between beliefs and their successors
        b_diffs = successor_beliefs - belief_set.belief_array[:,None,None,:]

        # Computing a 'next' alpha vector made of the max and min
        alphas_p = xp.where(b_diffs >= 0, r_max, r_min)

        # Difference between alpha vectors and their successors alpha vector
        alphas_diffs = alphas_p - b_alphas[:,None,None,:]

        # Computing epsilon for all successor beliefs
        eps = xp.einsum('baos,baos->bao', alphas_diffs, b_diffs)

        # Computing the probability of the b and doing action a and receiving observation o
        bao_probs = xp.einsum('bs,saor->bao', belief_set.belief_array, model.reachable_transitional_observation_table)

        # Taking the sumproduct of the probs with the epsilons
        res = xp.einsum('bao,bao->ba', bao_probs, eps)

        # Picking the correct amount of initial beliefs and ideal actions
        b_stars, a_stars = xp.unravel_index(xp.argsort(res, axis=None)[::-1][:to_generate], res.shape)

        # And picking the ideal observations
        o_star = xp.argmax(bao_probs[b_stars[:,None], a_stars[:,None], model.observations[None,:]] * eps[b_stars[:,None], a_stars[:,None], model.observations[None,:]], axis=1)

        # Selecting the successor beliefs
        selected_beliefs = successor_beliefs[b_stars[:,None], a_stars[:,None], o_star[:,None], model.states[None,:]]

        # Unioning with previous beliefs
        new_belief_array = xp.vstack((belief_set.belief_array, selected_beliefs))

        return BeliefSet(model, new_belief_array)


    def expand_hsvi(self,
                    model:Model,
                    b:Belief,
                    value_function:ValueFunction,
                    upper_bound_belief_value_map:BeliefValueMapping,
                    conv_term:Union[float,None]=None,
                    max_generation:int=10
                    ) -> BeliefSet:
        '''
        The expand function of the  Heruistic search value iteration technique.
        It is a redursive function attempting to minimize the bound between the upper and lower estimations of the value function.

        It is developped by Smith T. and Simmons R. and described in the paper "Heuristic Search Value Iteration for POMDPs"
        '''
        xp = np if not gpu_support else cp.get_array_module(b.values)

        if conv_term is None:
            conv_term = self.eps

        # Update convergence term
        conv_term /= self.gamma

        # Find best a based on upper bound v
        best_a = int(xp.argmax(xp.array([upper_bound_belief_value_map.qva(b, a, gamma=self.gamma) for a in model.actions])))

        # Choose o that max gap between bounds
        b_probs = xp.einsum('sor,s->o', model.reachable_transitional_observation_table[:,best_a,:,:], b.values)
        successors_ba = [b.update(best_a, o) for o in model.observations]

        upper_v_ba = xp.array([upper_bound_belief_value_map.evaluate(bao) for bao in successors_ba])
        lower_v_ba = xp.max((np.matmul(value_function.alpha_vector_array, xp.array([succ_b.values for succ_b in successors_ba]).T)), axis=0)

        best_o = int(xp.argmax((b_probs * (upper_v_ba - lower_v_ba)) - conv_term))

        # Chosen b
        next_b = successors_ba[best_o]

        # check the bounds
        bounds_split = upper_v_ba[best_o] - lower_v_ba[best_o]

        if bounds_split < conv_term or max_generation <= 0:
            return BeliefSet(model, [next_b])
        
        # Go one step deeper in the recursion
        b_set = self.expand_hsvi(model=model,
                                 b=next_b,
                                 value_function=value_function,
                                 upper_bound_belief_value_map=upper_bound_belief_value_map,
                                 conv_term=conv_term,
                                 max_generation=max_generation-1)
        
        # Append the nex belief of this iteration to the deeper beliefs
        new_belief_list = b_set.belief_list
        new_belief_list.append(next_b)

        # Add the belief point and associated value to the belief-value mapping
        upper_bound_belief_value_map.add(next_b, max([upper_bound_belief_value_map.qva(next_b, a, gamma=self.gamma) for a in model.actions]))

        return BeliefSet(model, new_belief_list)


    def expand(self, model:Model, belief_set:BeliefSet, max_generation:int, **function_specific_parameters) -> BeliefSet:
        '''
        Central method to call one of the functions for a particular expansion strategy:
            - Random selction (RA)
            - Stochastic Simulation with Random Action (ssra)
            - Stochastic Simulation with Greedy Action (ssga)
            - Stochastic Simulation with Exploratory Action (ssea)
            - Greedy Error Reduction (ger)
            - Heuristic Search Value Iteration (hsvi)
                
        Parameters
        ----------
        model : pomdp.Model
            The model on which to run the belief expansion on.
        belief_set : BeliefSet
            The set of beliefs to expand.
        max_generation : int
            The max amount of beliefs that can be added to the belief set at once.
        function_specific_parameters
            Potential additional parameters necessary for the specific expand function.

        Returns
        -------
        belief_set_new : BeliefSet
            The belief set the expansion function returns. 
        '''
        if self.expand_function in 'expand_ssra':
            return self.expand_ra(model=model, belief_set=belief_set, max_generation=max_generation)

        elif self.expand_function in 'expand_ssra':
            return self.expand_ssra(model=model, belief_set=belief_set, max_generation=max_generation)
        
        elif self.expand_function in 'expand_ssga':
            args = {arg: function_specific_parameters[arg] for arg in ['value_function', 'epsilon'] if arg in function_specific_parameters}
            return self.expand_ssga(model=model, belief_set=belief_set, max_generation=max_generation, **args)
        
        elif self.expand_function in 'expand_ssea':
            return self.expand_ssea(model=model, belief_set=belief_set, max_generation=max_generation)
        
        elif self.expand_function in 'expand_ger':
            args = {arg: function_specific_parameters[arg] for arg in ['value_function'] if arg in function_specific_parameters}
            return self.expand_ger(model=model, belief_set=belief_set, max_generation=max_generation, **args)
        
        elif self.expand_function in 'expand_hsvi':
            args = {arg: function_specific_parameters[arg] for arg in ['value_function', 'mdp_policy'] if arg in function_specific_parameters}
            upper_bound = self._upper_bound if hasattr(self, '_upper_bound') else BeliefValueMapping(model, args['mdp_policy'])
            return self.expand_hsvi(model=model, 
                                    b=Belief(model),
                                    value_function=args['value_function'],
                                    upper_bound_belief_value_map=upper_bound)
        else:
            raise Exception('Not implemented')

        return []


    def compute_change(self, value_function:ValueFunction, new_value_function:ValueFunction, belief_set:BeliefSet) -> float:
        '''
        Function to compute whether the change between two value functions can be considered as having converged based on the eps parameter of the Solver.
        It check for each belief, the maximum value and take the max change between believe's value functions.
        If this max change is lower than eps * (gamma / (1 - gamma)).

        Parameters
        ----------
        value_function : ValueFunction
            The first value function to compare.
        new_value_function : ValueFunction
            The second value function to compare.
        belief_set : BeliefSet
            The set of believes to check the values on to compute the max change on.

        Returns
        -------
        max_change : float
            The maximum change between value functions at belief points.
        '''
        # Get numpy corresponding to the arrays
        xp = np if not gpu_support else cp.get_array_module(value_function.alpha_vector_array)

        # Computing Delta for each beliefs
        max_val_per_belief = xp.max(xp.matmul(belief_set.belief_array, value_function.alpha_vector_array.T), axis=1)
        new_max_val_per_belief = xp.max(xp.matmul(belief_set.belief_array, new_value_function.alpha_vector_array.T), axis=1)
        max_change = xp.max(xp.abs(new_max_val_per_belief - max_val_per_belief))

        return max_change


    # def solve(self,
    #           model:Model,
    #           expansions:int,
    #           horizon:int,
    #           mdp_policy:ValueFunction,
    #           initial_belief:Union[Belief,None]=None,
    #           initial_value_function:Union[ValueFunction,None]=None,
    #           prune_level:int=1,
    #           prune_interval:int=10,
    #           belief_memory_depth:int=10,
    #           use_gpu:bool=False,
    #           history_tracking_level:int=1,
    #           print_progress:bool=True
    #           ) -> tuple[ValueFunction, SolverHistory]:
    def solve(self,
              model:Model,
              expansions:int,
              horizon:int,
              max_belief_growth:int=10,
              initial_belief:Union[BeliefSet, Belief, None]=None,
              initial_value_function:Union[ValueFunction,None]=None,
              prune_level:int=1,
              prune_interval:int=10,
              use_gpu:bool=False,
              history_tracking_level:int=1,
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
            - hsvi: Heuristic Search Value Iteration. Extra param: mdp_policy (ValueFunction) 

        Parameters
        ----------
        model : pomdp.Model
            The model to solve.
        expansions : int
            How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
        horizon : int
            How many times the alpha vector set must be updated every time the belief set is expanded.
        max_belief_growth : int, default=10
            How many beliefs can be added at every expansion step to the belief set.
        initial_belief : BeliefSet or Belief, optional
            An initial list of beliefs to start with.
        initial_value_function : ValueFunction, optional
            An initial value function to start the solving process with.
        prune_level : int, default=1
            Parameter to prune the value function further before the expand function.
        prune_interval : int, default=10
            How often to prune the value function. It is counted in number of backup iterations.
        use_gpu : bool, default=False
            Whether to use the GPU with cupy array to accelerate solving.
        history_tracking_level : int, default=1
            How thorough the tracking of the solving process should be. (0: Nothing; 1: Times and sizes of belief sets and value function; 2: The actual value functions and beliefs sets)
        print_progress : bool, default=True
            Whether or not to print out the progress of the value iteration process.

        Returns
        -------
        value_function : ValueFunction
            The alpha vectors approximating the value function.
        solver_history : SolverHistory
            The history of the solving process with some plotting options.
        '''
        # numpy or cupy module
        xp = np

        # If GPU usage
        if use_gpu:
            assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."
            model = model.gpu_model

            # Replace numpy module by cupy for computations
            xp = cp

        # Initial belief
        if initial_belief is None:
            belief_set = BeliefSet(model, [Belief(model)])
        elif isinstance(initial_belief, BeliefSet):
            belief_set = initial_belief.to_gpu() if use_gpu else initial_belief 
        else:
            initial_belief = Belief(model, xp.array(initial_belief.values))
            belief_set = BeliefSet(model, [initial_belief])
        
        # Initial value function
        if initial_value_function is None:
            value_function = ValueFunction(model, model.expected_rewards_table.T, model.actions)
        else:
            value_function = initial_value_function.to_gpu() if use_gpu else initial_value_function

        # Convergence check boundary
        max_allowed_change = self.eps * (self.gamma / (1-self.gamma))

        # History tracking
        solver_history = SolverHistory(tracking_level=history_tracking_level,
                                       model=model,
                                       gamma=self.gamma,
                                       eps=self.eps,
                                       expand_function=self.expand_function,
                                       expand_append=True,
                                       initial_value_function=value_function,
                                       initial_belief_set=belief_set)

        # Loop
        iteration = 0
        expand_value_function = value_function
        old_value_function = value_function

        try:
            for expansion_i in range(expansions) if not print_progress else trange(expansions, desc='Expansions'):

                # 1: Expand belief set
                start_ts = datetime.now()

                belief_set = self.expand(model=model,
                                        belief_set=belief_set,
                                        value_function=value_function,
                                        max_generation=max_belief_growth,
                                        **self.expand_function_params)

                expand_time = (datetime.now() - start_ts).total_seconds()
                solver_history.add_expand_step(expansion_time=expand_time, belief_set=belief_set)

                # 2: Backup, update value function (alpha vector set)
                for _ in range(horizon) if (not print_progress or horizon <= 1) else trange(horizon, desc=f'Backups {expansion_i}'):
                    start_ts = datetime.now()

                    # Backup step
                    value_function = self.backup(model, belief_set, value_function, belief_dominance_prune=False)
                    backup_time = (datetime.now() - start_ts).total_seconds()

                    # Additional pruning
                    if (iteration % prune_interval) == 0 and iteration > 0:
                        start_ts = datetime.now()
                        vf_len = len(value_function)

                        value_function.prune(prune_level)

                        prune_time = (datetime.now() - start_ts).total_seconds()
                        alpha_vectors_pruned = len(value_function) - vf_len
                        solver_history.add_prune_step(prune_time, alpha_vectors_pruned)
                    
                    # Compute the change between value functions
                    max_change = self.compute_change(value_function, old_value_function, belief_set)

                    # History tracking
                    solver_history.add_backup_step(backup_time, max_change, value_function)

                    # Convergence check
                    if max_change < max_allowed_change:
                        break

                    old_value_function = value_function

                    # Update iteration counter
                    iteration += 1

                # Compute change with old expansion value function
                expand_max_change = self.compute_change(expand_value_function, value_function, belief_set)

                if expand_max_change < max_allowed_change:
                    print('Converged!')
                    break

                expand_value_function = value_function
        except MemoryError as e:
            print(f'Memory full: {e}')
            print('Returning value function and history as is...')

        # Final pruning
        start_ts = datetime.now()
        vf_len = len(value_function)

        value_function.prune(prune_level)

        prune_time = (datetime.now() - start_ts).total_seconds()
        alpha_vectors_pruned = len(value_function) - vf_len
        solver_history.add_prune_step(prune_time, alpha_vectors_pruned)

        return value_function, solver_history


class FSVI_Solver(PBVI_Solver):
    '''
    Solver to solve a POMDP problem based on the Forward Search Value Iteration principle.
    It has been built based on the paper of G. Shani, R. I. Brafman, and S. I. Shimony: 'Forward Search Value Iteration for POMDPS'.
    It works by utilizing the optimal MDP policy to generate paths to explore that lead to series of Beliefs
    that can then be used by the Backup function to update the value function.
    
    ...
    Parameters
    ----------
    gamma : float, default=0.9
        The learning rate, used to control how fast the value function will change after the each iterations.
    eps : float, default=0.001
        The treshold for convergence. If the max change between value function is lower that eps, the algorithm is considered to have converged.

    Attributes
    ----------
    gamma : float
    eps : float
    '''
    def __init__(self, gamma:float=0.9, eps:float=0.001):
        self.gamma = gamma
        self.eps = eps

        self._belief_memory = {}


    def MDPExplore(self,
                   model:Model,
                   b:Belief,
                   s:int,
                   mdp_policy:ValueFunction,
                   depth:int,
                   horizon:int,
                   sequence_string:str='',
                   belief_memory_depth:int=10
                   ) -> BeliefSet:
        '''
        Function implementing the exploration process using the MDP policy in order to generate a sequence of Beliefs.
        It is a recursive function that is started by a initial state 's' and using the MDP policy, chooses the best action to take.
        Following this, a random next state 's_p' is being sampled from the transition probabilities and a random observation 'o' based on the observation probabilities.
        Then the given belief is updated using the chosen action and the observation received and the updated belief is added to the sequence.
        Once the state is a goal state, the recursion is done and the belief sequence is returned.

        Parameters
        ----------
        model : pomdp.Model
            The model in which the exploration process will happen.
        b : Belief
            A belief to be added to the returned belief sequence and updated for the next step of the recursion.
        s : int
            The state that starts the exploration sequence and based on which an action will be chosen.
        mdp_policy : ValueFunction
            The mdp policy used to choose the action from with the given state 's'.
        depth : int
            The current recursion depth.
        horizon : int
            The maximum recursion depth that can be reached before the generated belief sequence is returned.
        sequence_string : str, default=''
            The sequence of previously explored actions and observations.
        belief_memory_depth : int, default=10
            How deep the belief memory should be. It is used to speedup the MDP expansion process by caching previously explored beliefs based on the sequence that led it there.

        Returns
        -------
        belief_set : BeliefSet
            A new sequence of beliefs.
        '''
        xp = np if not gpu_support else cp.get_array_module(b.values)
        belief_list = [b]

        if depth >= horizon:
            log('Horizon reached before goal...')
        elif s not in model.end_states:
            # Choose action based on mdp value function
            a_star = xp.argmax(mdp_policy.alpha_vector_array[:,s])

            # Pick a random next state (weighted by transition probabilities)
            s_p = model.transition(s, a_star)

            # Pick a random observation weighted by observation probabilities in state s_p and after having done action a_star
            o = model.observe(s_p, a_star)

            # Update sequence string
            sequence_string += ('-' if depth > 0 else '') + f'{a_star},{o}'

            # If we are within the memory limits and the sequence is already in the memory, we retrieve it
            if depth < belief_memory_depth and sequence_string in self._belief_memory:
                b_p = self._belief_memory[sequence_string]
            
            else:
                # Generate a new belief based on a_star and o
                b_p = b.update(a_star, o)
                
                # Add it to the memory if we are in the depth
                if depth < belief_memory_depth:
                    self._belief_memory[sequence_string] = b_p

            # Recursive call to go closer to goal
            b_set = self.MDPExplore(model, b_p, s_p, mdp_policy, depth+1, horizon, sequence_string, belief_memory_depth)
            belief_list.extend(b_set.belief_list)
        
        return BeliefSet(model, belief_list)


    def solve(self,
              model:Model,
              expansions:int,
              horizon:int,
              mdp_policy:ValueFunction,
              initial_belief:Union[Belief,None]=None,
              initial_value_function:Union[ValueFunction,None]=None,
              prune_level:int=1,
              prune_interval:int=10,
              belief_memory_depth:int=10,
              use_gpu:bool=False,
              history_tracking_level:int=1,
              print_progress:bool=True
              ) -> tuple[ValueFunction, SolverHistory]:
        '''
        The main loop for the forward search value iteration process. The amount of 'expansions' will determine how many the exploration process will run (generating every time a sequence of beliefs).
        Then the 'horizon' parameter determines how deep the MDP exploration can run for. For example, is it set at 10 but it takes 15 steps to reach the end goal, the MDP exploration process will exit early.
        It should therefore be set to a bit higher than the maximum amount steps required to reach a goal state from any other state. It is mainly used as a safeguard to avoid infinite looping.

        Parameters
        ----------
        model : pomdp.Model
            The model to solve.
        expansions : int
            How many times the MDP exploration process will run.
        horizon : int
            How many deep the MDP exploration process can run for.
        mdp_policy : ValueFunction
            The policy that will be used to choose actions from states during the exploration process.
        initial_belief : Belief, optional
            An initial belief that will replace the default initial belief generated from the start probabilities of the model.
        initial_value_function : ValueFunction, optional
            An initial value function to start the solving process with. (Can for example be the MDP value function)
        prune_level : int, default=1
            Parameter to prune the value function further before the expand function.
        prune_interval : int, default=10
            How often to prune the value function. It is counted in number of iterations.
        belief_memory_depth : int, default=10
            How deep the belief memory should be. It is used to speedup the MDP expansion process by caching previously explored beliefs based on the sequence that led it there.
        use_gpu : bool, default=False
            Whether to use the GPU with cupy array to accelerate solving.
        history_tracking_level : int, default=1
            How thorough the tracking of the solving process should be. (0: Nothing; 1: Times and sizes of belief sets and value function; 2: The actual value functions and beliefs sets)
        print_progress : bool, default=True
            Whether or not to print out the progress of the value iteration process.

        Returns
        -------
        value_function : ValueFunction
            The alpha vectors approximating the value function.
        '''
        # numpy or cupy module
        xp = np

        # If GPU usage
        if use_gpu:
            assert gpu_support, "GPU support is not enabled, Cupy might need to be installed..."
            model = model.gpu_model

            # Replace numpy module by cupy for computations
            xp = cp

            # Make sure MDP solution is on gpu as well
            mdp_policy = mdp_policy.to_gpu()

        # Initial belief
        if initial_belief is None:
            b = Belief(model)
        else:
            b = Belief(model, xp.array(initial_belief.values))

        # Initial value function
        if initial_value_function is None:
            value_function = ValueFunction(model, model.expected_rewards_table.T, model.actions)
        else:
            value_function = initial_value_function.to_gpu() if use_gpu else initial_value_function

        # Convergence check boundary
        max_allowed_change = self.eps * (self.gamma / (1-self.gamma))

        # History tracking
        solver_history = SolverHistory(tracking_level=history_tracking_level,
                                       model=model,
                                       gamma=self.gamma,
                                       eps=self.eps,
                                       expand_function='MDP',
                                       expand_append=False,
                                       initial_value_function=value_function,
                                       initial_belief_set=BeliefSet(model, [b])
                                       )

        old_value_function = value_function
        try:
            for i in trange(expansions, desc='Expansions') if print_progress else range(expansions):

                # Expand (exploration)
                start_ts = datetime.now()

                s0 = b.random_state()
                belief_set = self.MDPExplore(model, b, s0, mdp_policy, 0, horizon, sequence_string='', belief_memory_depth=belief_memory_depth)

                expand_time = (datetime.now() - start_ts).total_seconds()
                solver_history.add_expand_step(expansion_time=expand_time, belief_set=belief_set)

                # Backup (update of value function)
                start_ts = datetime.now()

                value_function = self.backup(model, belief_set, value_function, append=True)
                backup_time = (datetime.now() - start_ts).total_seconds()

                # Additional pruning
                if (i % prune_interval) == 0 and i > 0:
                    start_ts = datetime.now()
                    vf_len = len(value_function)

                    value_function.prune(prune_level)

                    prune_time = (datetime.now() - start_ts).total_seconds()
                    alpha_vectors_pruned = len(value_function) - vf_len
                    solver_history.add_prune_step(prune_time, alpha_vectors_pruned)

                # Change computation
                max_change = self.compute_change(old_value_function, value_function, belief_set)

                # History tracking
                solver_history.add_backup_step(backup_time, max_change, value_function)
                
                # Convergence check
                if max_change < max_allowed_change:
                    print('Converged!')
                    break
                
                old_value_function = value_function
        except MemoryError as e:
            print(f'Memory full: {e}')
            print('Returning value function and history as is...')

        # Final pruning
        start_ts = datetime.now()
        vf_len = len(value_function)

        value_function.prune(prune_level)

        prune_time = (datetime.now() - start_ts).total_seconds()
        alpha_vectors_pruned = len(value_function) - vf_len
        solver_history.add_prune_step(prune_time, alpha_vectors_pruned)
        
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

    Parameters
    ----------
    model: mdp.Model
        The model on which the simulation happened on.
    start_state: int
        The initial state in the simulation.
    start_belief: Belief
        The initial belief the agent starts with during the simulation.

    Attributes
    ----------
    model : mdp.Model
    states : list[int]
        A list of recorded states through which the agent passed by during the simulation process.
    grid_point_sequence : list[list[int]]
        A list of 2D points of the grid state through which the agent passed by during the simulation process.
    actions : list[int]
        A list of recorded actions the agent took during the simulation process.
    rewards: RewardSet
        The set of rewards received by the agent throughout the simulation process.
    beliefs : list[Belief]
        A list of recorded beliefs the agent is in throughout the simulation process.
    observations : list[int]
        A list of recorded observations gotten by the agent during the simulation process.
    '''
    def __init__(self, model:Model, start_state:int, start_belief:Belief):
        super().__init__(model, start_state)
        self.beliefs = [start_belief]
        self.observations = []


    def add(self, action:int, reward, next_state:int, next_belief:Belief, observation:int) -> None:
        '''
        Function to add a step in the simulation history

        Parameters
        ----------
        action : int
            The action that was taken by the agent.
        reward
            The reward received by the agent after having taken action.
        next_state : int
            The state that was reached by the agent after having taken action.
        next_belief : Belief
            The new belief of the agent after having taken an action and received an observation.
        observation:int
            The observation the agent received after having made an action.
        '''
        super().add(action, reward, next_state)
        self.beliefs.append(next_belief)
        self.observations.append(observation)


    # Overwritten
    def _plot_to_frame_on_ax(self, frame_i, ax):
        # Data
        data = np.array(self.grid_point_sequence)[:(frame_i+1),:]
        belief = self.beliefs[frame_i]
        observations = self.observations[:(frame_i)]
        obs_colors = ['#000000'] + [COLOR_LIST[o]['hex'] for o in observations]

        # Ticks
        dimensions = self.model.state_grid.shape
        x_ticks = np.arange(0, dimensions[1], (1 if dimensions[1] < 10 else int(dimensions[1] / 10)))
        y_ticks = np.arange(0, dimensions[0], (1 if dimensions[0] < 5 else int(dimensions[0] / 5)))

        # Plotting
        ax.clear()
        ax.set_title(f'Simulation (Frame {frame_i})')

        # Observation labels legend
        proxy = [patches.Rectangle((0,0),1,1,fc = COLOR_LIST[o]['id']) for o in self.model.observations]
        ax.legend(proxy, self.model.observation_labels, title='Observations') # type: ignore

        grid_values = belief.values[self.model.state_grid]
        ax.imshow(grid_values, cmap='Blues')
        ax.plot(data[:,1], data[:,0], color='red', zorder=-1)
        ax.scatter(data[:,1], data[:,0], c=obs_colors)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)


class Simulation(MDP_Simulation):
    '''
    Class to reprensent a simulation process for a POMDP model.
    An initial random state is given and action can be applied to the model that impact the actual state of the agent along with returning a reward and an observation.

    ...

    Parameters
    ----------
    model: pomdp.Model
        The POMDP model the simulation will be applied on.

    Attributes
    ----------
    model: pomdp.Model
    agent_state : int
        The agent's state in the running simulation
    is_done : bool
        Whether or not the agent has reached an end state or performed an ending action.
    '''
    def __init__(self, model:Model) -> None:
        super().__init__(model)
        self.model = model


    def run_action(self, a:int) -> tuple[Union[int,float], int]:
        '''
        Run one step of simulation with action a.

        Parameters
        ----------
        a : int
            The action to take in the simulation.

        Returns
        -------
        r : int or float
            The reward given when doing action a in state s and landing in state s_p. (s and s_p are hidden from agent)
        o : int
            The observation following the action applied on the previous state.
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

    Parameters
    ----------
    model: pomdp.Model
        The model in which the agent can run
    value_function : ValueFunction, optional
        A value function the agent can use to play a simulation, in the case the model has been solved beforehand.
    
    Attributes
    ----------
    model: pomdp.Model
    value_function : ValueFunction
        The value function the agent has come up to after training.
    '''
    def __init__(self, model:Model, value_function:Union[ValueFunction,None]=None) -> None:
        self.model = model
        self.value_function = value_function


    def train(self, solver:PBVI_Solver, expansions:int, horizon:int) -> SolverHistory:
        '''
        Method to train the agent using a given solver.
        The solver will provide a value function that will map beliefs in belief space to actions.

        Parameters
        ----------
        solver : PBVI_Solver
            The solver to run.
        expansions : int
            How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
        horizon : int
            How many times the alpha vector set must be updated every time the belief set is expanded.
        
        Returns
        -------
        solve_history : SolverHistory
            The history of the solving process.
        '''
        self.value_function, solve_history = solver.solve(self.model, expansions, horizon)
        return solve_history


    def get_best_action(self, belief:Belief) -> int:
        '''
        Function to retrieve the best action for a given belief based on the value function retrieved from the training.

        Parameters
        ----------
        belief : Belief
            The belief to get the best action with.
                
        Returns
        -------
        best_action : int
            The best action found.
        '''
        assert self.value_function is not None, "No value function, training probably has to be run..."

        # GPU
        xp = np if not self.value_function.is_on_gpu else cp

        best_vector = xp.argmax(xp.dot(self.value_function.alpha_vector_array, belief.values))
        best_action = int(self.value_function.actions[best_vector])

        return best_action


    def simulate(self,
                 simulator:Union[Simulation,None]=None,
                 max_steps:int=1000,
                 start_state:Union[int,None]=None,
                 print_progress:bool=True,
                 print_stats:bool=True
                 ) -> SimulationHistory:
        '''
        Function to run a simulation with the current agent for up to 'max_steps' amount of steps using a Simulation simulator.

        Parameters
        ----------
        simulator : pomdp.Simulation, optional
            The simulation that will be used by the agent. If not provided, the default MDP simulator will be used.
        max_steps : int, default=1000
            The max amount of steps the simulation can run for.
        start_state : int, optional
            The state the agent should start in, if not provided, will be set at random based on start probabilities of the model.
        print_progress : bool, default=True
            Whether or not to print out the progress of the simulation.
        print_stats : bool, default=True
            Whether or not to print simulation statistics at the end of the simulation.

        Returns
        -------
        history : SimulationHistory
            A list of rewards with the additional functionality that the can be plot with the plot() function.
        '''
        assert self.value_function is not None, "No value function, training probably has to be run..."

        # GPU setup
        self.model = self.model.gpu_model if self.value_function.is_on_gpu else self.model.cpu_model

        # Get or generate a default simulator
        if simulator is None:
            simulator = Simulation(self.model)

        # reset
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
                          start_state:Union[int,None]=None,
                          print_progress:bool=True,
                          print_stats:bool=True
                          ) -> Tuple[RewardSet, list[SimulationHistory]]:
        '''
        Function to run a set of simulations in a row.
        This is useful when the simulation has a 'done' condition.
        In this case, the rewards of individual simulations are summed together under a single number.

        Not implemented:
            - Overal simulation stats

        Parameters
        ----------
        simulator : pomdp.Simulation, optional
            The simulation that will be used by the agent. If not provided, the default MDP simulator will be used. (Optional)
        n : int, default=1000
            The amount of simulations to run.
        max_steps : int, default=1000
            The max_steps to run per simulation.
        start_state : int, optional
            The state the agent should start in, if not provided, will be set at random based on start probabilities of the model (Default: random)
        print_progress : bool, default=True
            Whether or not to print out the progress of the simulation.
        print_stats : bool, default=True
            Whether or not to print simulation statistics at the end of the simulation.

        Returns
        -------
        all_histories : RewardSet
            A list of the final rewards after each simulation.
        all_histories : list[SimulationHistory]
            A list the simulation histories gathered at each simulation.
        '''
        if simulator is None:
            simulator = Simulation(self.model)

        sim_start_ts = datetime.now()

        all_histories = []
        all_final_rewards = RewardSet()
        all_discounted_rewards = []
        all_sim_length = []
        for _ in (trange(n) if print_progress else range(n)):
            sim_history = self.simulate(simulator, max_steps, start_state, False, False)

            all_histories.append(sim_history)
            all_final_rewards.append(np.sum(sim_history.rewards))
            all_discounted_rewards.append(sim_history.rewards.get_total_discounted_reward(0.99)) #TODO: Make it variable
            all_sim_length.append(len(sim_history))

        if print_stats:
            sim_end_ts = datetime.now()
            print(f'All {n} simulations done:')
            print(f'\t- Average runtime (s): {((sim_end_ts - sim_start_ts).total_seconds() / n)}')
            print(f'\t- Average step count: {(sum(all_sim_length) / n)}')
            print(f'\t- Average total rewards: {(sum(all_final_rewards) / n)}')
            print(f'\t- Average discounted rewards (ADR): {(sum(all_discounted_rewards) / n)}')

        return all_final_rewards, all_histories
    

def load_POMDP_file(file_name:str) -> Tuple[Model, PBVI_Solver]:
    '''
    Function to load files of .POMDP format.
    This file format was implemented by Cassandra and the specifications of the format can be found here: https://pomdp.org/code/pomdp-file-spec.html
     
    Then, example models can be found on the following page: https://pomdp.org/examples/

    Parameters
    ----------
    file_name : str
        The name and path of the file to be loaded.

    Returns
    -------
    loaded_model : pomdp.Model
        A POMDP model with the parameters found in the POMDP file. 
    loaded_solver : PBVI_Solver
        A solver with the gamma parameter from the POMDP file.
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
            if ('states' in model_params) and ('actions' in model_params) and ('observations' in model_params) and not ('transitions' in model_params):
                model_params['transitions'] = np.full((state_count, action_count, state_count), 0.0)
            
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
                                model_params['transitions'][s, a, s_p] = float(line_items[-1])
                
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
                            model_params['transitions'][s, a, :] = np.ones(state_count) / state_count
                            continue

                        for s_p, item in enumerate(line_items):
                            model_params['transitions'][s, a, s_p] = float(item)

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
                        model_params['transitions'][:, a, :] = np.ones((state_count, state_count)) / state_count
                        reading = ''
                        continue
                    # Identity
                    if 'identity' in line_items:
                        model_params['transitions'][:, a, :] = np.eye(state_count)
                        reading = ''
                        continue

                    for s_p, item in enumerate(line_items):
                        model_params['transitions'][s, a, s_p] = float(item)

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
            if ('states' in model_params) and ('actions' in model_params) and ('observations' in model_params) and not ('rewards' in model_params):
                model_params['rewards'] = np.full((state_count, action_count, state_count, observation_count), 0.0)

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
                                    model_params['rewards'][s, a, s_p, o] = float(line_items[-1])
                
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
                                model_params['rewards'][s, a, s_p, o] = float(item)

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
                            model_params['rewards'][s, a, s_p, o] = float(item)

                read_lines += 1
                if read_lines == state_count:
                    reading = ''
                    read_lines = 0

    # Generation of output
    loaded_model = Model(**model_params)
    loaded_solver = PBVI_Solver(gamma=gamma_param)

    return (loaded_model, loaded_solver)