from matplotlib import animation, cm, colors, ticker
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from tqdm import tqdm, trange
from typing import Self, Tuple, Union

import copy
import json
import numpy as np
import math
import random

from src.mdp import AlphaVector, ValueFunction
from src.mdp import Model as MDP_Model
from src.mdp import SimulationHistory
from src.mdp import SolverHistory as MDP_SolverHistory
from src.mdp import Solver as MDP_Solver
from src.mdp import Simulation as MDP_Simulation


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
    transition_table:
        The transition matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.
    immediate_rewards:
        The reward matrix, has to be |S| x |A| x |S| x |O|. If none is provided, it will be randomly generated.
    observation_table:
        The observation matrix, has to be |S| x |A| x |O|. If none is provided, it will be randomly generated.
    probabilistic_rewards: bool
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.
    grid_states: list[list[Union[str,None]]]
        Optional, if provided, the model will be converted to a grid model. Allows for 'None' states if there is a gaps in the grid.
    start_probabilities: list
        Optional, the distribution of chances to start in each state. If not provided, there will be an uniform chance for each state. It is also used to represent a belief of complete uncertainty.

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
                 transition_table=None,
                 immediate_reward_table=None,
                 observation_table=None,
                 probabilistic_rewards:bool=False,
                 grid_states:Union[None,list[list[Union[str,None]]]]=None,
                 start_probabilities:Union[list,None]=None
                 ):
        
        super().__init__(states=states,
                         actions=actions,
                         transition_table=transition_table,
                         immediate_reward_table=None, # Defined here lower since immediate reward table has different shape for MDP is different than for POMDP
                         probabilistic_rewards=probabilistic_rewards,
                         grid_states=grid_states,
                         start_probabilities=start_probabilities)

        if isinstance(observations, int):
            self.observation_labels = [f'o_{i}' for i in range(observations)]
        else:
            self.observation_labels = observations
        self.observation_count = len(self.observation_labels)
        self.observations = [obs for obs in range(self.observation_count)]

        if observation_table is None:
            # If no observation matrix given, generate random one
            random_probs = np.random.rand(self.state_count, self.action_count, self.observation_count)
            # Normalization to have s_p probabilies summing to 1
            self.observation_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.observation_table = np.array(observation_table)
            assert self.observation_table.shape == (self.state_count, self.action_count, self.observation_count), "observations table doesnt have the right shape, it should be SxAxO"

        # Rewards
        if immediate_reward_table is None:
            # If no reward matrix given, generate random one
            self.immediate_reward_table = np.random.rand(self.state_count, self.action_count, self.state_count, self.observation_count)
        else:
            self.immediate_reward_table = np.array(immediate_reward_table)
            assert self.immediate_reward_table.shape == (self.state_count, self.action_count, self.state_count, self.observation_count), "rewards table doesnt have the right shape, it should be SxAxSxO"

        # Expected rewards
        self.expected_rewards_table = np.zeros((self.state_count, self.action_count))
        for s in self.states:
            for a in self.actions:
                sum = 0
                for s_p in self.states:
                    inner_sum = 0
                    for o in self.observations:
                        inner_sum += (self.observation_table[s_p,a,o] * self.immediate_reward_table[s,a,s_p,o])
                    sum += (self.transition_table[s,a,s_p] * inner_sum)
                self.expected_rewards_table[s,a] = sum


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
        reward = self.immediate_reward_table[s,a,s_p,o]
        if self.probabilistic_rewards:
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
    

    def to_dict(self) -> dict:
        '''
        Function to return a python dictionary with all the information of the model.

                Returns:
                        model_dict (dict): The representation of the model in a dictionary format.
        '''
        model_dict = super().to_dict()
        model_dict['observations'] = self.observation_labels
        model_dict['observation_table'] = self.observation_table.tolist()

        return model_dict
    
    
    @classmethod
    def load_from_json(cls, file:str) -> Self:
        '''
        Function to load a POMDP model from a json file. The json structure must contain the same items as in the constructor of this class.

                Parameters:
                        file (str): The file and path of the model to be loaded.
                Returns:
                        loaded_model (pomdp.Model): An instance of the loaded model.
        '''
        with open(file, 'r') as openfile:
            json_model = json.load(openfile)

        loaded_model = Model(**json_model)

        if 'grid_states' in json_model:
            loaded_model.convert_to_grid(json_model['grid_states'])

        return loaded_model


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
    plot(size:int=5):
        Function to plot a value function as a grid heatmap.
    '''
    def __new__(cls, model:Model, state_probabilities:Union[np.ndarray,None]=None):
        if state_probabilities is not None:
            assert state_probabilities.shape[0] == model.state_count, "Belief must contain be of dimension |S|"
            prob_sum = np.round(sum(state_probabilities), decimals=3)
            assert prob_sum == 1, f"States probabilities in belief must sum to 1 (found: {prob_sum})"
            obj = np.asarray(state_probabilities).view(cls)
        else:
            obj = model.start_probabilities.view(cls)
        obj._model = model # type: ignore
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self._model = getattr(obj, '_model', None)


    @property
    def model(self) -> Model:
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

        # Formal definition of the normalizer as in paper
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
    

    def plot(self, size:int=5) -> None:
        '''
        Function to plot a heatmap of the belief distribution if the belief is of a grid model.

                Parameters:
                        size (int): The scale of the plot. (Default: 5)
        '''
        assert self.model.grid_states is not None
        dimensions = (len(self.model.grid_states), len(self.model.grid_states[0]))

        values = np.zeros(dimensions)

        for x in range(values.shape[0]):
            for y in range(values.shape[1]):
                state_label = self.model.grid_states[x][y]
                
                if state_label is not None:
                    s = self.model.state_labels.index(state_label) # type: ignore
                    values[x,y] = self[s]

        plt.figure(figsize=(size*1.2,size))
        plt.imshow(values,cmap='Blues')
        plt.colorbar()
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
    params:
        The solver parameters that have been used, to show better information on the visualizations.

    Methods
    -------
    plot_belief_set(size:int=15):
        Once solve() has been run, the explored beliefs can be plot for 2- and 3- state models.
    plot_solution(size:int=5, plot_belief:bool=True):
        Once solve() has been run, the value function solution can be plot for 2- and 3- state models.
    save_history_video(custom_name:Union[str,None]=None, compare_with:Union[list, ValueFunction, Self]=[]):
        Once the solve has been run, we can save a video of the history of the solving process.
    '''
    def __init__(self, model, **params):
        super().__init__(model, **params)


    @property
    def explored_beliefs(self) -> list[Belief]:
        return self[-1]['beliefs']
    

    def plot_belief_set(self, size:int=15):
        '''
        Function to plot the last belief set that was generated by the solve function.
        Note: Only works for 2-state and 3-state believes.

                Parameters:
                        size (int): The figure size and general scaling factor
        '''
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3"
        
        if self.model.state_count == 2:
            self._plot_belief_set_2D(size)
        elif self.model.state_count == 3:
            self._plot_belief_set_3D(size)


    def _plot_belief_set_2D(self, size=15):
        beliefs_x = np.array(self.explored_beliefs)[:,1]

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


    def _plot_belief_set_3D(self, size=15):
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
        belief_set = self.explored_beliefs

        fig = plt.figure(figsize=(size,size))

        cmap = cm.get_cmap('Blues')
        norm = colors.Normalize(vmin=0, vmax=len(belief_set))
        c = range(len(belief_set))
        # Do scatter plot
        fig = plotSimplex(np.array(belief_set), fig=fig, vertexlabels=self.model.state_labels, s=size, c=c,                      
                        cmap=cmap, norm=norm)

        plt.show()


    def plot_solution(self, size:int=5, plot_belief:bool=True):
        '''
        Function to plot the value function of the solution.
        Note: only works for 2 and 3 states models

                Parameters:
                        size (int): The figure size and general scaling factor
        '''
        self.solution.plot(size=size, belief_set=(self.explored_beliefs if plot_belief else None))


    def save_history_video(self,
                           custom_name:Union[str,None]=None,
                           compare_with:Union[list, ValueFunction, MDP_SolverHistory]=[],
                           graph_names:list[str]=[]
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
        '''
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3"
        if self.model.state_count == 2:
            self._save_history_video_2D(custom_name, compare_with, copy.copy(graph_names))
        elif self.model.state_count == 3:
            print('Not implemented...')


    def _save_history_video_2D(self, custom_name=None, compare_with=[], graph_names=[]):
        # Figure definition
        grid_spec = {'height_ratios': [19,1]}
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=grid_spec)

        # Figure title
        fig.suptitle(f"{self.model.state_count}-s {self.model.action_count}-a {self.model.observation_count}-o POMDP model solve history", fontsize=16)
        title = f'{self.params["expand_function"]} expand strat, {self.params["gamma"]}-gamma, {self.params["eps"]}-eps '

        # X-axis setting
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = self.model.state_labels[0]
        x_ticks[-1] = self.model.state_labels[1]

        # Colors and lines
        line_types = ['-', '--', '-.', ':']

        proxy = [Rectangle((0,0),1,1,fc = COLOR_LIST[a]['id']) for a in range(self.model.action_count)]

        # Solver list
        if isinstance(compare_with, ValueFunction) or isinstance(compare_with, MDP_SolverHistory):
            compare_with_list = [compare_with] # Single item
        else:
            compare_with_list = compare_with # Already group of items
        solver_list = [self] + compare_with_list
        
        assert len(solver_list) <= len(line_types), f"Plotting can only happen for up to {len(line_types)} solvers..."
        line_types = line_types[:len(solver_list)]

        assert len(graph_names) in [0, len(solver_list)], "Not enough graph names provided"
        if len(graph_names) == 0:
            graph_names.append('Main graph')
            for i in range(1,len(solver_list)):
                graph_names.append(f'Comparison {i}')

        def plot_on_ax(solver, frame_i:int, ax, line_type:str):
            if isinstance(solver, ValueFunction):
                value_function = solver
            else:
                frame_i = frame_i if frame_i < len(solver) else (len(solver) - 1)
                history_i = solver[frame_i]
                value_function = history_i['value_function']

            alpha_vects = np.array(value_function)
            m = np.subtract(alpha_vects[:,1], alpha_vects[:,0])
            m = m.reshape(m.shape[0],1)

            x = np.linspace(0, 1, 100)
            x = x.reshape((1,x.shape[0])).repeat(m.shape[0],axis=0)
            y = np.add((m*x), alpha_vects[:,0].reshape(m.shape[0],1))

            for i, alpha in enumerate(value_function):
                ax.plot(x[i,:], y[i,:], line_type, color=COLOR_LIST[alpha.action]['id'])

        def animate(frame_i):
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
            point = self[self_frame_i]['value_function'][0][0]
            for l in line_types:
                lines.append(Line2D([0,point],[0,point],linestyle=l))
            ax1.legend(lines, graph_names, loc='lower center')

            # Alpha vector plotting
            for solver, line_type in zip(solver_list, line_types):
                plot_on_ax(solver, frame_i, ax1, line_type)

            # Belief plotting
            history_i = self[self_frame_i]

            beliefs_x = np.array(history_i['beliefs'])[:,1]
            ax2.scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c='red')
            ax2.get_yaxis().set_visible(False)
            ax2.axhline(0, color='black')

        max_steps = max([len(solver) for solver in solver_list if not isinstance(solver,ValueFunction)])
        ani = FuncAnimation(fig, animate, frames=max_steps, interval=500, repeat=False)
        
        # File Title
        solved_time = self.run_ts.strftime('%Y%m%d_%H%M%S')

        video_title = f'{custom_name}-' if custom_name is not None else '' # Base
        video_title += f's{self.model.state_count}-a{self.model.action_count}-' # Model params
        video_title += f'{self.params["expand_function"]}-' # Expand function used
        video_title += f'g{self.params["gamma"]}-e{self.params["eps"]}-' # Solving params
        video_title += f'{solved_time}.mp4'

        # Video saving
        writervideo = animation.FFMpegWriter(fps=10)
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



    def backup(self, model:Model, belief_set:list[Belief], value_function:ValueFunction) -> ValueFunction:
        '''
        This function has purpose to update the set of alpha vectors. It does so in 3 steps:
        1. It creates projections from each alpha vector for each possible action and each possible observation
        2. It collapses this set of generated alpha vectors by taking the weighted sum of the alpha vectors weighted by the observation probability and this for each action and for each belief.
        3. Then it further collapses the set to take the best alpha vector and action per belief
        In the end we have a set of alpha vectors as large as the amount of beliefs.

        The alpha vectors are also pruned to avoid duplicates and remove dominated ones.

                Parameters:
                        model (POMDP): The model on which to run the backup method on.
                        belief_set (list[Belief]): The belief set to use to generate the new alpha vectors with.
                        alpha_set (ValueFunction): The alpha vectors to generate the new set from.
                        gamma (float): The discount factor used for training, default: 0.9.

                Returns:
                        new_alpha_set (ValueFunction): A list of updated alpha vectors.
        '''
        
        # Step 1
        gamma_a_o_t = {}
        for a in model.actions:
            for o in model.observations:
                alpa_a_o_set = []
                
                for alpha_i in value_function:
                    alpa_a_o_vect = []
                    
                    for s in model.states:
                        products = [(model.transition_table[s,a,s_p] * model.observation_table[s_p,a,o] * alpha_i[s_p])
                                    for s_p in model.states]
                        alpa_a_o_vect.append(self.gamma * sum(products))
                        
                    alpa_a_o_set.append(alpa_a_o_vect)
                    
                if a not in gamma_a_o_t:
                    gamma_a_o_t[a] = {o: alpa_a_o_set}
                else:
                    gamma_a_o_t[a][o] = alpa_a_o_set

        # Step 2
        new_value_function = ValueFunction(model)
        
        for b in belief_set:
            
            best_alpha = None
            best_alpha_val = -np.inf
            
            for a in model.actions:
                
                obs_alpha_sum = np.zeros(model.state_count)
                
                for o in model.observations:
                    
                    # Argmax of alphas
                    best_alpha_o = np.zeros(model.state_count)
                    best_alpha_o_val = -np.inf
                    
                    for alpha_o in gamma_a_o_t[a][o]:
                        val = np.dot(alpha_o, b)
                        if val > best_alpha_o_val:
                            best_alpha_o_val = val
                            best_alpha_o = alpha_o
                            
                    # Sum of the alpha_obs vectors
                    obs_alpha_sum += best_alpha_o
                        
                alpha_a_vect = model.expected_rewards_table[:,a] + obs_alpha_sum

                # Step 3
                val = np.dot(alpha_a_vect, b)
                if val > best_alpha_val:
                    best_alpha_val = val
                    best_alpha = AlphaVector(alpha_a_vect, a)

            assert best_alpha is not None
            new_value_function.append(best_alpha)

        # Pruning
        new_value_function = new_value_function.prune(level=1) # Just check for duplicates
                
        return new_value_function
    
    
    def expand_ssra(self, model:Model, belief_set:list[Belief]) -> list[Belief]:
        '''
        Stochastic Simulation with Random Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state (weighted by the belief) and taking a random action leading to a state s_p and a observation o.
        From this action a and observation o we can update our belief. 

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (list[Belief]): list of beliefs to expand on.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        belief_set_new = copy.deepcopy(belief_set)
        
        for b in belief_set:
            s = b.random_state()
            a = random.choice(model.actions)
            s_p = model.transition(s, a)
            o = model.observe(s_p, a)
            b_new = b.update(a, o)
            
            belief_set_new.append(b_new)
            
        return belief_set_new
    

    def expand_ssga(self, model:Model, belief_set:list[Belief], value_function:ValueFunction, eps:float=0.1) -> list[Belief]:
        '''
        Stochastic Simulation with Greedy Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state s (weighted by the belief),
         then taking the best action a based on the belief with probability 'epsilon'.
        These lead to a new state s_p and a observation o.
        From this action a and observation o we can update our belief. 

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (list[Belief]): list of beliefs to expand on.
                        value_function (ValueFunction): Used to find the best action knowing the belief.
                        eps (float): Parameter tuning how often we take a greedy approach and how often we move randomly.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        belief_set_new = copy.deepcopy(belief_set)
        
        for b in belief_set:
            s = b.random_state()
            
            if random.random() < eps:
                a = random.choice(model.actions)
            else:
                best_alpha_index = np.argmax([np.dot(alpha, b) for alpha in value_function])
                a = value_function[best_alpha_index].action
            
            s_p = model.transition(s, a)
            o = model.observe(s_p, a)
            b_new = b.update(a, o)
            
            belief_set_new.append(b_new)
            
        return belief_set_new
    

    def expand_ssea(self, model:Model, belief_set:list[Belief]) -> list[Belief]:
        '''
        Stochastic Simulation with Exploratory Action.
        Simulates running steps forward for each possible action knowing we are a state s, chosen randomly with according to the belief probability.
        These lead to a new state s_p and a observation o for each action.
        From all these and observation o we can generate updated beliefs. 
        Then it takes the belief that is furthest away from other beliefs, meaning it explores the most the belief space.

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (list[Belief]): list of beliefs to expand on.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        belief_set_new = copy.deepcopy(belief_set)
        
        for b in belief_set:
            best_b = None
            max_dist = -math.inf
            
            for a in model.actions:
                s = b.random_state()
                s_p = model.transition(s, a)
                o = model.observe(s_p, a)
                b_a = b.update(a, o)
                
                # Check distance with other beliefs
                min_dist = min(float(np.linalg.norm(b_p - b_a)) for b_p in belief_set_new)
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_b = b_a
            
            assert best_b is not None
            belief_set_new.append(best_b)
        
        return belief_set_new
    

    def expand_ger(self, model:Model, belief_set:list[Belief], value_function:ValueFunction) -> list[Belief]:
        '''
        Greedy Error Reduction

                Parameters:
                        model (POMDP): the POMDP model on which to expand the belief set on.
                        belief_set (list[Belief]): list of beliefs to expand on.
                        value_function (ValueFunction): Used to find the best action knowing the belief.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        print('Not implemented')
        return []


    def expand(self) -> list[Belief]:
        '''
        Central method to call one of the functions for a particular expansion strategy:
            - Stochastic Simulation with Random Action (ssra)
            - Stochastic Simulation with Greedy Action (ssga)
            - Stochastic Simulation with Exploratory Action (ssea)
            - Greedy Error Reduction (ger) - not implemented
                
                Returns:
                        belief_set_new (list[Belief]): The belief set the expansion function returns. 
        '''
        kwargs = self.expand_function_params

        if self.expand_function in 'expand_ssra':
            args = {arg: kwargs[arg] for arg in ['model', 'belief_set'] if arg in kwargs}
            return self.expand_ssra(**args)
        
        elif self.expand_function in 'expand_ssga':
            args = {arg: kwargs[arg] for arg in ['model', 'belief_set', 'value_function', 'eps'] if arg in kwargs}
            return self.expand_ssga(**args)
        
        elif self.expand_function in 'expand_ssea':
            args = {arg: kwargs[arg] for arg in ['model', 'belief_set'] if arg in kwargs}
            return self.expand_ssea(**args)
        
        elif self.expand_function in 'expand_ger':
            args = {arg: kwargs[arg] for arg in ['model', 'belief_set', 'value_function'] if arg in kwargs}
            return self.expand_ger(**args)
        
        return []


    def solve(self,
              model:Model,
              expansions:int,
              horizon:int,
              initial_belief:Union[list[Belief], Belief, None]=None,
              initial_value_function:Union[ValueFunction,None]=None,
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
                        initial_belief (list[Belief], Belief) - Optional: An initial list of beliefs to start with.
                        initial_value_function (ValueFunction) - Optional: An initial value function to start the solving process with.
                        print_progress (bool): Whether or not to print out the progress of the value iteration process. (Default: True)

                Returns:
                        value_function (ValueFunction): The alpha vectors approximating the value function.
        '''

        # Initial belief
        if initial_belief is None:
            belief_set = [Belief(model)]
        elif isinstance(initial_belief, list):
            belief_set = initial_belief
        else:
            belief_set = [Belief(model, np.array(initial_belief))]
        
        # Initial value function
        if initial_value_function is None:
            value_function = ValueFunction(model, [AlphaVector(model.expected_rewards_table[:,a], a) for a in model.actions])
        else:
            value_function = initial_value_function

        max_allowed_change = self.eps * (self.gamma / (1-self.gamma))

        # History tracking
        solver_history = SolverHistory(model=model,
                                       expand_function=self.expand_function,
                                       gamma=self.gamma,
                                       eps=self.eps)
        solver_history.append({
            'value_function': value_function,
            'beliefs': belief_set
        })

        # Loop
        for expansion_i in range(expansions) if not print_progress else trange(expansions, desc='Expansions'):
            # 1: Expand belief set
            self.expand_function_params['model'] = model
            self.expand_function_params['belief_set'] = belief_set
            self.expand_function_params['value_function'] = value_function
            belief_set = self.expand()

            old_max_val_per_belief = None

            # 2: Backup, update value function (alpha vector set)
            for _ in range(horizon) if not print_progress else trange(horizon, desc=f'Backups {expansion_i}'):
                old_value_function = copy.deepcopy(value_function)
                value_function = self.backup(model, belief_set, old_value_function)

                solver_history.append({
                    'value_function': value_function,
                    'beliefs': belief_set
                })

                # convergence check
                max_val_per_belief = np.max(np.matmul(np.array(belief_set), np.array(value_function).T), axis=1)
                if old_max_val_per_belief is not None:
                    max_change = np.max(np.abs(max_val_per_belief - old_max_val_per_belief))
                    if max_change < max_allowed_change:
                        print('Converged early...')
                        return value_function, solver_history
                old_max_val_per_belief = max_val_per_belief

        return value_function, solver_history


class Simulation(MDP_Simulation):
    '''
    Class to reprensent a simulation process for a POMDP model.
    An initial random state is given and action can be applied to the model that impact the actual state of the agent along with returning a reward and an observation.

    ...
    Attributes
    ----------
    model: pomdp.Model
        The POMDP model the simulation will be applied on.
    done_on_reward: bool
        If the simulation is to end whenever a reward other than zero is received.

    Methods
    -------
    initialize_simulation():
        The function to initialize the simulation with a random state for the agent.
    run_action(a:int):
        Runs the action a on the current state of the agent.
    '''
    def __init__(self,
                 model:Model,
                 done_on_reward:bool=False,
                 done_on_state: Union[int,list[int]]=[],
                 done_on_action:Union[int,list[int]]=[]
                 ) -> None:
        
        super().__init__(model, done_on_reward, done_on_state, done_on_action)
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

        # Reward Done check
        if self.done_on_reward and (r != 0):
            self.is_done = True

        # State Done check
        if s_p in self.done_on_state:
            self.is_done = True

        # Action Done check
        if a in self.done_on_action:
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

        best_value = -np.inf
        best_action = -1

        for alpha in self.value_function:
            value = np.dot(alpha, belief)
            if best_value < value:
                best_value = value
                best_action = alpha.action

        return best_action


    def simulate(self, simulator:Simulation, max_steps:int=1000) -> SimulationHistory:
        '''
        Function to run a simulation with the current agent for up to 'max_steps' amount of steps using a Simulation simulator.

        Not yet implemented:
            - Stats about how long the simulation took, how often the belief was right about the true state,...

                Parameters:
                        simulator (pomdp.Simulation): the simulation that will be used by the agent.
                        max_steps (int): the max amount of steps the simulation can run for.

                Returns:
                        history (SimulationHistory): A list of rewards with the additional functionality that the can be plot with the plot() function.
        '''
        simulator.initialize_simulation()
        belief = Belief(self.model)

        history = SimulationHistory()

        # Simulation loop
        for i in range(max_steps):
            # Iteration history dict
            history_dict = {
                'belief': belief,
                'state': simulator.agent_state
            }

            # Play best action
            a = self.get_best_action(belief)
            r,o = simulator.run_action(a)

            # Update the belief
            new_belief = belief.update(a, o)

            # Post action history recording
            history_dict.update({
                'action': a,
                'next_state': simulator.agent_state,
                'next_belief': new_belief,
                'reward': r,
                'observation': o
            })
            history.append(history_dict)

            # Replace belief
            belief = new_belief

            # If simulation is considered done, the rewards are simply returned
            if simulator.is_done:
                break
            
        return history


    def run_n_simulations(self, simulator:Simulation, n:int) -> list[SimulationHistory]:
        '''
        Function to run a set of simulations in a row.
        This is useful when the simulation has a 'done' condition.
        In this case, the rewards of individual simulations are summed together under a single number.

        Not implemented:
            - Overal simulation stats

                Parameters:
                        simulator (Simulation): the simulation that will be used by the agent.
                        n (int): the amount of simulations to run.

                Returns:
                        all_histories (list[SimulationHistory]): A list of simulation histories.
        '''
        all_histories = []
        for _ in range(n):
            history = self.simulate(simulator)
            all_histories.append(np.sum(history))

        return all_histories
    

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