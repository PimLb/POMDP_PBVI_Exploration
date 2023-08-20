from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from typing import Self, Union, Set

import copy
import datetime
import numpy as np
import matplotlib.ticker as MT
import matplotlib.lines as L
import matplotlib.cm as CM
import matplotlib.colors as C
import math
import random

from src.framework import AlphaVector, ValueFunction, Agent
from src.mdp import MDP_Model, MDP_SolverHistory

class POMDP_Model(MDP_Model):
    '''
    POMDP Model class. Partially Observable Markov Decision Process Model.

    ...

    Attributes
    ----------
    states: int|list
        A list of state labels or an amount of states to be used.
    actions: int|list
        A list of action labels or an amount of actions to be used.
    observations:
        A list of observation labels or an amount of observations to be used
    transitions:
        The transition matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.
    rewards:
        The reward matrix, has to be |S| x |A|. If none is provided, it will be randomly generated.
    observation_table:
        The observation matrix, has to be |S| x |A| x |O|. If none is provided, it will be randomly generated.
    probabilistic_rewards: bool
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.

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
                 observations:Union[int, list],
                 transitions=None,
                 rewards=None,
                 observation_table=None,
                 probabilistic_rewards:bool=False
                 ):
        
        super(POMDP_Model, self).__init__(states, actions, transitions, rewards, probabilistic_rewards)

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


class POMDP_SolverHistory(MDP_SolverHistory):
    '''
    
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
            l1 = L.Line2D([0, 0.5, 1.0, 0], # xcoords
                        [0, np.sqrt(3) / 2, 0, 0], # ycoords
                        color='k')
            fig.gca().add_line(l1)
            fig.gca().xaxis.set_major_locator(MT.NullLocator())
            fig.gca().yaxis.set_major_locator(MT.NullLocator())
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

        cmap = CM.get_cmap('Blues')
        norm = C.Normalize(vmin=0, vmax=len(belief_set))
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
        self.solution.plot(size, self.model.state_labels, self.model.action_labels, (self.explored_beliefs if plot_belief else None))


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
        colors = plt.get_cmap('Set1').colors #type: ignore
        line_types = ['-', '--', '-.', ':']

        proxy = [Rectangle((0,0),1,1,fc = colors[a]) for a in range(self.model.action_count)]

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
                ax.plot(x[i,:], y[i,:], line_type, color=colors[alpha.action])

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

class PBVI_Solver:
    '''
    The Point-Based Value Iteration solver for POMDP Models. It works in two steps, first the backup step that updates the alpha vector set that approximates the value function.
    Then, the expand function that expands the belief set.

    ...
    Attributes
    ----------
    expansions: int
        How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
    horizon: int
        How many times the alpha vector set must be updated every time the belief set is expanded.
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
    backup(model:POMDP_Model, belief_set:list[Belief], alpha_set:list[AlphaVector], discount_factor:float=0.9):
        The backup function, responsible to update the alpha vector set.
    expand_ssra(model:POMDP_Model, belief_set:list[Belief]):
        Random action, belief expansion strategy function.
    expand_ssga(model:POMDP_Model, belief_set:list[Belief], alpha_set:list[AlphaVector], eps:float=0.1):
        Expsilon greedy action, belief expansion strategy function.
    expand_ssea(model:POMDP_Model, belief_set:list[Belief], alpha_set:list[AlphaVector]):
        Exploratory action, belief expansion strategy function.
    expand_ger(model:POMDP_Model, belief_set, alpha_set):
        Greedy error reduction, belief expansion strategy function.
    expand():
        The general expand function, used to call the other expand_* functions.
    solve(model:POMDP_Model, initial_belief:Union[list[Belief], Belief, None]=None, initial_value_function:Union[ValueFunction,None]=None):
        The general solving function that will call iteratively the expand and the backup function.
    '''
    def __init__(self,
                 expansions:int,
                 horizon:int,
                 gamma:float=0.9,
                 eps:float=0.001,
                 expand_function:str='ssea',
                 expand_function_params:dict={}):

        self.expansions = expansions
        self.horizon = horizon
        self.gamma = gamma
        self.eps = eps
        self.expand_function = expand_function
        self.expand_function_params = expand_function_params



    def backup(self, model:POMDP_Model, belief_set:list[Belief], value_function:ValueFunction) -> ValueFunction:
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
        new_value_function = ValueFunction([])
        
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
    
    
    def expand_ssra(self, model:POMDP_Model, belief_set:list[Belief]) -> list[Belief]:
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
    

    def expand_ssga(self, model:POMDP_Model, belief_set:list[Belief], value_function:ValueFunction, eps:float=0.1) -> list[Belief]:
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
    

    def expand_ssea(self, model:POMDP_Model, belief_set:list[Belief]) -> list[Belief]:
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
    

    def expand_ger(self, model:POMDP_Model, belief_set:list[Belief], value_function:ValueFunction) -> list[Belief]:
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
              model:POMDP_Model,
              initial_belief:Union[list[Belief], Belief, None]=None,
              initial_value_function:Union[ValueFunction,None]=None,
              ) -> tuple[ValueFunction, POMDP_SolverHistory]:
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
                        model (POMDP_Model) - The model to solve.
                        initial_belief (list[Belief], Belief) - Optional: An initial list of beliefs to start with.
                        initial_value_function (ValueFunction) - Optional: An initial value function to start the solving process with.

                Returns:
                        value_function (ValueFunction): The alpha vectors approximating the value function.

        '''

        # Initial belief
        if initial_belief is None:
            uni_prob = np.ones(model.state_count) / model.state_count
            belief_set = [Belief(model, uni_prob)]
        elif isinstance(initial_belief, list):
            belief_set = initial_belief
        else:
            belief_set = [Belief(model, np.array(initial_belief))]
        
        # Initial value function
        if initial_value_function is None:
            value_function = ValueFunction([AlphaVector(model.expected_rewards_table[:,a], a) for a in model.actions])
        else:
            value_function = initial_value_function

        # History tracking
        solver_history = POMDP_SolverHistory(model=model,
                                             expand_function=self.expand_function,
                                             gamma=self.gamma,
                                             eps=self.eps)
        solver_history.append({
            'value_function': value_function,
            'beliefs': belief_set
        })

        # Loop
        for _ in range(self.expansions):
            # 1: Expand belief set
            self.expand_function_params['model'] = model
            self.expand_function_params['belief_set'] = belief_set
            self.expand_function_params['value_function'] = value_function
            belief_set = self.expand()

            old_max_val_per_belief = None

            # 2: Backup, update value function (alpha vector set)
            for _ in range(self.horizon):
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
                    if max_change < self.eps:
                        print('Converged early...')
                        return value_function, solver_history
                old_max_val_per_belief = max_val_per_belief

        return value_function, solver_history


class Simulation:
    '''
    Class to reprensent a simulation process for a POMDP model.
    An initial random state is given and action can be applied to the model that impact the actual state of the agent along with returning a reward and an observation.

    ...
    Attributes
    ----------
    model: POMDP_Model
        The POMDP model the simulation will be applied on.

    Methods
    -------
    initialize_simulation():
        The function to initialize the simulation with a random state for the agent.
    run_action(a:int):
        Runs the action a on the current state of the agent.
    '''

    def __init__(self, model:POMDP_Model) -> None:
        self.model = model

        self.initialize_simulation()


    def initialize_simulation(self) -> None:
        '''
        Function to initialize the simulation by setting a random start state to the agent.
        '''
        self.agent_state = random.choice(self.model.states)


    def run_action(self, a:int) -> tuple[int, Union[int,float]]:
        '''
        Run one step of simulation with action a.

                Parameters:
                        a (int): the action to take in the simulation.

                Returns:
                        o (int): the observation following the action applied on the previous state
                        r (int, float): the reward given when doing action a in state s and landing in state s_p. (s and s_p are hidden from agent)
        '''
        s = self.agent_state
        s_p = self.model.transition(s,a)
        r = self.model.reward(s,a,s_p)
        o = self.model.observe(s_p, a)
        return (o,r)


class POMDP_Agent(Agent):
    def __init__(self, model):
        super().__init__()

        self.model = model


    def train(self, solver:PBVI_Solver):
        # solver.solve()
        pass