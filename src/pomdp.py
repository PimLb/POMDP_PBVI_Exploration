from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from typing import Self, Union

import copy
import datetime
import numpy as np
import matplotlib.ticker as MT
import matplotlib.lines as L
import matplotlib.cm as CM
import matplotlib.colors as C
import math
import random

from src.framework import AlphaVector, ValueFunction, Solver
from src.mdp import Model as MDP_Model

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
    transitions:
        The transition matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.
    rewards:
        The reward matrix, has to be |S| x |A|. If none is provided, it will be randomly generated.
    observations:
        The observation matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.

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
                 observation_table=None
                 ):
        
        super(POMDP_Model, self).__init__(states, actions, transitions, rewards)

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
            assert self.observation_table.shape == (self.state_count, self.action_count, self.observation_count), "observations table doesnt have the right shape, it should be SxAxS"
    

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


class PBVI_Solver(Solver):
    '''
    The Point-Based Value Iteration solver for POMDP Models. It works in two steps, first the backup step that updates the alpha vector set that approximates the value function.
    Then, the expand function that expands the belief set.

    ...
    Attributes
    ----------
    model: POMDP_Model
        The POMDP model the solver will be applied on.

    Methods
    -------
    backup(belief_set:list[Belief], alpha_set:list[AlphaVector], discount_factor:float=0.9):
        The backup function, responsible to update the alpha vector set.
    expand_ssra(belief_set:list[Belief]):
        Random action, belief expansion strategy function.
    expand_ssga(belief_set:list[Belief], alpha_set:list[AlphaVector], eps:float=0.1):
        Expsilon greedy action, belief expansion strategy function.
    expand_ssea(belief_set:list[Belief], alpha_set:list[AlphaVector]):
        Exploratory action, belief expansion strategy function.
    expand_ger(belief_set, alpha_set):
        Greedy error reduction, belief expansion strategy function.
    expand(expand_function:str='ssea', **kwargs):
        The general expand function, used to call the other expand_* functions.
    solve(expansions:int, horizon:int, expand_function:str='ssea', initial_belief=None):
        The general solving function that will call iteratively the expand and the backup function.
    '''
    def __init__(self, model:POMDP_Model):
        super().__init__()
        self.model = model


    def backup(self, belief_set:list[Belief], value_function:ValueFunction, gamma:float=0.9) -> ValueFunction:
        '''
        This function has purpose to update the set of alpha vectors. It does so in 3 steps:
        1. It creates projections from each alpha vector for each possible action and each possible observation
        2. It collapses this set of generated alpha vectors by taking the weighted sum of the alpha vectors weighted by the observation probability and this for each action and for each belief.
        3. Then it further collapses the set to take the best alpha vector and action per belief
        In the end we have a set of alpha vectors as large as the amount of beliefs.

        The alpha vectors are also pruned to avoid duplicates and remove dominated ones.

                Parameters:
                        belief_set (list[Belief]): The belief set to use to generate the new alpha vectors with.
                        alpha_set (ValueFunction): The alpha vectors to generate the new set from.
                        gamma (float): The discount factor used for training, default: 0.9.

                Returns:
                        new_alpha_set (ValueFunction): A list of updated alpha vectors.
        '''
        
        # Step 1
        gamma_a_o_t = {}
        for a in self.model.actions:
            for o in self.model.observations:
                alpa_a_o_set = []
                
                for alpha_i in value_function:
                    alpa_a_o_vect = []
                    
                    for s in self.model.states:
                        products = [(self.model.transition_table[s,a,s_p] * self.model.observation_table[s_p,a,o] * alpha_i[s_p])
                                    for s_p in self.model.states]
                        alpa_a_o_vect.append(gamma * sum(products))
                        
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
            
            for a in self.model.actions:
                
                obs_alpha_sum = np.zeros(self.model.state_count)
                
                for o in self.model.observations:
                    
                    # Argmax of alphas
                    best_alpha_o = np.zeros(self.model.state_count)
                    best_alpha_o_val = -np.inf
                    
                    for alpha_o in gamma_a_o_t[a][o]:
                        val = np.dot(alpha_o, b)
                        if val > best_alpha_o_val:
                            best_alpha_o_val = val
                            best_alpha_o = alpha_o
                            
                    # Sum of the alpha_obs vectors
                    obs_alpha_sum += best_alpha_o
                        
                alpha_a_vect = self.model.reward_table[:,a] + obs_alpha_sum

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
    
    
    def expand_ssra(self, belief_set:list[Belief]) -> list[Belief]:
        '''
        Stochastic Simulation with Random Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state (weighted by the belief) and taking a random action leading to a state s_p and a observation o.
        From this action a and observation o we can update our belief. 

                Parameters:
                        belief_set (list[Belief]): list of beliefs to expand on.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        belief_set_new = copy.deepcopy(belief_set)
        
        for b in belief_set:
            s = b.random_state()
            a = random.choice(self.model.actions)
            s_p = self.model.transition(s, a)
            o = self.model.observe(s_p, a)
            b_new = b.update(a, o)
            
            belief_set_new.append(b_new)
            
        return belief_set_new
    

    def expand_ssga(self, belief_set:list[Belief], value_function:ValueFunction, eps:float=0.1) -> list[Belief]:
        '''
        Stochastic Simulation with Greedy Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state s (weighted by the belief),
         then taking the best action a based on the belief with probability 'epsilon'.
        These lead to a new state s_p and a observation o.
        From this action a and observation o we can update our belief. 

                Parameters:
                        belief_set (list[Belief]): list of beliefs to expand on.
                        value_function (ValueFunction): Used to find the best action knowing the belief.
                        epsilon (float): Parameter tuning how often we take a greedy approach and how often we move randomly.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        belief_set_new = copy.deepcopy(belief_set)
        
        for b in belief_set:
            s = b.random_state()
            
            if random.random() < eps:
                a = random.choice(self.model.actions)
            else:
                best_alpha_index = np.argmax([np.dot(alpha, b) for alpha in value_function])
                a = value_function[best_alpha_index].action
            
            s_p = self.model.transition(s, a)
            o = self.model.observe(s_p, a)
            b_new = b.update(a, o)
            
            belief_set_new.append(b_new)
            
        return belief_set_new
    

    def expand_ssea(self, belief_set:list[Belief]) -> list[Belief]:
        '''
        Stochastic Simulation with Exploratory Action.
        Simulates running steps forward for each possible action knowing we are a state s, chosen randomly with according to the belief probability.
        These lead to a new state s_p and a observation o for each action.
        From all these and observation o we can generate updated beliefs. 
        Then it takes the belief that is furthest away from other beliefs, meaning it explores the most the belief space.

                Parameters:
                        belief_set (list[Belief]): list of beliefs to expand on.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        belief_set_new = copy.deepcopy(belief_set)
        
        for b in belief_set:
            best_b = None
            max_dist = -math.inf
            
            for a in self.model.actions:
                s = b.random_state()
                s_p = self.model.transition(s, a)
                o = self.model.observe(s_p, a)
                b_a = b.update(a, o)
                
                # Check distance with other beliefs
                min_dist = min(float(np.linalg.norm(b_p - b_a)) for b_p in belief_set_new)
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_b = b_a
            
            assert best_b is not None
            belief_set_new.append(best_b)
        
        return belief_set_new
    

    def expand_ger(self, belief_set:list[Belief], value_function:ValueFunction) -> list[Belief]:
        '''
        Greedy Error Reduction

                Parameters:
                        belief_set (list[Belief]): list of beliefs to expand on.
                        value_function (ValueFunction): Used to find the best action knowing the belief.

                Returns:
                        belief_set_new (list[Belief]): union of the belief_set and the expansions of the beliefs in the belief_set
        '''
        print('Not implemented')
        return []


    def expand(self, expand_function:str='ssea', **kwargs) -> list[Belief]:
        '''
        Central method to call one of the functions for a particular expansion strategy:
            - Stochastic Simulation with Random Action (ssra)
            - Stochastic Simulation with Greedy Action (ssga)
            - Stochastic Simulation with Exploratory Action (ssea)
            - Greedy Action Reduction (ger) - not implemented

                Parameters:
                        expand_function (str): One of ssra, ssga, ssea, ger; the expansion strategy
                        kwargs: The arguments to pass to the expansion function
                
                Returns:
                        belief_set_new (list[Belief]): The belief set the expansion function returns. 
        '''
        if expand_function in 'expand_ssra':
            args = {arg: kwargs[arg] for arg in ['belief_set'] if arg in kwargs}
            return self.expand_ssra(**args)
        
        elif expand_function in 'expand_ssga':
            args = {arg: kwargs[arg] for arg in ['belief_set', 'value_function', 'eps'] if arg in kwargs}
            return self.expand_ssga(**args)
        
        elif expand_function in 'expand_ssea':
            args = {arg: kwargs[arg] for arg in ['belief_set'] if arg in kwargs}
            return self.expand_ssea(**args)
        
        elif expand_function in 'expand_ger':
            args = {arg: kwargs[arg] for arg in ['belief_set', 'value_function'] if arg in kwargs}
            return self.expand_ger(**args)
        
        return []


    def solve(self, expansions:int, horizon:int, expand_function:str='ssea', initial_belief=None) -> ValueFunction:
        '''
        Main loop of the Point-Based Value Iteration algorithm.
        It consists in 2 steps, Backup and Expand.
        1. Expand: Expands the belief set base with a expansion strategy given by the parameter expand_function
        2. Backup: Updates the alpha vectors based on the current belief set

                Parameters:
                        expansions (int): How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
                        horizon (int): How many times the alpha vector set must be updated every time the belief set is expanded.
                
                Returns:
                        value_function (ValueFunction): The alpha vectors approximating the value function.

        '''
        if initial_belief is None:
            uni_prob = np.ones(self.model.state_count) / self.model.state_count
            initial_belief = Belief(self.model, uni_prob)
        else:
            initial_belief = Belief(self.model, np.array(initial_belief))
        
        belief_set = [initial_belief]
        value_function = ValueFunction([AlphaVector(self.model.reward_table[:,a], a) for a in self.model.actions])

        # History tracking
        self._solve_history = [{
            'value_functions': value_function,
            'beliefs': belief_set
        }]
        self._solve_run_ts = datetime.datetime.now()
        self._solve_steps_count = 0

        # Loop
        for _ in range(expansions):
            # 1: Expand belief set
            belief_set = self.expand(belief_set=belief_set, value_function=value_function)

            # 2: Backup, update value function (alpha vector set)
            for _ in range(horizon):
                value_function = self.backup(belief_set, value_function)
                self._solve_history.append({
                    'value_functions': value_function,
                    'beliefs': belief_set
                })
                self._solve_steps_count += 1

        self._solved = True
        return value_function


    @property
    def explored_beliefs(self) -> list[Belief]:
        assert self._solved, "solve() has to be run first..."
        return self._solve_history[-1]['beliefs']
    

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
        plt.xticks(np.arange(0, 1.1, 0.1))
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
        def plotSimplex(points, fig=None, 
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
        self.solution.plot(self.model.state_labels, self.model.action_labels, size, (self.explored_beliefs if plot_belief else None))


    def save_history_video(self, custom_name:Union[str,None]=None, compare_with:Union[list, ValueFunction, Self]=[]):
        '''
        Function to generate a video of the training history. Another solved solver or list of solvers can be put in the 'compare_with' parameter.
        These other solver's value function will be overlapped with the 1st value function.
        The explored beliefs of the main solver are also mapped out. (other solvers's explored beliefs will not be plotted)
        Also, a single value function or list of value functions can be sent in but they will be fixed in the video.

        Note: only works for 2-state models.

                Parameters:
                        custom_name (str): Optional, the name the video will be saved with.
                        compare_with (PBVI, ValueFunction, list): Optional, value functions or other solvers to plot against the current solver's history
        '''
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3"
        if self.model.state_count == 2:
            self._save_history_video_2D(custom_name, compare_with)
        elif self.model.state_count == 3:
            print('Not implemented...')


    def _save_history_video_2D(self, custom_name=None, compare_with=[]):
        # Figure definition
        grid_spec = {'height_ratios': [19,1]}
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw=grid_spec)

        colors = plt.get_cmap('Set1').colors #type: ignore
        line_types = ['-', '--', '-.', ':']

        # Solver list
        if isinstance(compare_with, ValueFunction) or isinstance(compare_with, PBVI_Solver):
            compare_with_list = [compare_with] # Single item
        else:
            compare_with_list = compare_with # Already group of items
        solver_list = [self] + compare_with_list
        
        assert len(solver_list) <= len(line_types), f"Plotting can only happen for up to {len(line_types)} solvers..."
        line_types = line_types[:len(solver_list)]


        def plot_on_ax(solver, frame_i:int, ax, line_type:str):
            if isinstance(solver, ValueFunction):
                value_function = solver
            else:
                assert solver._solve_history is not None
                frame_i = frame_i if frame_i <= solver._solve_steps_count else (solver._solve_steps_count - 1)
                history_i = solver._solve_history[frame_i]
                value_function = history_i['value_functions']

            alpha_vects = np.array(value_function)

            m = alpha_vects[:,1] - alpha_vects[:,0] # type: ignore
            m = m.reshape(m.shape[0],1)

            x = np.linspace(0, 1, 100)
            x = x.reshape((1,x.shape[0])).repeat(m.shape[0],axis=0)
            y = (m*x) + alpha_vects[:,0].reshape(m.shape[0],1)

            for i, alpha in enumerate(value_function):
                ax.plot(x[i,:], y[i,:], line_type, color=colors[alpha.action]) # type: ignore

        def animate(frame_i):
            ax1.clear()

            # Alpha vector plotting
            for solver, line_type in zip(([self] + compare_with_list), line_types):
                plot_on_ax(solver, frame_i, ax1, line_type)

            # Belief plotting
            assert self._solved
            history_i = self._solve_history[frame_i]

            beliefs_x = np.array(history_i['beliefs'])[:,1]
            ax2.clear()
            ax2.scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c='red')
            ax2.get_yaxis().set_visible(False)
            ax2.axhline(0, color='black')

        assert self._solved
        max_steps = max([solver._solve_steps_count for solver in solver_list if not isinstance(solver,ValueFunction)])
        ani = FuncAnimation(fig, animate, frames=max_steps, interval=500, repeat=False)
        
        # Title
        assert self._solved
        solved_time = self._solve_run_ts.strftime('%Y%m%d_%H%M%S')

        video_title = f'{custom_name}-' if custom_name is not None else f's_{self.model.state_count}-a_{self.model.action_count}-'
        video_title += f'{solved_time}.mp4'

        # Video saving
        writervideo = animation.FFMpegWriter(fps=10)
        ani.save('./Results/' + video_title, writer=writervideo)
        print(f'Video saved at \'Results/{video_title}\'...')
        plt.close()