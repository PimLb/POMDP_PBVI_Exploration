import copy
import datetime
import matplotlib.ticker as MT
import matplotlib.lines as L
import matplotlib.cm as CM
import matplotlib.colors as C

import numpy as np
import math
import random

from matplotlib import pyplot as plt
from typing import Union

from src.POMDP.pomdp_model  import POMDPModel
from src.POMDP.belief import Belief
from src.POMDP.alpha_vector import AlphaVector
from src.POMDP.value_function import ValueFunction
from src.Util import arreq_in_list


class PBVI:
    '''
    The Point-Based Value Iteration for POMDP Models. It works in two steps, first the backup step that updates the alpha vector set that approximates the value function.
    Then, the expand function that expands the belief set.

    ...
    Attributes
    ----------
    model: Model
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
    def __init__(self, model:POMDPModel):
        self.model = model

        self._solve_history = None
        self._solve_run_ts = None


    def backup(self, belief_set:list[Belief], value_function:ValueFunction, discount_factor:float=0.9) -> ValueFunction:
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
                        discount_factor (float): The discount factor used for training (also known as gamma), default: 0.9.

                Returns:
                        new_alpha_set (ValueFunction): A list of updated alpha vectors.
        '''
        
        # Step 1
        gamma_a_o_t = {}
        for a in self.model.actions:
            for o in self.model.states:
                alpa_a_o_set = []
                
                for alpha_i in value_function:
                    alpa_a_o_vect = []
                    
                    for s in self.model.states:
                        products = [(self.model.transition_table[s,a,s_p] * self.model.observation_table[s_p,a,o] * alpha_i[s_p])
                                    for s_p in self.model.states]
                        alpa_a_o_vect.append(discount_factor * sum(products))
                        
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
                
                for o in self.model.states:
                    
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
        1. Backup: Updates the alpha vectors based on the current belief set
        2. Expand: Expands the belief set base with a expansion strategy given by the parameter expand_function

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
            'alphas': value_function,
            'beliefs': belief_set
        }]
        self._solve_run_ts = datetime.datetime.now()

        # Loop
        for _ in range(expansions):
            for _ in range(horizon):
                value_function = self.backup(belief_set, value_function)
                self._solve_history.append({
                    'alphas': value_function,
                    'beliefs': belief_set
                })

            belief_set = self.expand(belief_set=belief_set, value_function=value_function)

        return value_function
    

    def plot_belief_set(self, size:int=15):
        '''
        Function to plot the last belief set that was generated by the solve function.
        Note: Only works for 2-state and 3-state believes.

                Parameters:
                        size (int): The figure size and general scaling factor
        '''
        assert self._solve_history is not None, "solve() has to be run first..."
        assert self.model.state_count in [2,3], "Can't plot for models with state count other than 2 or 3"
        
        if self.model.state_count == 2:
            self._plot_belief_set_2D(size)
        elif self.model.state_count == 3:
            self._plot_belief_set_3D(size)


    def _plot_belief_set_2D(self, size=15):
        assert self._solve_history is not None
        beliefs_x = np.array(self._solve_history[-1]['beliefs'])[:,1]

        plt.figure(figsize=(size, max([int(size/7),1])))
        plt.scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c=range(beliefs_x.shape[0]), cmap='Blues')
        ax = plt.gca()
        ax.get_yaxis().set_visible(False)
        plt.xticks(np.arange(0,1.1,0.1))
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
        assert self._solve_history is not None
        belief_set = self._solve_history[-1]['beliefs']

        fig = plt.figure(figsize=(size,size))

        cmap = CM.get_cmap('Blues')
        norm = C.Normalize(vmin=0, vmax=len(belief_set))
        c = range(len(belief_set))
        # Do scatter plot
        fig = plotSimplex(np.array(belief_set), fig=fig, vertexlabels=self.model.state_labels, s=size, c=c,                      
                        cmap=cmap, norm=norm)

        plt.show()


    @property
    def explored_beliefs(self) -> list[Belief]:
        assert self._solve_history is not None, "solve() has to be run first..."
        return self._solve_history[-1]['beliefs']
