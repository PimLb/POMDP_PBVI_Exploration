from matplotlib import pyplot as plt
from scipy.optimize import milp, LinearConstraint
from typing import Self

import copy
import datetime
import numpy as np
import random


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
    '''

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
        
        alpha_set = []

        # Level 1 pruning: Check for duplicates
        if level >= 1:
            L = {array.tobytes(): array for array in self}
            alpha_set = list(L.values())
        
        # Level 2 pruning: Check for absolute domination
        if level >= 2:
            new_alpha_set = []
            for alpha_vector in alpha_set:
                dominated = False
                for compare_alpha_vector in alpha_set:
                    if all(alpha_vector < compare_alpha_vector):
                        dominated = True
                if not dominated:
                    new_alpha_set.append(alpha_vector)
            alpha_set = new_alpha_set

        # Level 3 pruning: LP to check for more complex domination
        elif level >= 3:
            pruned_alphas = []
            state_count = alpha_set[0].shape[0]

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
                belief_constraint = LinearConstraint(np.array([0] + ([1]*state_count)), 1, 1)

                # Solve problem
                res = milp(c=c, constraints=[alpha_constraints, belief_constraint])

                # Check if dominated
                is_dominated = (res.x[0] - np.dot(res.x[1:], alpha_vect)) >= 0
                if is_dominated:
                    print(alpha_vect)
                    print(' -> Dominated\n')

                else:
                    pruned_alphas.append(alpha_vect)

            alpha_set = pruned_alphas
        
        return ValueFunction(alpha_set)
    

    def plot(self, state_list:list[str], action_list:list[str], size:int=5, belief_set=None):
        '''
        Function to plot out the value function in 2 or 3 dimensions.

                Parameters:
                        size (int): The actual plot scale
                        belief_set (list[Belief]): Optional, a set of belief to plot the belief points that were explored.
        '''
        assert len(self) > 0, "Value function is empty, plotting is impossible..."
        dimension = self[0].shape[0]
        assert dimension in [2,3], "Value function plotting only available for MDP's of 2 or 3 states."

        if dimension == 2:
            self._plot_2D(state_list, action_list, size, belief_set)
        elif dimension == 3:
            self._plot_3D(state_list, action_list, size, belief_set)


    def _plot_2D(self, state_list, action_list, size=5, belief_set=None):
        x = np.linspace(0, 1, 100)
        colors = plt.get_cmap('Set1').colors # type: ignore

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
            ax1.plot(x[i,:], y[i,:], color=colors[alpha.action]) # type: ignore

        # Belief plotting
        if belief_set is not None:
            beliefs_x = np.array(belief_set)[:,1]
            ax[1].scatter(beliefs_x, np.zeros(beliefs_x.shape[0]), c='red')
            ax[1].get_yaxis().set_visible(False)
            ax[1].axhline(0, color='black')

        plt.show()


    def _plot_3D(self, state_list, action_list, size=5, belief_set=None):

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
                    
        fig, ((ax0, ax1),(ax2,ax3)) = plt.subplots(2, 2, figsize=(size*4,size*3.5), sharex=True, sharey=True)

        # Set ticks
        ticks = [0,0.25,0.5,0.75,1]
        x_ticks = [str(t) for t in ticks]
        x_ticks[0] = state_list[0]
        x_ticks[-1] = state_list[1]
        
        y_ticks = [str(t) for t in ticks]
        y_ticks[0] = ''
        y_ticks[-1] = state_list[2]

        plt.setp([ax0,ax1,ax2,ax3], xticks=ticks, xticklabels=x_ticks, yticks=ticks, yticklabels=y_ticks)

        # Value function ax
        ax0.set_title("Value function")
        ax0_plot = ax0.contourf(x, y, max_z, 100, cmap="viridis")
        plt.colorbar(ax0_plot, ax=ax0)
        if belief_points is not None:
            ax0.scatter(belief_points[:,0], belief_points[:,1], s=1, c='red')

        # Alpha planes ax
        ax1.set_title("Alpha planes")
        ax1_plot = ax1.contourf(x, y, plane, 100, cmap="viridis")
        plt.colorbar(ax1_plot, ax=ax1)
        if belief_points is not None:
            ax1.scatter(belief_points[:,0], belief_points[:,1], s=1, c='red')
        
        # Gradient of planes ax
        ax2.set_title("Gradients of planes")
        ax2_plot = ax2.contourf(x, y, gradients, 100, cmap="Blues")
        plt.colorbar(ax2_plot, ax=ax2)
        if belief_points is not None:
            ax2.scatter(belief_points[:,0], belief_points[:,1], s=1, c='red')

        # Action policy ax
        # actions colors
        colors = plt.get_cmap('Set1').colors #type: ignore

        ax3.set_title("Action policy")
        ax3_plot = ax3.contourf(x, y, best_a, 1, colors=colors)
        plt.colorbar(ax3_plot, ax=ax3)
        if belief_points is not None:
            ax3.scatter(belief_points[:,0], belief_points[:,1], s=1, c='red')

        plt.show()


class Solver:
    def __init__(self):
        self._solved = False
        self._solve_run_ts = datetime.datetime.min
        self._solve_steps_count = 0
        self._solve_history = []


    def solve(self):
        pass

    
    @property
    def solution(self) -> ValueFunction:
        assert self._solved, "solve() has to be run first..."
        return self._solve_history[-1]['value_functions']

