import copy
import numpy as np
import math
import random

from typing import Union

from src.POMDP.model  import Model
from src.POMDP.belief import Belief
from src.POMDP.alpha_vector import AlphaVector
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
    def __init__(self, model:Model):
        self.model = model


    def backup(self, belief_set:list[Belief], alpha_set:list[AlphaVector], discount_factor:float=0.9):
        '''
        This function has purpose to update the set of alpha vectors. It does so in 3 steps:
        1. It creates projections from each alpha vector for each possible action and each possible observation
        2. It collapses this set of generated alpha vectors by taking the weighted sum of the alpha vectors weighted by the observation probability and this for each action and for each belief.
        3. Then it further collapses the set to take the best alpha vector and action per belief
        In the end we have a set of alpha vectors as large as the amount of beliefs.

        The alpha vectors are also pruned to avoid duplicates and remove dominated ones.

                Parameters:
                        belief_set (list[Belief]): The belief set to use to generate the new alpha vectors with.
                        alpha_set (list[AlphaVector]): The alpha vectors to generate the new set from.
                        discount_factor (float): The discount factor used for training (also known as gamma), default: 0.9.

                Returns:
                        new_alpha_set (list[AlphaVector]): A list of updated alpha vectors.
        '''
        
        # Step 1
        gamma_a_o_t = {}
        for a in self.model.actions:
            for o in self.model.states:
                alpa_a_o_set = []
                
                for alpha_i in alpha_set:
                    alpa_a_o_vect = []
                    
                    for s in self.model.states:
                        products = [(self.model.transition_table[s,a,s_p] * self.model.observation_table[s,a,o] * alpha_i[s_p])  # In the paper it also shows s_p for the observation table
                                    for s_p in self.model.states]
                        alpa_a_o_vect.append(discount_factor * sum(products))
                        
                    alpa_a_o_set.append(alpa_a_o_vect)
                    
                if a not in gamma_a_o_t:
                    gamma_a_o_t[a] = {o: alpa_a_o_set}
                else:
                    gamma_a_o_t[a][o] = alpa_a_o_set

        # Step 2
        alpha_set_t = []
        alpha_actions = []
        
        for b in belief_set:
            
            best_alpha = None
            best_action = None
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
                    best_alpha = alpha_a_vect
                    best_action = a

            # Pruning step computationally expensive and saves very few vectors
            dominated = False
            # for other_vect in alpha_set_t:
            #     if all(np.array(best_alpha) < np.array(other_vect)):
            #         print(f'dominated by {other_vect}')
            #         dominated = True
            #     if all(np.array(best_alpha) > np.array(other_vect)):
            #         print(f'removing: {other_vect}')
            #         alpha_set_t.remove(other_vect)
            
            if not arreq_in_list(best_alpha, alpha_set_t) and not dominated:
                alpha_set_t.append(best_alpha)
                alpha_actions.append(best_action)
            else:
                # print('prune')
                pass
                
        return (alpha_set_t, alpha_actions)
    
    
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
            o = self.model.observe(s_p, a) # Weird to use s_p here
            b_new = b.update(a, o)
            
            belief_set_new.append(b_new)
            
        return belief_set_new
    

    def expand_ssga(self, belief_set:list[Belief], alpha_set:list[AlphaVector], eps:float=0.1) -> list[Belief]:
        '''
        Stochastic Simulation with Greedy Action.
        Simulates running a single-step forward from the beliefs in the "belief_set".
        The step forward is taking assuming we are in a random state s (weighted by the belief),
         then taking the best action a based on the belief with probability 'epsilon'.
        These lead to a new state s_p and a observation o.
        From this action a and observation o we can update our belief. 

                Parameters:
                        belief_set (list[Belief]): list of beliefs to expand on.
                        alpha_set (list[AlphaVector]): Used to find the best action knowing the belief.
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
                best_alpha_index = np.argmax([np.dot(alpha, b) for alpha in alpha_set])
                a = alpha_set[best_alpha_index].action
            
            s_p = self.model.transition(s, a)
            o = self.model.observe(s_p, a)
            b_new = b.update(a, o)
            
            belief_set_new.append(b_new)
            
        return belief_set_new
    

    def expand_ssea(self, belief_set:list[Belief], alpha_set:list[AlphaVector]) -> list[Belief]:
        '''
        Stochastic Simulation with Exploratory Action.
        Simulates running steps forward for each possible action knowing we are a state s, chosen randomly with according to the belief probability.
        These lead to a new state s_p and a observation o for each action.
        From all these and observation o we can generate updated beliefs. 
        Then it takes the belief that is furthest away from other beliefs, meaning it explores the most the belief space.

                Parameters:
                        belief_set (list[Belief]): list of beliefs to expand on.
                        alpha_set (list[AlphaVector]): Used to find the best action knowing the belief.

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
    

    def expand_ger(self, belief_set, alpha_set) -> list[Belief]:
        '''
        Greedy Error Reduction

                Parameters:
                        belief_set (list[Belief]): list of beliefs to expand on.
                        alpha_set (list[AlphaVector]): Used to find the best action knowing the belief.
                        alpha_actions (list[int]): Used to find the best action knowing the belief.

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
            return self.expand_ssra(**kwargs)
        elif expand_function in 'expand_ssga':
            return self.expand_ssga(**kwargs)
        elif expand_function in 'expand_ssea':
            return self.expand_ssea(**kwargs)
        elif expand_function in 'expand_ger':
            return self.expand_ger(**kwargs)
        else:
            return []


    def solve(self, expansions:int, horizon:int, expand_function:str='ssea', initial_belief=None) -> list[AlphaVector]:
        '''
        Main loop of the Point-Based Value Iteration algorithm.
        It consists in 2 steps, Backup and Expand.
        1. Backup: Updates the alpha vectors based on the current belief set
        2. Expand: Expands the belief set base with a expansion strategy given by the parameter expand_function

                Parameters:
                        expansions (int): How many times the algorithm has to expand the belief set. (the size will be doubled every time, eg: for 5, the belief set will be of size 32)
                        horizon (int): How many times the alpha vector set must be updated every time the belief set is expanded.
                
                Returns:
                        alpha_set (list[AlphaVector]): The alpha vectors approximating the value function.

        '''
        if initial_belief is None:
            uni_prob = np.ones(self.model.state_count) / self.model.state_count
            initial_belief = Belief(self.model, uni_prob)
        else:
            initial_belief = Belief(self.model, np.array(initial_belief))
        
        belief_set = [initial_belief]
        alpha_set = [AlphaVector(self.model.reward_table[:,a], a) for a in self.model.actions]

        for _ in range(expansions):
            for _ in range(horizon):
                alpha_set, alpha_actions = self.backup(belief_set, alpha_set)

            belief_set = self.expand(belief_set=belief_set, alpha_set=alpha_set)

        return alpha_set
