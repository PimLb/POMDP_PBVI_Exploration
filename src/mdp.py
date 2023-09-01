import random
from typing import Union

import copy
import datetime
import numpy as np

from src.framework import AlphaVector, ValueFunction

class Model:
    '''
    MDP Model class.

    ...

    Attributes
    ----------
    states: int|list
        A list of state labels or an amount of states to be used.
    actions: int|list
        A list of action labels or an amount of actions to be used.
    transitions:
        The transition matrix, has to be |S| x |A| x |S|. If none is provided, it will be randomly generated.
    immediate_rewards:
        The reward matrix, has to be |S| x |A| x |S|. If provided, it will be use in combination with the transition matrix to fill to expected rewards.
    probabilistic_rewards: bool
        Whether the rewards provided are probabilistic or pure rewards. If probabilist 0 or 1 will be the reward with a certain probability.

    Methods
    -------
    transition(s:int, a:int):
        Returns a random state given a prior state and an action.
    '''
    def __init__(self,
                 states:Union[int, list],
                 actions:Union[int, list],
                 transitions=None,
                 immediate_rewards=None,
                 probabilistic_rewards:bool=False
                 ):
        
        # States
        if isinstance(states, int):
            self.state_labels = [f's_{i}' for i in range(states)]
        else:
            self.state_labels = states
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
        if transitions is None:
            # If no transitiong matrix given, generate random one
            random_probs = np.random.rand(self.state_count, self.action_count, self.state_count)
            # Normalization to have s_p probabilies summing to 1
            self.transition_table = random_probs / np.sum(random_probs, axis=2, keepdims=True)
        else:
            self.transition_table = np.array(transitions)
            assert self.transition_table.shape == (self.state_count, self.action_count, self.state_count), "transitions table doesnt have the right shape, it should be SxAxS"

        # Rewards
        if immediate_rewards is None:
            # If no reward matrix given, generate random one
            self.immediate_reward_table = np.random.rand(self.state_count, self.action_count, self.state_count)
        else:
            self.immediate_reward_table = immediate_rewards
            assert self.immediate_reward_table.shape == (self.state_count, self.action_count, self.state_count), "rewards table doesnt have the right shape, it should be SxAxS"

        # Expected rewards
        self.expected_rewards_table = np.zeros((self.state_count, self.action_count))
        for s in self.states:
            for a in self.actions:
                self.expected_rewards_table[s,a] = np.dot(self.transition_table[s,a,:], self.immediate_reward_table[s,a,:])

        # Rewards are probabilistic
        self.probabilistic_rewards = probabilistic_rewards

    
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


    def solve(self, model: Model) -> tuple[ValueFunction, SolverHistory]:
        # Initiallize V as a |S| x |A| matrix of the reward expected when being in state s and taking action a
        V = ValueFunction([AlphaVector(model.expected_rewards_table[:,a], a) for a in model.actions])
        V_opt = V[0]
        converged = False

        solve_history = SolverHistory(model)
        solve_history.append({'value_function': V})

        while (not converged) and (len(solve_history) <= self.horizon):
            old_V_opt = copy.deepcopy(V_opt)

            V = []
            for a in model.actions:
                alpha_vect = []
                for s in model.states:
                    summer = sum(model.transition_table[s, a, s_p] * old_V_opt[s_p] for s_p in model.states)
                    alpha_vect.append(model.expected_rewards_table[s,a] + (self.gamma * summer))

                V.append(AlphaVector(alpha_vect, a))

            V_opt = np.max(np.array(V), axis=1)

            solve_history.append({'value_function': ValueFunction(V)})
                
            avg_delta = np.max(np.abs(V_opt - old_V_opt))
            if avg_delta < self.eps:
                break

        return ValueFunction(V), solve_history


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