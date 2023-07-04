import numpy as np
import gym
from gym import spaces
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as clr
# This is the environment you implemented last lab! Nothing to do here.
class CourseEnv(gym.Env):
    """ Gridworld environment from the Course. A 4x3 grid with 2 states in the upper right corner 
    leading to the terminal state.
        """
    def __init__(self):
        self.height = 3
        self.width = 4
        self.action_space = spaces.Discrete(5) # maximum amount of actions possible in a state

        self.observation_space = spaces.Tuple(( # observations come in (x,y) tuples with x:height, y:width.
                spaces.Discrete(self.height),
                spaces.Discrete(self.width)
                ))
        self.moves = { # converts actions to 2D moves
                'north': (-1, 0),
                'east':  (0, 1),
                'south': (1, 0),
                'west' : (0, -1),
                }
        self.moves_list = [ # enables shortcuts in applying noise
                'north',
                'east',
                'south',
                'west',
                ]
        self.noise = .2
        self.start = (2,0)
        self.near_terminals = ((0,3), (1,3)) # arbitrary terminal states
        self.obstacles = [(1,1)]
        self.living_reward = -0.1
        # begin in start state
        self.reset()

    def _move(self, action, state=None):
        """ Moves the agent according to the required action, taking hitboxes into account. """
        dx, dy = self.moves[action]
        if state is None:
            state = self.S
            
        state = state[0] + dx, state[1] + dy

        if state in self.obstacles: # cancel movement
            state = state[0] - dx, state[1] - dy            
        
        # Finally, setting the agent back into the grid if fallen out
        state = (max(0, state[0]), max(0, state[1]))
        state = (min(state[0], self.height - 1),
                 min(state[1], self.width - 1))

        return state

    def step(self, action):
        """ Moves the agent in the action direction.
            """
        if self.S is 'TERMINAL':
            raise ValueError("Trying to step from TERMINAL state.")

        if self.S in self.near_terminals:
            assert action == 'exit', "Non-exit action in state {}".format(self.S)
            if self.S == self.near_terminals[0]: # rewarding state
                return 'TERMINAL', +1, True, {}
            if self.S == self.near_terminals[1]: # punishing state
                return 'TERMINAL', -1, True, {}
            
        assert action in self.moves_list, "invalid action {} in state {}".format(action, self.S)
        # Otherwise, moving according to action.
        # First, maybe apply some noise:
        if np.random.rand() < self.noise: # Apply noise!
            action = self.moves_list[(self.moves_list.index(action)+np.random.choice((-1,1))) % 4]
        
        print("action executed: {}".format(action))
        self.S = self._move(action)
        
        # Returns; anything brings a reward of living
        return self.S, self.living_reward, self.S is 'TERMINAL', {}

    def reset(self):
        self.S = self.start
        return self.S
    
    def available_actions(self, state=None):
        """
        List of available actions in the provided state
        Parameters
        ----------
        state: tuple (position), string ('TERMINAL') or None
            state from which to provide all actions. If None, use the current environment state.
        Returns
        -------
        ret : list
            List of all actions available in the provided state.
        """
        if state is None:
            state = self.S
        if state is 'TERMINAL':
            return []
        if state in self.near_terminals:
            return ['exit']
        
        return self.moves_list
    
    def p(self, state, action):
        """
        Dynamics function p of the MDP in this state and action.
        Parameters
        ----------
        state: tuple (position) or string ('TERMINAL')
            state from which to provide all actions. If the terminal state is provided, raises an error, 
            as there are no dynamics from the terminal state. 
        action: string 
            in list in ['north', 'east', 'south', 'west', 'exit'] with proper state
        Returns
        -------
        ret : dict
            dictionary of (next_state, reward) pairs with: corresponding probabilities
        """
        # Terminal state: return error
        assert state is not 'TERMINAL', "asking for dynamics from terminal state"
        # Near terminal state
        if state in self.near_terminals:
            assert action == 'exit', "Non exit action ({}) in near terminal state".format(action)
            if state == self.near_terminals[0]:
                return {('TERMINAL', +1): 1.}
            if state == self.near_terminals[1]:
                return {('TERMINAL', -1): 1.}
        # Other states: 3 possibilities: normal development of the action doing its job, or noise hindering        
        action_n1 = self.moves_list[(self.moves_list.index(action)-1) % 4] # noise-impacted action 1
        action_n2 = self.moves_list[(self.moves_list.index(action)+1) % 4] # noise-impacted action 2
        d = {}
        # The main problem is to make sure you're not counting the same state-reward pair twice
        # instead of summing the probabilities
        d[(self._move(action, state), self.living_reward)] = 1-self.noise
        sr2 = (self._move(action_n1, state), self.living_reward)
        if sr2 in d.keys():
            d[sr2] += self.noise/2
        else:
            d[sr2] = self.noise/2
        
        sr3 = (self._move(action_n2, state), self.living_reward)
        if sr3 in d.keys():
            d[sr3] += self.noise/2
        else:
            d[sr3] = self.noise/2
        return d
    
    def is_terminal(self, state=None):
        if state is None:
            state = self.S
        return state is 'TERMINAL'
    
    def states(self):
        states = [(x,y) for x in range(self.height) for y in range(self.width) if (x,y) not in self.obstacles]
        states += ['TERMINAL']
        return states
        
    def render(self):
        s = np.zeros((self.height, self.width), dtype=int).astype(str)
        s[self.start] = 'S'
        s[self.obstacles[0]] = 'X'
        s[self.near_terminals[0]] = '+'
        s[self.near_terminals[1]] = '-'
        s[self.S] = '.'
        
        print(self.S)
        print(s)
        print("Available actions: {}".format(self.available_actions()))



class PolicyIteration():
    """ Dynamic Programming Agent that performs Policy Iteration."""

    def __init__(self, mdp):
        self.mdp = mdp # MDP we're trying to solve
        self.gamma = 0.9
        self.states = mdp.states()
        self.V = {i:0 for i in self.states} # values dictionary
        self.states.remove("TERMINAL")
        self.pi = {i:np.random.choice(mdp.available_actions(i)) for i in self.states} # policy dictionary
        self.env_details = {}

        for s in self.states:
            action_dict = {}
            for a in mdp.available_actions(s):
                p = mdp.p(s, a)
                nxt_state_dict = {}
                for i in range(len(p)):
                    nxt_state_dict[list(p)[i][0]] = {"reward": list(p)[i][1],
                                                     "prob": list(p.values())[i]}

                action_dict[a] = nxt_state_dict                
                # action_dict[a] =  {"next_state": [list(p)[i][0] for i in range(len(p))],
                #         "reward": [list(p)[i][1] for i in range(len(p))],
                #         "prob": [list(p.values())[i] for i in range(len(p))]
                #         }
            self.env_details[s] = action_dict
        

        # TODO: initialize your value and policy dictionaries. The policy should be random.
        #raise NotImplementedError("Initialization of the values V and pi dictionaries")
        
        # Plotting. You can use this method in the `run` method as well.
        # print("Initial Value and Policy:")
        # plot_value_policy(self)
    def check(self):
        return self.env_details

    def policy_evaluation(self, delta):
        check = np.inf
        while check > delta:
            check = 0
            for s in self.states:
                v = self.V[s]
                for next_state in self.env_details[s][self.pi[s]].keys():
                    print("Current state: ", s, "Next state: ", next_state)
                    probability = self.env_details[s][self.pi[s]][next_state]["prob"]
                    print("Probability: ", probability)
                    reward = self.env_details[s][self.pi[s]][next_state]["reward"]
                    print("Reward: ", reward)
                    self.V[s] += probability * (reward + self.gamma * self.V[next_state])
                    print("New Value: ", self.V[s])
                diff = max(check, abs(v-self.V[s]))
            print(diff)
            check = diff
        return self.V
               
                
    def policy(self,s):
        return self.pi[s]

    def run(self, delta=1e-5):
        """
        Runs the Policy Iteration algorithm until convergence. 
        The Policy Evaluation steps are run until inf norm < delta.
        Parameters
        ----------
        delta: float
            Precision required to exit Policy Evaluation, in terms of inf norm.
        Returns
        -------
        sweeps : int
            Number of performed sweeps over the state space
        """
        sweeps = 0
        # TODO: Implement the Policy Iteration algorithm.
        raise NotImplementedError("run function of Policy Iteration")
        return sweeps
    

mdp = CourseEnv()

agent = PolicyIteration(mdp)
value = agent.policy_evaluation(0.01)
x = agent.check()
x.keys()
x = [1,2,3]
u = [1,2,3]
x-u
value