import numpy as np
from action import Action

def dummy(env):
    return 0

class QCell:
    discount_factor = 0.99
    learning_rate   = 0.5

    def __init__(self, num_actions, module):
        self._q_value = np.zeros(num_actions)
        self._parent_module = module

    def qValue(self, action: Action):
        return self._q_value[action.value]

    # def temporalDifference(self, next_q_cell, action_index, env):
    #     agent_loc = env.agent.loc

    #     immediate_reward = self._parent_module.reward_func(env)
    #     future_reward    = QCell.discount_factor * max(next_q_cell.q_value)
    #     applied_q_val    = self._q_value[action_index]

    #     return immediate_reward + future_reward - applied_q_val

    # def updateQValue(self, action_index, next_q_cell, env):
    #     TD = self.temporalDifference(next_q_cell, action_index, env)
    #     self._q_value[action_index] += QCell.learning_rate * TD

    # def updateFatalQValue(self, action_index, penalty):
    #     TD = penalty - self.qValue(action_index)
    #     self._q_value[action_index] += QCell.learning_rate * TD


class Module:
    count = 0
    fatal_cost = -100

    def __init__(self, num_states, actions, name = None):
        # Count the number of existing modules
        Module.count += 1

        # Create the list of states for the Q-Learning module
        num_actions = len(actions)
        self.q_table = [np.zeros(num_actions) for ii in range(num_states + 1)]

        # How this modules weighs compared to others
        self.weight = 1

        # The reward function for this module
        self.reward_func = dummy
        self.state_func  = dummy

        # Label for the module
        if name is None:
            self.name = f"Module {Module.count}"
        else:
            self.name = name

    def temporalDifference(self, start_index, end_index, action_index, env):
        immediate_reward = self.reward_func(env)
        future_reward    = QCell.discount_factor * max(self.q_table[end_index])
        applied_q_val    = self.q_table[start_index][action_index]

        return immediate_reward + future_reward - applied_q_val

    def updateQValue(self, start_index, end_index, action_index, env):
        # Calculate the temporal difference
        immediate_reward = self.reward_func(env)
        future_reward    = QCell.discount_factor * max(self.q_table[end_index])
        applied_q_val    = self.q_table[start_index][action_index]

        TD = immediate_reward + future_reward - applied_q_val

        # Update the qValue
        self.q_table[start_index][action_index] += QCell.learning_rate * TD


