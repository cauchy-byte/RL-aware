import numpy as np
from collections import deque
import copy

class HistoryConstructor:
    """
    state0, action0, state1, action1, ..., state_{max_len}, action_{max_len}, state_{current}
    支持维护最近H个动作的历史
    """
    def __init__(self, max_len, state_dim, action_dim, need_lst_action=False, action_history_length=1,
                 ep_state_history_length=8, ep_action_window_length=10):
        self.max_len = max_len
        self.buffer = deque(maxlen=max_len * 2 + 1)
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 保留原有的lst_action用于兼容性
        self.lst_action = np.zeros((self.action_dim,))
        self.need_lst_action = need_lst_action
        
        self.action_history_length = action_history_length

        self.ep_state_history_length = ep_state_history_length
        self.ep_action_window_length = ep_action_window_length

        self.action_history = deque(maxlen=self.ep_action_window_length)
        
        self.reset()

    def __call__(self, current_state):
        if self.max_len == 0:
            if self.need_lst_action:
                obs_dim = len(np.shape(current_state))
                # 将最近H个动作拼接到状态前面
                # 如果动作历史不足H个，用零动作填充
                action_history_array = self.get_action_history_array()
                if obs_dim == 1:
                    return np.hstack((action_history_array.reshape((-1)), current_state))
                else:
                    return np.hstack((action_history_array.reshape((1, -1)), current_state))
            else:
                return current_state

        self.buffer.append(np.squeeze(current_state))
        return np.hstack(self.buffer)

    def reset(self):
        self.lst_action = np.zeros((self.action_dim, ))
        for i in range(self.max_len):
            self.buffer.append(np.zeros((self.state_dim,)))
            self.buffer.append(np.zeros((self.action_dim,)))
        
        # 重置动作窗口队列，用零动作填充
        self.action_history.clear()
        for _ in range(self.ep_action_window_length):
            self.action_history.append(np.zeros((self.action_dim,)))

    def update_action(self, action):
        # 更新最新的动作（用于兼容性）
        self.lst_action = copy.deepcopy(action)
        # 将新动作添加到历史队列（自动移除最旧的动作）
        self.action_history.append(copy.deepcopy(action))
        
        if self.max_len == 0:
            return
        self.buffer.append(np.squeeze(action))
    
    def get_action_history_array(self):
        """
        获取EP动作窗口数组
        返回形状: (ep_action_window_length * action_dim,)
        动作按时间顺序拼接：[a_{t}, a_{t-1}, ..., a_{t-T+1}]
        """
        # 将deque转换为list并反转，使得最新的动作在前
        actions = list(self.action_history)
        actions.reverse()  # 现在actions[0]是最新的动作a_{t-1}
        
        # 拼接所有动作
        if len(actions) > 0:
            return np.concatenate(actions)
        else:
            # 如果没有动作历史，返回零数组
            return np.zeros((self.ep_action_window_length * self.action_dim,))


if __name__ == '__main__':
    import gymnasium as gym
    env = gym.make('Hopper-v3')
    constructor = HistoryConstructor(4, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    state = constructor(env.reset())
    for _ in range(100):
        np.set_printoptions(suppress=True, threshold=int(1e5), linewidth=150, precision=2)
        print('unified state: ', state)
        action = env.action_space.sample()
        next_state_tmp, reward, done, _ = env.step(action)
        print('state: ', next_state_tmp, ', action: ', action)
        constructor.update_action(action)
        next_state = constructor(next_state_tmp)
        state = next_state
