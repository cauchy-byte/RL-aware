from collections import deque
import gymnasium as gym
import numpy as np
import copy
from gymnasium.utils import seeding

from envs.delayed_distribution import DoubleGaussianDistribution, GammaDistribution, UniformDistribution


class NonstationaryDelayedEnv(gym.Wrapper):
    """
    非平稳延迟环境包装器
    结合了观察延迟和元强化学习的特性，延迟分布和延迟参数会随着步数动态变化
    """

    # 支持的延迟分布类型
    DELAY_DISTRIBUTIONS = ['gamma', 'uniform', 'doublegaussian']

    def __init__(self, env,
                 initial_delay_type="gamma",
                 delay_changing_period=1000,  # 延迟分布变化的周期
                 delay_changing_interval=100,  # 延迟分布变化的间隔
                 delay_type_changing_period=10000,  # 延迟类型变化的周期（仅类型，不含参数）
                 max_delay_range=(3, 10),  # 最大延迟的范围
                 initial_action=None,
                 skip_initial_actions=False,
                 random_delay_per_episode=True):  # 每个episode是否随机延迟
        """
        初始化非平稳延迟环境

        参数:
            env: 基础环境
            initial_delay_type: 初始延迟分布类型 ('gamma', 'uniform', 'doublegaussian')
            delay_changing_period: 延迟分布完整变化周期的步数
            delay_changing_interval: 每次检查是否变化延迟分布的间隔步数
            delay_type_changing_period: 延迟类型（delay_type）变化的周期步数
            max_delay_range: 最大延迟的范围 (min_delay, max_delay)
            initial_action: 初始动作
            skip_initial_actions: 是否跳过初始动作
        """
        super().__init__(env)

        self.wrapped_env = env
        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions

        # 延迟分布变化的参数
        self.delay_changing_period = delay_changing_period
        self.delay_changing_interval = delay_changing_interval
        self.delay_type_changing_period = delay_type_changing_period
        self.max_delay_range = max_delay_range

        # 每个episode是否随机延迟
        self.random_delay_per_episode = random_delay_per_episode

        # 初始化延迟分布
        self.current_delay_type = initial_delay_type
        self.current_max_delay = np.random.randint(max_delay_range[0], max_delay_range[1] + 1)
        self.obs_delay_dis = self._create_delay_distribution(initial_delay_type, self.current_max_delay)

        # 延迟任务列表（用于非平稳变化）
        self.delay_tasks = None
        self.delay_task_index = 0
        self._last_delay_type_index = None
        self._delay_type_tasks = None
        self._explicit_delay_task_active = False

        # 初始化缓冲区
        self.past_observations = deque(maxlen=max_delay_range[1] + 1)
        self.arrival_times_observations = deque(maxlen=max_delay_range[1] + 1)

        # 时间和奖励追踪
        self.t = 0
        self.done_signal_sent = False
        self.cum_rew_actor = 0.  # actor实际获得的累积奖励
        self.cum_rew_brain = 0.  # agent观察到的累积奖励
        self.cur_step_ind = 0  # 当前步数索引

        # 动作空间归一化参数
        self.min_action = env.action_space.low
        self.max_action = env.action_space.high
        self.range_action = self.max_action - self.min_action

    def _create_delay_distribution(self, delay_type, max_delay):
        """
        创建延迟分布对象

        参数:
            delay_type: 延迟类型 ('gamma', 'uniform', 'doublegaussian')
            max_delay: 最大延迟步数

        返回:
            延迟分布对象
        """
        if delay_type == "gamma":
            return GammaDistribution(max_delay=min(max_delay, 6))  # gamma分布最大支持6
        elif delay_type == "uniform":
            return UniformDistribution(max_delay=min(max_delay, 9))  # uniform分布最大支持9
        elif delay_type == "doublegaussian":
            return DoubleGaussianDistribution(max_delay=min(max_delay, 10))  # doublegaussian最大支持10
        else:
            raise ValueError(f"不支持的延迟类型: {delay_type}")

    def _find_env_with_attr(self, attr_name):
        """
        递归查找具有指定属性或方法的环境

        参数:
            attr_name: 要查找的属性或方法名

        返回:
            具有该属性的环境对象，如果找不到则返回 None
        """
        # 首先检查 wrapped_env
        if hasattr(self, 'wrapped_env'):
            env = self.wrapped_env
            if hasattr(env, attr_name):
                return env
            # 递归查找
            if hasattr(env, 'env') or hasattr(env, 'wrapped_env'):
                current = env
                while current is not None:
                    if hasattr(current, attr_name):
                        return current
                    # 尝试下一层
                    if hasattr(current, 'wrapped_env'):
                        current = current.wrapped_env
                    elif hasattr(current, 'env'):
                        current = current.env
                    else:
                        break

        # 然后检查 env
        if hasattr(self, 'env'):
            env = self.env
            if hasattr(env, attr_name):
                return env
            # 递归查找
            if hasattr(env, 'env') or hasattr(env, 'wrapped_env'):
                current = env
                while current is not None:
                    if hasattr(current, attr_name):
                        return current
                    # 尝试下一层
                    if hasattr(current, 'wrapped_env'):
                        current = current.wrapped_env
                    elif hasattr(current, 'env'):
                        current = current.env
                    else:
                        break

        return None

    def normalization(self, action):
        """
        将动作归一化到 [-1, 1] 范围

        参数:
            action: 原始动作

        返回:
            归一化后的动作
        """
        return (action - self.min_action) / self.range_action * 2 - 1

    def denormalization(self, action):
        """
        将归一化的动作反归一化到原始范围

        参数:
            action: 归一化的动作 ([-1, 1] 范围)

        返回:
            原始范围的动作
        """
        return (action + 1) / 2 * self.range_action + self.min_action

    def sample_tasks(self, n_tasks, **kwargs):
        """
        采样物理参数任务（代理到被包装的 NonstationaryEnv）

        对于纯延迟环境（不包含物理参数非平稳），返回空任务列表
        对于包含 NonstationaryEnv 的包装链，代理到 NonstationaryEnv

        参数:
            n_tasks: 任务数量
            **kwargs: 传递给 NonstationaryEnv.sample_tasks 的其他参数

        返回:
            任务列表
        """
        # 递归查找有 sample_tasks 方法的环境
        env_with_method = self._find_env_with_attr('sample_tasks')
        if env_with_method is not None:
            return env_with_method.sample_tasks(n_tasks, **kwargs)
        else:
            # 纯延迟环境没有物理参数任务，返回空任务列表
            # 这是正常情况：延迟环境不需要物理参数变化
            return [None] * n_tasks

    def sample_delay_tasks(self, n_tasks):
        """
        采样延迟任务

        参数:
            n_tasks: 任务数量

        返回:
            延迟任务列表，每个任务包含 (delay_type, max_delay)
        """
        tasks = []
        for _ in range(n_tasks):
            # 随机选择延迟类型
            delay_type = np.random.choice(self.DELAY_DISTRIBUTIONS)

            # 根据延迟类型限制最大延迟
            if delay_type == "gamma":
                max_delay = np.random.randint(3, 7)  # gamma: 3-6
            elif delay_type == "uniform":
                max_delay = np.random.randint(3, 10)  # uniform: 3-9
            else:  # doublegaussian
                max_delay = np.random.randint(3, 11)  # doublegaussian: 3-10

            tasks.append({
                'delay_type': delay_type,
                'max_delay': max_delay
            })
        return tasks

    def sample_delay_type_tasks(self, n_tasks):
        """
        采样仅包含延迟类型的任务列表（用于类型独立变化）

        参数:
            n_tasks: 任务数量

        返回:
            延迟类型列表
        """
        return [np.random.choice(self.DELAY_DISTRIBUTIONS) for _ in range(n_tasks)]

    def set_task(self, task):
        """
        设置物理参数任务（代理到被包装的 NonstationaryEnv）

        参数:
            task: 物理参数任务
        """
        # 递归查找有 set_task 方法的环境
        env_with_method = self._find_env_with_attr('set_task')
        if env_with_method is not None:
            env_with_method.set_task(task)
        else:
            raise AttributeError(f"在环境包装链中找不到 set_task 方法")

    def _update_buffer_maxlen(self, max_delay):
        """
        更新缓冲区大小以适应新的最大延迟

        参数:
            max_delay: 新的最大延迟步数
        """
        new_maxlen = max(self.max_delay_range[1] + 1, max_delay + 1)
        if self.past_observations.maxlen != new_maxlen:
            old_obs = list(self.past_observations)
            old_times = list(self.arrival_times_observations)
            self.past_observations = deque(old_obs, maxlen=new_maxlen)
            self.arrival_times_observations = deque(old_times, maxlen=new_maxlen)

    def set_delay_task(self, task):
        """
        设置当前延迟任务

        参数:
            task: 延迟任务字典，包含 'delay_type' 和 'max_delay'
        """
        delay_type = task['delay_type']
        max_delay = task['max_delay']

        # 更新当前延迟分布
        self.current_delay_type = delay_type
        self.current_max_delay = max_delay
        self.obs_delay_dis = self._create_delay_distribution(delay_type, max_delay)
        self._explicit_delay_task_active = True

        # 更新缓冲区大小
        self._update_buffer_maxlen(max_delay)

    def set_nonstationary_para(self, setting_env_params, changine_period, changing_interval):
        """
        设置物理参数非平稳变化（代理到被包装的 NonstationaryEnv）

        参数:
            setting_env_params: 环境参数列表
            changine_period: 变化周期
            changing_interval: 变化间隔
        """
        # 递归查找有 set_nonstationary_para 方法的环境
        env_with_method = self._find_env_with_attr('set_nonstationary_para')
        if env_with_method is not None:
            env_with_method.set_nonstationary_para(setting_env_params, changine_period, changing_interval)
        else:
            raise AttributeError(f"在环境包装链中找不到 set_nonstationary_para 方法")

    def set_nonstationary_delay_para(self, delay_tasks, changing_period, changing_interval, delay_type_changing_period=None):
        """
        设置非平稳延迟参数（支持参数与类型独立变化）

        参数:
            delay_tasks: 延迟任务列表（包含 delay_type 和 max_delay）
            changing_period: 延迟参数（max_delay）变化周期
            changing_interval: 参数变化检查间隔
            delay_type_changing_period: 延迟类型（delay_type）变化周期，None时禁用类型独立变化
        """
        self.delay_tasks = delay_tasks
        self.delay_changing_period = changing_period
        self.delay_changing_interval = changing_interval
        self.delay_type_changing_period = delay_type_changing_period
        self.delay_task_index = 0
        self._last_delay_type_index = None

        # 如果启用了类型独立变化，则预采样独立的类型任务列表
        if delay_type_changing_period is not None:
            n_type_tasks = max(3, len(delay_tasks))
            self._delay_type_tasks = self.sample_delay_type_tasks(n_type_tasks)
        else:
            self._delay_type_tasks = None

    def reset_nonstationary_delay(self):
        """重置非平稳延迟参数"""
        self.delay_tasks = None
        self.delay_changing_period = None
        self.delay_changing_interval = None
        self.delay_type_changing_period = None
        self.delay_task_index = 0
        self._last_delay_type_index = None
        self._delay_type_tasks = None
        self._explicit_delay_task_active = False

    def reset(self, **kwargs):
        """重置环境"""
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.done_signal_sent = False
        self.cur_step_ind = 0

        # 每个episode开始时随机选择延迟参数（如果启用）
        if self.random_delay_per_episode and not self._explicit_delay_task_active:
            # 随机选择延迟类型
            self.current_delay_type = np.random.choice(self.DELAY_DISTRIBUTIONS)
            # 随机选择最大延迟（在范围内）
            self.current_max_delay = np.random.randint(
                self.max_delay_range[0], 
                self.max_delay_range[1] + 1
            )
            # 创建新的延迟分布
            self.obs_delay_dis = self._create_delay_distribution(
                self.current_delay_type, 
                self.current_max_delay
            )

        # 使用新的gymnasium API
        if hasattr(super(), 'reset'):
            result = super().reset(**kwargs)
            if isinstance(result, tuple):
                first_observation, info = result
            else:
                first_observation = result
                info = {}
        else:
            first_observation = self.env.reset(**kwargs)
            info = {}

        # 填充缓冲区
        self.t = - (self.obs_delay_dis.max_delay + 1)  # 这个值 <= -1
        while self.t <= 0:  # 注意：包含 t=0，以确保有观察可用
            self.send_observation((first_observation, 0., False, False, {}, 0))
            self.t += 1

        # 现在 self.t == 1，但我们要获取 t=0 时的观察
        self.t = 0
        received_observation, *_ = self.receive_observation()
        return received_observation, info

    def seed(self, seed=None):
        """设置随机种子"""
        # 直接调用unwrapped环境的seed方法
        if hasattr(self.unwrapped, 'seed'):
            return self.unwrapped.seed(seed)
        # 如果环境没有seed方法，尝试设置_np_random
        if hasattr(self, 'np_random'):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]
        return None

    def step(self, action):
        """执行一步动作"""
        self.cur_step_ind += 1

        # =========================================================
        # 延迟参数变化：每 delay_changing_interval 步检查一次，
        # 按 delay_changing_period 节奏改变 max_delay
        # =========================================================
        if (self.delay_tasks is not None and
                self.delay_changing_interval > 0 and
                self.cur_step_ind % self.delay_changing_interval == 0):
            param_ind = (self.cur_step_ind // self.delay_changing_period) % len(self.delay_tasks)
            param_task = self.delay_tasks[param_ind]

            # =========================================================
            # 延迟类型变化：每 delay_type_changing_period 步改变一次 delay_type
            # 类型和参数独立变化，互不干扰
            # =========================================================
            if (self.delay_type_changing_period is not None and
                    self._delay_type_tasks is not None and
                    self.cur_step_ind % self.delay_type_changing_period == 0):
                type_ind = (self.cur_step_ind // self.delay_type_changing_period) % len(self._delay_type_tasks)
                new_delay_type = self._delay_type_tasks[type_ind]
                new_max_delay = param_task['max_delay']

                if (new_delay_type != self.current_delay_type or
                        new_max_delay != self.current_max_delay):
                    self.current_delay_type = new_delay_type
                    self.current_max_delay = new_max_delay
                    self.obs_delay_dis = self._create_delay_distribution(new_delay_type, new_max_delay)
                    self._update_buffer_maxlen(new_max_delay)
                    self._last_delay_type_index = type_ind
                    self.delay_task_index = param_ind
            else:
                # 仅 max_delay 变化，delay_type 保持不变
                if param_task['max_delay'] != self.current_max_delay:
                    self.current_max_delay = param_task['max_delay']
                    self.obs_delay_dis = self._create_delay_distribution(self.current_delay_type, self.current_max_delay)
                    self._update_buffer_maxlen(self.current_max_delay)
                    self.delay_task_index = param_ind

        # 执行动作并获取观察
        true_obs = action
        if self.done_signal_sent:
            # 如果已经完成，发送之前的观察
            self.send_observation(self.past_observations[0])
        else:
            # 执行动作
            result = self.env.step(action)
            if len(result) == 4:
                # 旧版gym API
                m, r, done, info = result
                terminated = done
                truncated = False
            else:
                # 新版gymnasium API
                m, r, terminated, truncated, info = result

            true_obs = m
            self.cum_rew_actor += r
            self.done_signal_sent = terminated or truncated
            self.send_observation((m, self.cum_rew_actor, terminated, truncated, info, 0))

        # 接收延迟的观察
        m, cum_rew_actor_delayed, terminated, truncated, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed

        self.t += 1

        # 在info中添加延迟信息
        info['current_delay_type'] = self.current_delay_type
        info['current_max_delay'] = self.current_max_delay

        return m, r, terminated, truncated, info

    def send_observation(self, obs):
        """
        发送观察并添加延迟

        参数:
            obs: 观察元组 (observation, reward, terminated, truncated, info, _)
        """
        alpha = self.obs_delay_dis.dis_sample()  # 采样延迟步数
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        """
        接收延迟后的观察

        返回:
            延迟后的观察元组
        """
        # 找到第一个已经到达的观察（arrival_time <= current_time）
        # 使用 default 参数来处理没有符合条件的情况
        alpha = next((i for i, t in enumerate(self.arrival_times_observations) if t <= self.t), None)

        # 如果找不到已到达的观察，使用最早到达的（索引最大的，因为是 appendleft）
        if alpha is None:
            alpha = len(self.arrival_times_observations) - 1 if len(self.arrival_times_observations) > 0 else 0

        m, r, terminated, truncated, info, _ = self.past_observations[alpha]
        return m, r, terminated, truncated, info

    @property
    def delay_parameter_vector(self):
        """
        获取当前延迟参数向量（用于元强化学习）

        返回:
            包含延迟类型和最大延迟的向量
        """
        # 将延迟类型编码为one-hot向量
        delay_type_encoding = np.zeros(len(self.DELAY_DISTRIBUTIONS))
        delay_type_idx = self.DELAY_DISTRIBUTIONS.index(self.current_delay_type)
        delay_type_encoding[delay_type_idx] = 1.0

        # 归一化最大延迟
        normalized_max_delay = (self.current_max_delay - self.max_delay_range[0]) / \
                              (self.max_delay_range[1] - self.max_delay_range[0])

        return np.concatenate([delay_type_encoding, [normalized_max_delay]])

    @property
    def delay_parameter_length(self):
        """获取延迟参数向量的长度"""
        return len(self.DELAY_DISTRIBUTIONS) + 1  # delay_type (one-hot) + max_delay

    @property
    def env_parameter_length(self):
        """
        获取物理环境参数向量的长度（代理到被包装的 NonstationaryEnv）
        """
        # 递归查找有 env_parameter_length 属性的环境
        env_with_attr = self._find_env_with_attr('env_parameter_length')
        if env_with_attr is not None:
            return env_with_attr.env_parameter_length
        else:
            return 0

    @property
    def env_parameter_vector(self):
        """
        获取环境参数向量（包含物理参数和延迟分布，用于 WMCL Loss 计算）
        """
        # 获取物理参数
        env_with_attr = self._find_env_with_attr('env_parameter_vector')
        if env_with_attr is not None:
            physics_params = env_with_attr.env_parameter_vector
        else:
            physics_params = np.array([], dtype=np.float32)
        
        # 获取延迟分布
        delay_params = self.delay_distribution_vector
        
        # 合并：如果物理参数为空，只返回延迟分布
        if len(physics_params) == 0:
            return delay_params
        else:
            return np.concatenate([physics_params, delay_params])

    @property
    def delay_distribution_vector(self):
        """
        获取延迟分布的概率向量，用于 WMCL Loss 计算
        返回: [K] 概率向量，固定为11维（0-10延迟步数的概率）
        """
        if hasattr(self, 'obs_delay_dis') and self.obs_delay_dis is not None:
            if hasattr(self.obs_delay_dis, 'dis_probability'):
                probs = self.obs_delay_dis.dis_probability()
                # 确保返回11维向量
                result = np.zeros(11, dtype=np.float32)
                result[:len(probs)] = probs
                return result
        return np.zeros(11, dtype=np.float32)  # 默认返回零向量

    @property
    def delay_distribution_dim(self):
        """
        获取延迟分布的维度
        """
        return 11  # 固定为11维

    @property
    def _elapsed_steps(self):
        """
        获取当前环境已执行的步数
        """
        return self.cur_step_ind

    @property
    def _max_episode_steps(self):
        """
        获取最大episode步数
        """
        if hasattr(self.env, '_max_episode_steps'):
            return self.env._max_episode_steps
        elif hasattr(self.wrapped_env, '_max_episode_steps'):
            return self.wrapped_env._max_episode_steps
        else:
            return 1000  # 默认值


if __name__ == '__main__':
    # 测试代码
    import gymnasium as gym

    print("=== 测试非平稳延迟环境 ===")

    # 创建环境
    try:
        base_env = gym.make('Hopper-v4')
    except:
        base_env = gym.make('Hopper-v4')

    env = NonstationaryDelayedEnv(
        base_env,
        initial_delay_type="gamma",
        delay_changing_period=1000,
        delay_changing_interval=100
    )
    
    # 采样延迟任务
    delay_tasks = env.sample_delay_tasks(10)
    print(f"采样了 {len(delay_tasks)} 个延迟任务:")
    for i, task in enumerate(delay_tasks):
        print(f"  任务 {i}: {task}")
    
    # 设置非平稳延迟
    env.set_nonstationary_delay_para(delay_tasks, changing_period=1000, changing_interval=100)
    
    # 重置环境
    obs, info = env.reset()
    print(f"\n初始观察形状: {obs.shape}")
    print(f"初始延迟类型: {env.current_delay_type}")
    print(f"初始最大延迟: {env.current_max_delay}")
    print(f"延迟参数向量: {env.delay_parameter_vector}")
    print(f"延迟参数长度: {env.delay_parameter_length}")
    
    # 运行一些步骤
    print("\n=== 运行环境步骤 ===")
    total_reward = 0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 每100步打印一次信息
        if step % 100 == 0:
            print(f"\n步数 {step}:")
            print(f"  当前延迟类型: {info.get('current_delay_type', 'N/A')}")
            print(f"  当前最大延迟: {info.get('current_max_delay', 'N/A')}")
            print(f"  累积奖励: {total_reward:.2f}")
        
        if terminated or truncated:
            print(f"\nEpisode 在步数 {step} 结束")
            obs, info = env.reset()
            total_reward = 0
    
    print("\n=== 测试完成 ===")
