# 非平稳延迟环境集成指南

## 概述

本文档说明如何将`NonstationaryDelayedEnv`集成到ESCP-master项目中，用元强化学习处理延迟环境。

## 主要特性

### 1. 非平稳延迟环境 (`NonstationaryDelayedEnv`)

新的环境包装器`NonstationaryDelayedEnv`具有以下特性：

- **观察延迟**: 模拟真实世界中的传感器延迟
- **动态延迟分布**: 支持三种延迟分布（Gamma、Uniform、DoubleGaussian）
- **延迟参数随时间变化**: 延迟分布类型和最大延迟会随训练步数动态变化
- **元强化学习支持**: 提供延迟任务采样和参数向量编码
- **与原有NonstationaryEnv兼容**: 可以同时使用物理参数变化和延迟变化

### 2. 改进的延迟分布类

更新了三个延迟分布类，使其支持动态的最大延迟参数：

- `GammaDistribution`: 延迟范围 1-6 步
- `UniformDistribution`: 延迟范围 1-9 步  
- `DoubleGaussianDistribution`: 延迟范围 1-10 步

每个分布类现在都支持：
- 动态设置最大延迟
- 自动归一化概率分布
- 更灵活的采样机制

## 文件结构

```
ESCP-master/
├── envs/
│   ├── delayed_env.py                    # 原始延迟环境（保留）
│   ├── delayed_distribution.py           # 更新：支持动态参数的延迟分布
│   ├── nonstationary_env.py             # 原始非平稳环境
│   └── nonstationary_delayed_env.py     # 新增：非平稳延迟环境
├── examples/
│   └── example_nonstationary_delayed_env.py  # 使用示例
└── DELAYED_ENV_INTEGRATION.md           # 本文档
```

## 使用方法

### 方法1: 单独使用延迟环境

```python
import gym
from envs.nonstationary_delayed_env import NonstationaryDelayedEnv

# 创建基础环境
base_env = gym.make('Hopper-v2')

# 创建非平稳延迟环境
env = NonstationaryDelayedEnv(
    base_env,
    initial_delay_type="gamma",      # 初始延迟分布类型
    delay_changing_period=1000,      # 变化周期（步数）
    delay_changing_interval=100,     # 检查间隔（步数）
    max_delay_range=(3, 10)          # 最大延迟范围
)

# 采样延迟任务
delay_tasks = env.sample_delay_tasks(n_tasks=10)

# 设置非平稳延迟
env.set_nonstationary_delay_para(
    delay_tasks=delay_tasks,
    changing_period=1000,
    changing_interval=100
)

# 训练
obs, info = env.reset()
for step in range(1000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### 方法2: 结合物理参数非平稳环境使用

```python
from envs.nonstationary_env import NonstationaryEnv
from envs.nonstationary_delayed_env import NonstationaryDelayedEnv

# 创建基础环境
base_env = gym.make('Hopper-v2')

# 添加物理参数非平稳包装器
physics_env = NonstationaryEnv(
    base_env,
    rand_params=['gravity', 'body_mass', 'dof_damping'],
    log_scale_limit=3.0
)

# 添加延迟非平稳包装器
delayed_env = NonstationaryDelayedEnv(
    physics_env,
    initial_delay_type="uniform",
    delay_changing_period=1000,
    delay_changing_interval=100
)

# 采样任务
physics_tasks = physics_env.sample_tasks(10)
delay_tasks = delayed_env.sample_delay_tasks(10)

# 设置非平稳参数
physics_env.set_nonstationary_para(physics_tasks, 1000, 100)
delayed_env.set_nonstationary_delay_para(delay_tasks, 1000, 100)

# 现在环境同时具有物理参数变化和延迟变化
```

### 方法3: 在SAC算法中集成

修改`algorithms/sac.py`中的环境创建部分：

```python
# 原来的代码
self.env = NonstationaryEnv(
    gym.make(self.parameter.env_name), 
    log_scale_limit=self.parameter.env_default_change_range,
    rand_params=self.parameter.varying_params
)

# 修改为
from envs.nonstationary_delayed_env import NonstationaryDelayedEnv

base_env = gym.make(self.parameter.env_name)
self.env = NonstationaryEnv(
    base_env,
    log_scale_limit=self.parameter.env_default_change_range,
    rand_params=self.parameter.varying_params
)
# 添加延迟包装器
self.env = NonstationaryDelayedEnv(
    self.env,
    initial_delay_type="gamma",
    delay_changing_period=self.parameter.delay_changing_period,
    delay_changing_interval=self.parameter.delay_changing_interval
)
```

### 方法4: 在Agent中集成

修改`agent/Agent.py`中的`EnvWorker`初始化：

```python
# 在EnvWorker.__init__中
if env_decoration is not None:
    # 添加物理参数非平稳包装器
    self.env = env_decoration(
        self.env, 
        log_scale_limit=default_change_range
    )
    
    # 添加延迟包装器（如果需要）
    if parameter.use_delay_wrapper:
        from envs.nonstationary_delayed_env import NonstationaryDelayedEnv
        self.env = NonstationaryDelayedEnv(
            self.env,
            initial_delay_type=parameter.delay_type,
            delay_changing_period=parameter.delay_changing_period,
            delay_changing_interval=parameter.delay_changing_interval
        )
```

## 元强化学习支持

### 延迟参数向量

环境提供延迟参数的向量表示，可用作meta-learning的context：

```python
# 获取延迟参数向量
param_vector = env.delay_parameter_vector
# 向量格式: [is_gamma, is_uniform, is_doublegaussian, normalized_max_delay]

# 获取参数向量维度
param_dim = env.delay_parameter_length  # 返回 4
```

### 结合物理参数

如果同时使用物理参数和延迟参数：

```python
# 获取完整的context向量
physics_context = physics_env.env_parameter_vector
delay_context = delayed_env.delay_parameter_vector
full_context = np.concatenate([physics_context, delay_context])

# 完整context维度
total_dim = physics_env.env_parameter_length + delayed_env.delay_parameter_length
```

### 在策略网络中使用

```python
# 修改Policy配置以包含延迟参数
policy_config = {
    'ep_dim': env_parameter_length + delay_parameter_length,
    # ... 其他配置
}

# 在forward中拼接context
def forward(self, obs, physics_context, delay_context):
    full_context = torch.cat([physics_context, delay_context], dim=-1)
    # ... 使用full_context
```

## 配置参数说明

### Parameter配置文件需要添加的参数

在`parameter/Parameter.py`中添加以下参数：

```python
# 延迟环境相关参数
self.use_delay_wrapper = True  # 是否使用延迟包装器
self.delay_type = 'gamma'  # 初始延迟分布类型
self.delay_changing_period = 1000  # 延迟变化周期
self.delay_changing_interval = 100  # 延迟变化间隔
self.max_delay_range = (3, 10)  # 最大延迟范围
self.delay_task_num = 10  # 延迟任务数量
```

## 测试和验证

运行示例脚本验证集成：

```bash
cd ESCP-master
python examples/example_nonstationary_delayed_env.py
```

运行单元测试（如果需要创建）：

```bash
python -m pytest tests/test_nonstationary_delayed_env.py
```

## 常见问题

### Q1: 如何选择延迟分布类型？

- **Gamma**: 短延迟为主，适合模拟网络延迟
- **Uniform**: 均匀分布，延迟不确定性高
- **DoubleGaussian**: 双峰分布，适合模拟有两种模式的延迟

### Q2: 如何调整延迟变化频率？

通过`delay_changing_interval`控制检查频率：
- 值越小，检查越频繁，延迟变化越平滑
- 值越大，延迟保持时间越长，变化越突然

通过`delay_changing_period`控制完整周期：
- 值越小，完整周期越短，任务切换越快
- 值越大，每个任务持续时间越长

### Q3: 延迟环境会影响训练速度吗？

会有轻微影响，因为需要维护延迟缓冲区和计算延迟采样。但影响很小，通常可以忽略。

### Q4: 可以只使用延迟环境而不使用物理参数变化吗？

可以。直接在基础环境上包装`NonstationaryDelayedEnv`即可，不需要`NonstationaryEnv`。

### Q5: 如何在训练中监控延迟变化？

```python
obs, reward, terminated, truncated, info = env.step(action)
current_delay_type = info['current_delay_type']
current_max_delay = info['current_max_delay']
# 记录到logger中
logger.add_scalar('delay/type', delay_type_to_int(current_delay_type), step)
logger.add_scalar('delay/max_delay', current_max_delay, step)
```

## 实验建议

### 实验1: 固定延迟分布训练

```python
# 不设置非平稳参数，使用固定的延迟分布
env = NonstationaryDelayedEnv(base_env, initial_delay_type="gamma")
# 直接训练，延迟分布不变
```

### 实验2: 非平稳延迟训练

```python
# 设置非平稳参数，延迟随时间变化
delay_tasks = env.sample_delay_tasks(10)
env.set_nonstationary_delay_para(delay_tasks, 1000, 100)
# 训练时延迟分布会自动变化
```

### 实验3: Meta-Learning设置

```python
# 使用MAML或其他meta-learning算法
# 每个task对应一个特定的延迟配置
for task in delay_tasks:
    env.set_delay_task(task)
    # 内循环训练
    for step in range(inner_steps):
        # ... 训练
    # 收集梯度并meta-update
```

## 性能优化建议

1. **缓冲区大小**: 根据实际需要设置`max_delay_range`，避免不必要的大缓冲区
2. **变化频率**: 合理设置`delay_changing_interval`，过于频繁的变化可能影响学习
3. **并行训练**: 使用`ray`并行多个环境实例时，每个实例独立维护延迟状态
4. **Context编码**: 在策略网络中使用延迟参数向量作为额外输入，帮助快速适应

## 下一步工作

1. 添加可视化工具，展示延迟分布随时间的变化
2. 实现更多延迟分布类型（如Exponential、Poisson等）
3. 添加自适应延迟调整机制
4. 创建延迟环境的benchmark任务
5. 与RMDM等方法结合进行实验

## 参考资料

- 原始延迟环境论文: [Addressing Signal Delay in Deep RL]
- ESCP论文: [Environment-Specific Contextual Policy]
- Meta-RL综述: [A Survey of Meta-Reinforcement Learning]










