# RNN动作历史输入修改说明

## 修改概述

本次修改将ESCP算法中RNN（EP模块）的输入从**"状态 + 上一个动作"**改为**"状态 + 最近H个动作"**，其中H是可配置的超参数。

## 修改动机

根据您提供的图片，新的输入格式为：
- 每个时间步 t: `[s_t, a_{t-1}, a_{t-2}, ..., a_{t-H}]`
- s_t: 当前状态
- a_{t-1}, a_{t-2}, ..., a_{t-H}: 最近H个动作（按时间倒序）

这种设计可以让RNN模块利用更丰富的动作历史信息来推断环境特征。

## 修改文件列表

### 1. **parameter/Parameter.py**
- **新增超参数**: `action_history_length` (默认值: 3)
- **说明**: 控制RNN输入中包含的历史动作数量H
- **用法**: `--action_history_length 3`

### 2. **utils/history_construct.py**
- **新增属性**: `action_history_length`, `action_history`
- **新增方法**: `get_action_history_array()`
- **修改逻辑**: 
  - 维护一个长度为H的动作队列
  - 返回最近H个动作的拼接数组: `[a_{t-1}, a_{t-2}, ..., a_{t-H}]`
  - 形状: `(action_history_length * action_dim,)`

### 3. **models/value.py**
- **修改__init__**: 
  - 新增参数 `action_history_length`
  - EP RNN输入维度从 `obs_dim + act_dim` 改为 `obs_dim + act_dim * action_history_length`
- **修改meta_forward**: 
  - lst_a参数现在包含H个动作，维度为 `[batch, act_dim * action_history_length]`
  - 拼接方式: `torch.cat((x, lst_a), -1)`
- **修改make_config_from_param**: 添加 `action_history_length` 参数

### 4. **models/transition.py**
- **修改内容**: 与value.py相同
- **说明**: Transition模型继承自Value，使用相同的EP模块结构

### 5. **models/policy.py**
- **修改__init__**: 
  - 新增参数 `action_history_length`
  - EP RNN输入维度从 `obs_dim + act_dim` 改为 `obs_dim + act_dim * action_history_length`
  - ep_temp也做相应修改
- **修改meta_forward**: EP模块输入包含H个动作
- **修改tmp_ep_res**: EP模块输入包含H个动作
- **修改make_config_from_param**: 添加 `action_history_length` 参数

### 6. **agent/Agent.py**
- **EnvWorker类修改**:
  - 新增属性 `action_history_length`
  - 初始化HistoryConstructor时传入 `action_history_length` 参数
  - 修改mem.push中的数据切片:
    - 原来: `state[:act_dim]` 提取上一个动作
    - 现在: `state[:action_history_length * act_dim]` 提取最近H个动作

- **EnvRemoteArray类修改**:
  - 新增属性 `action_history_length`
  - 在sample1step和sample1step1env中修改数据切片

### 7. **train.sh**
- **修改路径**: 从 `/home/zby/ESCP-master` 改为 `/home/zby/ESCP-master(扩展状态)`
- **新增变量**: `ACTION_HISTORY_LENGTH=3`
- **新增参数**: `--action_history_length $ACTION_HISTORY_LENGTH`

## 数据流说明

### 原始流程（H=1）:
```
状态: [s_t]  (obs_dim,)
上一个动作: [a_{t-1}]  (act_dim,)
EP输入: [s_t, a_{t-1}]  (obs_dim + act_dim,)
```

### 新流程（H=3）:
```
状态: [s_t]  (obs_dim,)
动作历史: [a_{t-1}, a_{t-2}, a_{t-3}]  (act_dim * 3,)
EP输入: [s_t, a_{t-1}, a_{t-2}, a_{t-3}]  (obs_dim + act_dim * 3,)
```

## 使用方法

### 运行训练脚本:
```bash
cd "/home/zby/ESCP-master(扩展状态)"
./train.sh
```

### 自定义动作历史长度:
在train.sh中修改：
```bash
ACTION_HISTORY_LENGTH=5  # 使用最近5个动作
```

或直接运行python命令：
```bash
python main.py \
    --env_name Walker2d-v2 \
    --min_batch_size 300 \
    --action_history_length 5
```

## 重要说明

### 1. 初始化阶段
- 在episode开始时，动作历史队列会用零动作填充
- 随着episode进行，队列逐渐填充真实的动作

### 2. 维度计算
- EP RNN输入维度 = obs_dim + act_dim × action_history_length
- 例如Walker2d-v2: obs_dim=17, act_dim=6, H=3
  - 原来EP输入维度: 17 + 6 = 23
  - 现在EP输入维度: 17 + 6×3 = 35

### 3. 兼容性
- 设置 `action_history_length=1` 可以回退到原始行为（只使用上一个动作）
- 默认值为3，这是一个平衡的选择

### 4. 内存消耗
- 动作历史长度越大，RNN输入维度越大，网络参数和计算量也会增加
- 建议从小的H值开始实验（如3或5）

## 验证修改

### 代码检查通过
- ✅ Parameter.py: 成功添加action_history_length参数
- ✅ HistoryConstructor: 正确维护H个动作队列
- ✅ Value模型: EP RNN输入维度正确修改
- ✅ Transition模型: EP RNN输入维度正确修改
- ✅ Policy模型: EP RNN输入维度正确修改
- ✅ Agent: 数据切片正确修改
- ✅ train.sh: 路径和参数正确设置

### Linter检查
只有1个可忽略的警告（gym import在测试代码中）

## 技术细节

### HistoryConstructor动作队列实现
```python
# deque自动保持最大长度
self.action_history = deque(maxlen=action_history_length)

# 每次更新动作时，自动移除最旧的动作
self.action_history.append(new_action)

# 获取时反转顺序，使最新的在前
actions = list(self.action_history)
actions.reverse()  # [a_{t-1}, a_{t-2}, ..., a_{t-H}]
```

### RNN输入拼接
```python
# EP模块前向传播
ep_input = torch.cat((state, action_history), -1)
# state: [batch, obs_dim]
# action_history: [batch, act_dim * H]
# ep_input: [batch, obs_dim + act_dim * H]

ep_output, ep_hidden = self.ep.forward(ep_input, ep_hidden)
```

## 下一步建议

1. **实验不同的H值**: 尝试H=1,3,5,7,10，观察性能变化
2. **分析EP学习效果**: 查看EP模块能否更好地识别延迟模式
3. **对比baseline**: 与原始版本（H=1）对比训练曲线
4. **可视化动作历史**: 分析哪些历史动作对环境识别最重要

## 修改日期
2025-11-06

## 修改者
AI Assistant (基于用户需求)




