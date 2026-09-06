"""
环境包装器
根据配置选择使用物理参数非平稳或延迟非平稳
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.nonstationary_env import NonstationaryEnv
from envs.nonstationary_delayed_env import NonstationaryDelayedEnv


ACDA_TASKS = (
    {"task_id": 1, "delay_process": "ge1_23"},
    {"task_id": 2, "delay_process": "ge4_32"},
    {"task_id": 3, "delay_process": "mm1"},
)


def create_env_decoration(parameter):
    """
    根据参数创建环境装饰器函数
    
    参数:
        parameter: Parameter对象，包含环境配置
    
    返回:
        装饰器函数，接受(env, log_scale_limit, rand_params)参数
    """
    use_delay = getattr(parameter, 'use_delay', False)
    
    if use_delay:
        
        # 只使用延迟非平稳环境（不使用物理参数非平稳）
        initial_delay_type = getattr(parameter, 'initial_delay_type', 'gamma')
        max_delay_range_min = getattr(parameter, 'max_delay_range_min', 3)
        max_delay_range_max = getattr(parameter, 'max_delay_range_max', 10)
        nonstationary_delay = getattr(parameter, 'nonstationary_delay', False)
        delay_task_num = getattr(parameter, 'delay_task_num', 10)
        delay_changing_period = getattr(parameter, 'delay_changing_period', 500)
        delay_changing_interval = getattr(parameter, 'delay_changing_interval', 500)
        delay_type_changing_period = getattr(parameter, 'delay_type_changing_period', 10000)
        # 每个episode是否随机延迟参数
        random_delay_per_episode = getattr(parameter, 'random_delay_per_episode', True)
        delay_process = getattr(parameter, 'delay_process', 'legacy')
        delay_task_strategy = getattr(parameter, 'delay_task_strategy', 'random')

        if delay_task_strategy == 'acda_cycle' and delay_process == 'legacy':
            raise ValueError('acda_cycle requires a concrete delay_process')

        def delay_decoration(env, log_scale_limit=3.0, rand_params=None):
            """
            延迟环境装饰器：只使用NonstationaryDelayedEnv
            注意：log_scale_limit和rand_params参数被保留是为了接口兼容，但不会被使用
            """
            # 直接包装延迟环境
            env = NonstationaryDelayedEnv(
                env,
                initial_delay_type=initial_delay_type,
                max_delay_range=(max_delay_range_min, max_delay_range_max),
                random_delay_per_episode=random_delay_per_episode,
                delay_process=delay_process,
            )

            # 如果需要非平稳延迟，采样延迟任务并设置
            # 注意：这里只是初始化，实际的任务设置会在Agent中进行
            if delay_task_strategy == 'acda_cycle':
                env._delay_tasks = [dict(task) for task in ACDA_TASKS]
                env._delay_task_strategy = delay_task_strategy
            elif delay_process != 'legacy':
                env._delay_tasks = env.sample_delay_tasks(delay_task_num)
            elif nonstationary_delay:
                delay_tasks = env.sample_delay_tasks(delay_task_num)
                # 保存delay_tasks供后续使用
                env._delay_tasks = delay_tasks
                env._delay_changing_period = delay_changing_period
                env._delay_changing_interval = delay_changing_interval
                env._delay_type_changing_period = delay_type_changing_period

            return env

        return delay_decoration
    else:
        # 只使用物理参数非平稳环境（不使用延迟）
        def nonstationary_decoration(env, log_scale_limit=3.0, rand_params=None):
            if rand_params is None:
                rand_params = []
            return NonstationaryEnv(env, log_scale_limit=log_scale_limit, rand_params=rand_params)
        
        return nonstationary_decoration
