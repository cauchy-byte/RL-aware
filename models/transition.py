from models.rnn_base import RNNBase
from models.value import Value
import torch


class Trainsition(Value):
    def __init__(self, obs_dim, act_dim, up_hidden_size, up_activations, up_layer_type,
                 ep_hidden_size, ep_activation, ep_layer_type, ep_dim, use_gt_env_feature,
                 logger=None, freeze_ep=False, enhance_ep=False, stop_pg_for_ep=False, action_history_length=1,
                 ep_state_history_length=8, ep_action_window_length=10):
        super(Value, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_history_length = action_history_length  # 动作历史长度H
        self.ep_state_history_length = ep_state_history_length
        self.ep_action_window_length = ep_action_window_length
        self.use_gt_env_feature = use_gt_env_feature
        # aux dim: we add ep to every layer inputs.
        aux_dim = ep_dim if enhance_ep else 0
        # UP模块输入：状态 + EP输出 + 当前动作 -> 输出：下一状态
        self.up = RNNBase(obs_dim + ep_dim + act_dim, obs_dim, up_hidden_size, up_activations, up_layer_type, logger, aux_dim)
        # EP模块输入：状态 + 动作窗口（长度T）
        self.ep = RNNBase(obs_dim + act_dim * ep_action_window_length, ep_dim, ep_hidden_size, ep_activation, ep_layer_type, logger)
        self.ep_rnn_count = self.ep.rnn_num
        self.up_rnn_count = self.up.rnn_num
        # ep first, up second
        self.module_list = torch.nn.ModuleList(self.up.total_module_list + self.ep.total_module_list)
        self.min_log_std = -7.0
        self.max_log_std = 2.0
        self.sample_hidden_state = None
        self.ep_tensor = None
        self.freeze_ep = freeze_ep
        self.enhance_ep = enhance_ep
        self.stop_pg_for_ep = stop_pg_for_ep


    def forward(self, x, lst_a, a, h, ep_out=None):
        next_x_delta, h_out = self.meta_forward(x, lst_a, a, h, ep_out=ep_out)
        next_x = next_x_delta + x
        return next_x, h_out