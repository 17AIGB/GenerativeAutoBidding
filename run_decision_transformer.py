
# 一、数据加载（EpisodeReplayBuffer）：
# 1、读取trajectory_data.csv文件中
# 2、按done字段将数据分割；state, action, reward, done, next_state
# 3、计算采样概率sampler

# 二、数据采样 (EpisodeReplayBuffer.__getitem__)
# 1、随机选择episode和起始位置start_t
# 2、提取K个时间步的序列（默认EpisodeReplayBuffer.K=20）
# 3、EpisodeReplayBuffer.discount_cumsum:计算discounted return-to-go (RTG)
# 4、对序列进行padding（前面补0）和归一化
# 5、返回：states, actions, rewards, dones, rtg, timesteps, mask

# 三、模型训练 (DecisionTransformer.step)
# 前向传播获取预测
# 计算仅针对action的MSE损失
# 梯度裁剪并更新参数

# 四、推理执行 (DecisionTransformer.get_action, DecisionTransformer.take_actions)
# 维护历史序列状态
# 动态更新return-to-go目标
# 预测下一步最优action

# 五、DecisionTransformer模型层级结构
# DecisionTransformer:
# ├── Transformer Encoder (3层)
# │   ├── CausalSelfAttention (多头注意力)
# │   ├── LayerNorm
# │   └── MLP FeedForward
# │
# ├── 嵌入层
# │   ├── embed_state: Linear(state_dim → 64)
# │   ├── embed_action: Linear(act_dim → 64)
# │   ├── embed_return: Linear(1 → 64)
# │   ├── embed_reward: Linear(1 → 64)
# │   └── embed_timestep: Embedding(max_ep_len → 64)
# │
# └── 预测头
#     ├── predict_state: Linear(64 → state_dim)
#     ├── predict_action: Linear(64 → act_dim)
#     └── predict_return: Linear(64 → 1)

# -*- coding: utf-8 -*-
import numpy as np
from bidding_train_env.common.utils import normalize_state, normalize_reward, save_normalize_dict
from bidding_train_env.baseline.dt.utils import EpisodeReplayBuffer
from bidding_train_env.baseline.dt.dt import DecisionTransformer
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging
import pickle

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(filename)s(%(lineno)d)] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_dt():
    train_model()


def train_model():
    state_dim = 16

    replay_buffer = EpisodeReplayBuffer(16, 1, "./data/trajectory/trajectory_data.csv")
    save_normalize_dict({"state_mean": replay_buffer.state_mean, "state_std": replay_buffer.state_std},
                        "saved_model/DTtest")
    logger.info(f"Replay buffer size: {len(replay_buffer.trajectories)}")

    model = DecisionTransformer(state_dim=state_dim, act_dim=1, state_mean=replay_buffer.state_mean,
                                state_std=replay_buffer.state_std)

    # 根据数据集大小调整训练步数
    num_trajectories = len(replay_buffer.trajectories)
    if num_trajectories < 50:
        step_num = 1000
        batch_size = 8
    elif num_trajectories < 500:
        step_num = 2000
        batch_size = 16
    else:
        step_num = 10000
        batch_size = 32

    logger.info(f"Trajectories: {num_trajectories}, Steps: {step_num}, Batch size: {batch_size}")

    sampler = WeightedRandomSampler(replay_buffer.p_sample, num_samples=step_num * batch_size, replacement=True)
    dataloader = DataLoader(replay_buffer, sampler=sampler, batch_size=batch_size)

    model.train()
    i = 0
    for states, actions, rewards, dones, rtg, timesteps, attention_mask in dataloader:
        train_loss = model.step(states, actions, rewards, dones, rtg, timesteps, attention_mask)
        i += 1
        if i % 50 == 0:
            logger.info(f"Step: {i} Action loss: {np.mean(train_loss)}")
        model.scheduler.step()

    model.save_net("saved_model/DTtest")
    # import pdb
    # pdb.set_trace()
    test_state = np.ones(state_dim, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")
    logger.info(f"训练完成! 总步数: {i}")


def load_model():
    """
    加载模型。
    """
    with open('./Model/DT/saved_model/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)
    model = DecisionTransformer(state_dim=16, act_dim=1, state_mean=normalize_dict["state_mean"],
                                state_std=normalize_dict["state_std"])
    model.load_net("Model/DT/saved_model")
    test_state = np.ones(16, dtype=np.float32)
    logger.info(f"Test action: {model.take_actions(test_state)}")


if __name__ == "__main__":
    run_dt()

