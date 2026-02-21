#!/usr/bin/env python3
"""
测试训练好的Decision Transformer模型
"""
import sys
import os
import pickle
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bidding_train_env.baseline.dt.dt import DecisionTransformer

def load_and_test_model():
    """加载并测试模型"""
    print("=" * 60)
    print("加载训练好的模型...")
    print("=" * 60)

    # 加载归一化参数
    with open('./saved_model/DTtest/normalize_dict.pkl', 'rb') as f:
        normalize_dict = pickle.load(f)

    print(f"State mean: {normalize_dict['state_mean']}")
    print(f"State std: {normalize_dict['state_std']}")

    # 加载模型
    state_dim = 16
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=1,
        state_mean=normalize_dict["state_mean"],
        state_std=normalize_dict["state_std"]
    )
    model.load_net("saved_model/DTtest/dt.pt")
    model.eval()

    print("\n模型加载成功！")
    print("=" * 60)

    # 加载测试数据
    print("\n加载测试数据...")
    test_data = pd.read_csv("./data/trajectory/trajectory_data.csv")

    # 解析state
    import ast
    test_data["state"] = test_data["state"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    print(f"测试数据行数: {len(test_data)}")
    print(f"广告主数量: {test_data['advertiserNumber'].nunique()}")

    # 测试几个状态
    print("\n" + "=" * 60)
    print("测试模型推理...")
    print("=" * 60)

    # 随机选择一个广告主的所有状态进行连续测试
    advertiser_id = test_data['advertiserNumber'].iloc[0]
    advertiser_data = test_data[test_data['advertiserNumber'] == advertiser_id].sort_values('timeStepIndex')

    print(f"\n测试广告主 {advertiser_id} 的连续决策序列:")
    print(f"总步数: {len(advertiser_data)}")

    # 初始化评估状态
    model.init_eval()

    total_error = 0
    test_count = min(10, len(advertiser_data))

    for i in range(test_count):
        row = advertiser_data.iloc[i]
        state = np.array(row['state'], dtype=np.float32)
        true_action = row['action']
        true_reward = row['reward']

        # 模型推理
        with torch.no_grad():
            if i == 0:
                action = model.take_actions(state)
            else:
                prev_row = advertiser_data.iloc[i-1]
                prev_reward = prev_row['reward']
                action = model.take_actions(state, pre_reward=prev_reward)

        error = abs(true_action - action[0])
        total_error += error

        print(f"\n步骤 {i+1}:")
        print(f"  状态: timeleft={state[0]:.4f}, bgtleft={state[1]:.4f}")
        print(f"  真实action: {true_action:.6f}")
        print(f"  预测action: {action[0]:.6f}")
        print(f"  真实reward: {true_reward:.6f}")
        print(f"  误差: {error:.6f}")

    avg_error = total_error / test_count
    print(f"\n平均误差: {avg_error:.6f}")

    print("\n" + "=" * 60)
    print("测试多个广告主...")
    print("=" * 60)

    # 测试多个广告主
    all_errors = []
    num_advertisers_to_test = min(3, test_data['advertiserNumber'].nunique())
    advertiser_ids = test_data['advertiserNumber'].unique()[:num_advertisers_to_test]

    for adv_idx, advertiser_id in enumerate(advertiser_ids, 1):
        advertiser_data = test_data[test_data['advertiserNumber'] == advertiser_id].sort_values('timeStepIndex')

        print(f"\n广告主 {adv_idx}/{num_advertisers_to_test}: ID={advertiser_id}")
        print(f"总步数: {len(advertiser_data)}")

        # 初始化评估状态
        model.init_eval()

        for i in range(len(advertiser_data)):
            row = advertiser_data.iloc[i]
            state = np.array(row['state'], dtype=np.float32)
            true_action = row['action']
            true_reward = row['reward']

            # 模型推理
            with torch.no_grad():
                if i == 0:
                    action = model.take_actions(state)
                else:
                    prev_row = advertiser_data.iloc[i-1]
                    prev_reward = prev_row['reward']
                    action = model.take_actions(state, pre_reward=prev_reward)

            error = abs(true_action - action[0])
            all_errors.append(error)

    # 统计总体误差
    if all_errors:
        print("\n" + "=" * 60)
        print("总体统计:")
        print("=" * 60)
        print(f"平均绝对误差(MAE): {np.mean(all_errors):.6f}")
        print(f"均方根误差(RMSE): {np.sqrt(np.mean(np.array(all_errors)**2)):.6f}")
        print(f"最大误差: {np.max(all_errors):.6f}")
        print(f"最小误差: {np.min(all_errors):.6f}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    load_and_test_model()
