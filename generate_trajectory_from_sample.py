#!/usr/bin/env python3
"""
从样本traffic数据生成trajectory数据
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bidding_train_env.dataloader.rl_data_generator import RlDataGenerator
import pandas as pd

def generate_sample_trajectory():
    """从样本数据生成trajectory数据"""
    print("开始从样本数据生成trajectory数据...")

    # 使用样本数据文件夹
    file_folder_path = "./data/traffic"

    # 生成RL数据
    data_loader = RlDataGenerator(file_folder_path=file_folder_path)

    # 只处理样本文件
    import glob
    csv_files = glob.glob(os.path.join(file_folder_path, '*-sample.csv'))

    if not csv_files:
        print("错误: 没有找到样本数据文件 (*-sample.csv)")
        return

    print(f"找到样本文件: {csv_files}")

    # 输出到trajectory文件夹
    os.makedirs("./data/trajectory", exist_ok=True)

    training_data_list = []
    for csv_path in csv_files:
        print(f"\n开始处理文件: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"原始数据行数: {len(df)}")
        print(f"广告主数量: {df['advertiserNumber'].nunique()}")

        df_processed = data_loader._generate_rl_data(df)
        print(f"处理后数据行数: {len(df_processed)}")

        csv_filename = os.path.basename(csv_path)
        trainData_filename = csv_filename.replace('.csv', '-trajectory.csv')
        trainData_path = os.path.join("./data/trajectory", trainData_filename)
        df_processed.to_csv(trainData_path, index=False)
        print(f"保存到: {trainData_path}")

        training_data_list.append(df_processed)
        del df, df_processed
        print(f"处理文件成功: {csv_path}")

    if training_data_list:
        # 合并所有数据
        combined_dataframe = pd.concat(training_data_list, axis=0, ignore_index=True)
        combined_dataframe_path = os.path.join("./data/trajectory", "trajectory_data.csv")
        combined_dataframe.to_csv(combined_dataframe_path, index=False)
        print(f"\n整合数据成功; 保存至: {combined_dataframe_path}")
        print(f"总行数: {len(combined_dataframe)}")
        print(f"广告主数量: {combined_dataframe['advertiserNumber'].nunique()}")

if __name__ == '__main__':
    generate_sample_trajectory()
