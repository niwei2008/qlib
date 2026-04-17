import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.eva.alpha import calc_ic
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 初始化 Qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    
    # 定义参数
    start_time = "2018-01-01"
    end_time = "2018-12-31"
    freq = "day"
    
    # 使用 CSI300 指数成分股
    instruments = "csi300"
    
    # 初始化 Alpha158 处理器
    print("初始化 Alpha158 处理器...")
    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        freq=freq
    )
    
    # 获取因子数据和标签数据
    print("获取因子数据和标签数据...")
    data = handler.fetch()
    
    # 分离因子数据和标签数据
    feature_data = data.iloc[:, :-1]  # 因子数据
    label_data = data.iloc[:, -1]     # 标签数据
    
    # 打印数据形状
    print(f"因子数据形状: {feature_data.shape}")
    print(f"标签数据形状: {label_data.shape}")
    print(f"因子数量: {feature_data.shape[1]}")
    
    # 为每个因子单独计算 IC 和 Rank IC
    print("\n计算每个因子的 IC 和 Rank IC...")
    
    # 存储每个因子的平均 IC 和 Rank IC
    avg_ic_dict = {}
    avg_rank_ic_dict = {}
    
    # 存储每个因子的 IC 和 Rank IC 时间序列
    ic_dict = {}
    rank_ic_dict = {}
    
    # 计算每个因子的 IC 和 Rank IC
    for i, factor in enumerate(feature_data.columns):
        print(f"计算因子 {i+1}/{feature_data.shape[1]}: {factor}...")
        # 为每个因子单独计算 IC 和 Rank IC
        factor_data = feature_data[factor]
        ic_factor, rank_ic_factor = calc_ic(factor_data, label_data)
        
        # 存储结果
        ic_dict[factor] = ic_factor
        rank_ic_dict[factor] = rank_ic_factor
        
        # 计算平均 IC 和 Rank IC
        avg_ic_dict[factor] = ic_factor.mean()
        avg_rank_ic_dict[factor] = rank_ic_factor.mean()
    
    # 创建结果 DataFrame
    avg_ic_rank_ic = pd.DataFrame({
        "avg_ic": avg_ic_dict,
        "avg_rank_ic": avg_rank_ic_dict
    })
    
    # 处理可能的 NaN 值
    avg_ic_rank_ic = avg_ic_rank_ic.dropna()
    
    # 按平均 IC 排序并添加排名
    avg_ic_rank_ic = avg_ic_rank_ic.sort_values(by="avg_ic", ascending=False)
    avg_ic_rank_ic["rank_ic"] = range(1, len(avg_ic_rank_ic) + 1)
    
    # 按平均 Rank IC 排序并添加排名
    if not avg_ic_rank_ic.empty:
        avg_ic_rank_ic["rank_rank_ic"] = avg_ic_rank_ic["avg_rank_ic"].rank(ascending=False, method="first").astype(int)
    
    # 打印前 10 个因子的结果
    print("\n每个因子的平均 IC 和 Rank IC (前 10 个因子，按平均 IC 排序):")
    print(avg_ic_rank_ic.head(10))
    
    # 保存结果到文件
    output_file = "all_alpha158_ic_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Alpha158 因子 IC 和 Rank IC 分析结果\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n")
        f.write(f"时间范围: {start_time} 到 {end_time}\n")
        f.write(f"股票池: {instruments}\n")
        f.write(f"因子数量: {feature_data.shape[1]}\n\n")
        
        f.write("## 每个因子的平均 IC 和 Rank IC (按平均 IC 排序)\n")
        f.write(avg_ic_rank_ic.to_string())
        f.write("\n\n")
        
        # 获取前 3 个因子（按平均 IC 排序）
        top_factors = avg_ic_rank_ic.head(3).index.tolist()
        
        # 保存前 3 个因子的 IC 和 Rank IC 时间序列
        f.write("## 前 3 个因子的 IC 分析\n")
        for i, factor in enumerate(top_factors):
            f.write(f"\n### 因子: {factor} (排名: {avg_ic_rank_ic.loc[factor, 'rank_ic']})\n")
            f.write(ic_dict[factor].to_string())
            f.write("\n")
        
        f.write("\n## 前 3 个因子的 Rank IC 分析\n")
        for i, factor in enumerate(top_factors):
            f.write(f"\n### 因子: {factor} (排名: {avg_ic_rank_ic.loc[factor, 'rank_ic']})\n")
            f.write(rank_ic_dict[factor].to_string())
            f.write("\n")
    
    print(f"\n分析完成！结果已输出到文件: {output_file}")
