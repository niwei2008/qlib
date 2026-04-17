"""
Alpha158 因子 IC 和 Rank IC 分析脚本（优化版）
"""
import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
import pandas as pd
import numpy as np


def calculate_daily_ic(factor_series, label_series):
    """计算单日 IC 和 Rank IC"""
    # 确保索引一致
    common_index = factor_series.dropna().index.intersection(label_series.dropna().index)
    
    if len(common_index) < 2:
        return np.nan, np.nan
    
    factor_clean = factor_series.loc[common_index]
    label_clean = label_series.loc[common_index]
    
    # 移除 NaN 值
    valid_mask = ~(factor_clean.isna() | label_clean.isna())
    factor_clean = factor_clean[valid_mask]
    label_clean = label_clean[valid_mask]
    
    if len(factor_clean) < 2:
        return np.nan, np.nan
    
    # 计算 IC 和 Rank IC
    ic = factor_clean.corr(label_clean, method='pearson')
    rank_ic = factor_clean.rank().corr(label_clean.rank(), method='spearman')
    
    return ic, rank_ic


def analyze_factor(factor_name, feature_data, label_data, dates):
    """分析单个因子的 IC"""
    daily_ic = []
    daily_rank_ic = []
    
    for date in dates:
        try:
            # 获取当天数据
            factor_day = feature_data.xs(date, level='datetime')[factor_name]
            label_day = label_data.xs(date, level='datetime')
            
            # 计算 IC
            ic, rank_ic = calculate_daily_ic(factor_day, label_day)
            
            if not np.isnan(ic) and not np.isnan(rank_ic):
                daily_ic.append(ic)
                daily_rank_ic.append(rank_ic)
        except Exception as e:
            print(f"  日期 {date} 处理出错: {e}")
            continue
    
    if daily_ic:
        return {
            'avg_ic': np.mean(daily_ic),
            'std_ic': np.std(daily_ic),
            'pos_ic_ratio': np.mean(np.array(daily_ic) > 0),
            'avg_rank_ic': np.mean(daily_rank_ic),
            'std_rank_ic': np.std(daily_rank_ic),
            'pos_rank_ic_ratio': np.mean(np.array(daily_rank_ic) > 0),
        }
    return None


if __name__ == '__main__':
    # 初始化 Qlib
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    
    # 参数配置
    start_time = "2018-01-01"
    end_time = "2018-12-31"
    freq = "day"
    instruments = "csi300"
    
    print("初始化 Alpha158 处理器...")
    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        freq=freq
    )
    
    data = handler.fetch()
    feature_data = data.iloc[:, :-1]
    label_data = data.iloc[:, -1]
    dates = feature_data.index.get_level_values('datetime').unique().sort_values()
    
    print(f"因子数量: {feature_data.shape[1]}")
    print(f"日期数量: {len(dates)}")
    print(f"日期范围: {dates.min()} 到 {dates.max()}")
    
    # 计算所有因子 IC
    results = []
    for i, factor in enumerate(feature_data.columns):
        result = analyze_factor(factor, feature_data, label_data, dates)
        if result:
            result['factor'] = factor
            results.append(result)
        print(f"[{i+1}/{feature_data.shape[1]}] 完成因子: {factor}")
    
    # 保存结果
    output_file = "alpha158_ic_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Alpha158 因子 IC 和 Rank IC 分析结果\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n")
        f.write(f"时间范围: {start_time} 到 {end_time}\n")
        f.write(f"股票池: {instruments}\n")
        f.write(f"因子数量: {len(results)}\n\n")
        
        f.write("因子名称,平均 IC,IC 标准差,IC>0 比例,平均 Rank IC,Rank IC 标准差,Rank IC>0 比例\n")
        for r in results:
            f.write(f"{r['factor']},{r['avg_ic']:.6f},{r['std_ic']:.6f},{r['pos_ic_ratio']:.4f},"
                   f"{r['avg_rank_ic']:.6f},{r['std_rank_ic']:.6f},{r['pos_rank_ic_ratio']:.4f}\n")
    
    print(f"\n分析完成！结果已保存到: {output_file}")
    print(f"共计算了 {len(results)} 个因子")
