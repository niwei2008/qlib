import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':
    # 初始化 Qlib
    # 使用 os.path.expanduser 处理 ~ 路径，确保在不同操作系统上都能正确解析
    provider_uri = os.path.expanduser("~/.qlib/qlib_data/cn_data")
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    
    # 定义参数
    start_time = "2018-01-01"  # 分析开始时间
    end_time = "2018-12-31"    # 分析结束时间
    freq = "day"              # 数据频率
    
    # 使用 CSI300 指数成分股
    instruments = "csi300"
    
    # 初始化 Alpha158 处理器
    print("初始化 Alpha158 处理器...")
    print(f"配置参数: instruments={instruments}, start_time={start_time}, end_time={end_time}, freq={freq}")
    handler = Alpha158(
        instruments=instruments,  # 股票池
        start_time=start_time,    # 开始时间
        end_time=end_time,        # 结束时间
        freq=freq                 # 数据频率
    )
    print("Alpha158 处理器初始化成功！")
    
    # 获取因子数据和标签数据
    print("获取因子数据和标签数据...")
    try:
        print(f"开始获取数据，时间范围: {start_time} 到 {end_time}")
        data = handler.fetch()  # 生成包含因子值和标签数据
        print("数据获取成功！")
    except Exception as e:
        print(f"获取数据时出错: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # 分离因子数据和标签数据
    feature_data = data.iloc[:, :-1]  # 因子数据（所有列除了最后一列）
    label_data = data.iloc[:, -1]     # 标签数据（最后一列）
    
    # 打印数据基本信息
    print(f"因子数据形状: {feature_data.shape}")
    print(f"标签数据形状: {label_data.shape}")
    print(f"因子数量: {feature_data.shape[1]}")
    
    # 打印数据索引信息
    print(f"因子数据索引类型: {type(feature_data.index)}")
    print(f"因子数据索引级别: {feature_data.index.names}")
    
    # 打印前几行数据，了解数据结构
    print("\n因子数据前几行:")
    print(feature_data.head())
    print("\n标签数据前几行:")
    print(label_data.head())
    
    # 打印日期范围
    dates = feature_data.index.get_level_values('datetime').unique()
    print(f"\n日期数量: {len(dates)}")
    print(f"日期范围: {dates.min()} 到 {dates.max()}")
    
    # 测试获取数据，验证数据结构和访问方式
    print("\n测试获取数据...")
    if len(dates) > 0:
        test_date = dates[0]
        print(f"测试日期: {test_date}")
        try:
            test_factor = 'KMID'  # 选择一个测试因子
            # 获取当天的因子和标签数据
            print(f"尝试获取因子 {test_factor} 在日期 {test_date} 的数据...")
            factor_day = feature_data.loc[pd.IndexSlice[test_date, :], test_factor]
            # 提取因子数据的 instrument 索引
            factor_instruments = factor_day.index.get_level_values('instrument')
            print(f"因子数据形状: {factor_day.shape}")
            print(f"因子数据前几行: {factor_day.head()}")
            print(f"因子数据是否包含 NaN: {factor_day.isna().any()}")
            print(f"因子数据 NaN 数量: {factor_day.isna().sum()}")
            
            print(f"尝试获取标签数据在日期 {test_date} 的数据...")
            label_day = label_data.loc[pd.IndexSlice[test_date, :]]
            # 提取标签数据的 instrument 索引
            label_instruments = label_day.index.get_level_values('instrument')
            print(f"标签数据形状: {label_day.shape}")
            print(f"标签数据前几行: {label_day.head()}")
            print(f"标签数据是否包含 NaN: {label_day.isna().any()}")
            print(f"标签数据 NaN 数量: {label_day.isna().sum()}")
            
            # 确保索引匹配，避免计算时的索引不一致问题
            print("计算共同索引...")
            # 使用 instrument 作为共同索引
            common_instruments = set(factor_instruments) & set(label_instruments)
            print(f"共同索引数量: {len(common_instruments)}")
            
            # 重新索引因子和标签数据
            if len(common_instruments) > 0:
                # 转换为列表并排序，确保顺序一致
                common_instruments = sorted(list(common_instruments))
                # 重新索引因子数据（因子数据有多层索引）
                factor_day = factor_day.loc[pd.IndexSlice[test_date, common_instruments]]
                # 重新索引标签数据（标签数据可能只有单层索引）
                try:
                    # 尝试使用多层索引
                    label_day = label_day.loc[pd.IndexSlice[test_date, common_instruments]]
                except Exception:
                    # 如果失败，使用单层索引
                    label_day = label_day.loc[common_instruments]
            
            if len(common_instruments) > 0:
                
                # 确保没有 NaN 值，避免相关性计算错误
                # 分别检查因子和标签数据的 NaN 值
                factor_valid = ~factor_day.isna()
                label_valid = ~label_day.isna()
                
                # 确保索引一致
                common_indices = factor_day.index.intersection(label_day.index)
                if len(common_indices) > 1:
                    # 重新索引到共同的索引
                    factor_day_common = factor_day.loc[common_indices]
                    label_day_common = label_day.loc[common_indices]
                    
                    # 再次检查 NaN 值
                    valid_mask = (~factor_day_common.isna()) & (~label_day_common.isna())
                    valid_count = valid_mask.sum()
                    print(f"有效数据点数量: {valid_count}")
                    
                    if valid_count > 1:  # 至少需要 2 个有效数据点才能计算相关性
                        # 计算 Pearson 相关系数 (IC) - 衡量因子值与收益的线性相关程度
                        ic = factor_day_common[valid_mask].corr(label_day_common[valid_mask], method='pearson')
                        print(f"IC: {ic:.4f}")
                        
                        # 计算 Spearman 秩相关系数 (Rank IC) - 衡量因子排序与收益排序的相关程度
                        rank_ic = factor_day_common[valid_mask].rank().corr(label_day_common[valid_mask].rank(), method='spearman')
                        print(f"Rank IC: {rank_ic:.4f}")
                    else:
                        print("有效数据点数量不足，无法计算 IC 和 Rank IC")
                else:
                    print("共同索引数量不足，无法计算 IC 和 Rank IC 1")
            else:
                print("共同索引数量不足，无法计算 IC 和 Rank IC 2")
        except Exception as e:
            print(f"获取数据时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 打开输出文件，准备写入分析结果
    output_file = "alpha158_ic_analysis.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        # 写入分析开始信息
        f.write("# Alpha158 因子 IC 和 Rank IC 分析结果\n")
        f.write(f"分析时间: {pd.Timestamp.now()}\n")
        f.write(f"时间范围: {start_time} 到 {end_time}\n")
        f.write(f"股票池: {instruments}\n")
        f.write(f"因子数量: {feature_data.shape[1]}\n\n")
        
        # 计算每个因子的 IC 和 Rank IC
        f.write("## 因子 IC 和 Rank IC 分析\n")
        f.write("因子名称,平均 IC,IC 标准差,IC>0 比例,平均 Rank IC,Rank IC 标准差,Rank IC>0 比例\n")
        
        # 按日期分组计算 IC
        dates = feature_data.index.get_level_values('datetime').unique()
        print(f"\n计算 IC 和 Rank IC...")
        print(f"日期数量: {len(dates)}")
        
        # 存储每个因子的 IC 和 Rank IC 时间序列
        factor_ic_dict = {}      # 存储每个因子的 IC 时间序列
        factor_rank_ic_dict = {} # 存储每个因子的 Rank IC 时间序列
        
        # 遍历所有因子，计算每个因子的 IC 和 Rank IC
        # 注释：如果因子数量太多，可修改为只计算部分因子
        # factors_to_analyze = feature_data.columns  # 分析所有因子
        factors_to_analyze = ['KMID', 'KLEN', 'KMID2']  # 仅分析部分因子
        
        for i, factor in enumerate(factors_to_analyze):
            daily_ic = []       # 存储该因子每天的 IC
            daily_rank_ic = []  # 存储该因子每天的 Rank IC
            print(f"\n计算因子 {i+1}/{len(factors_to_analyze)}: {factor}...")
            
            # 遍历所有日期，计算每天的 IC 和 Rank IC
            # 注释：如果日期太多，可修改为只计算部分日期
            for date in dates[:5]:  # 仅测试前 5 个日期
            # for date in dates:  # 分析所有日期
                try:
                    # 获取当天的因子和标签数据
                    # print(f"  处理日期: {date}")  # 注释掉，减少输出
                    factor_day = feature_data.loc[pd.IndexSlice[date, :], factor]
                    label_day = label_data.loc[pd.IndexSlice[date, :]]
                    
                    # 提取 instrument 索引
                    factor_instruments = factor_day.index.get_level_values('instrument')
                    label_instruments = label_day.index.get_level_values('instrument')
                    
                    # 确保索引匹配
                    common_instruments = set(factor_instruments) & set(label_instruments)
                    
                    if len(common_instruments) > 1:  # 至少需要 2 个有效数据点
                        # 转换为列表并排序，确保顺序一致
                        common_instruments = sorted(list(common_instruments))
                        # 重新索引因子数据（因子数据有多层索引）
                        factor_day = factor_day.loc[pd.IndexSlice[date, common_instruments]]
                        # 重新索引标签数据（标签数据可能只有单层索引）
                        try:
                            # 尝试使用多层索引
                            label_day = label_day.loc[pd.IndexSlice[date, common_instruments]]
                        except Exception:
                            # 如果失败，使用单层索引
                            label_day = label_day.loc[common_instruments]
                        
                        # 确保没有 NaN 值
                        # 确保索引一致
                        common_indices = factor_day.index.intersection(label_day.index)
                        if len(common_indices) > 1:
                            # 重新索引到共同的索引
                            factor_day_common = factor_day.loc[common_indices]
                            label_day_common = label_day.loc[common_indices]
                            
                            # 再次检查 NaN 值
                            valid_mask = (~factor_day_common.isna()) & (~label_day_common.isna())
                            valid_count = valid_mask.sum()
                            
                            if valid_count > 1:  # 至少需要 2 个有效数据点
                                # 计算 Pearson 相关系数 (IC)
                                ic = factor_day_common[valid_mask].corr(label_day_common[valid_mask], method='pearson')
                                daily_ic.append(ic)
                                
                                # 计算 Spearman 秩相关系数 (Rank IC)
                                rank_ic = factor_day_common[valid_mask].rank().corr(label_day_common[valid_mask].rank(), method='spearman')
                                daily_rank_ic.append(rank_ic)
                            
                            # print(f"  日期 {date}: IC={ic:.4f}, Rank IC={rank_ic:.4f}")  # 注释掉，减少输出
                        # else:
                        #     print(f"  有效数据点数量不足，跳过")  # 注释掉，减少输出
                    # else:
                    #     print(f"  共同索引数量不足，跳过")  # 注释掉，减少输出
                except Exception as e:
                    # 跳过不存在的日期或其他错误
                    # print(f"  日期 {date} 出错: {e}")  # 注释掉，减少输出
                    # import traceback
                    # traceback.print_exc()  # 注释掉，减少输出
                    pass
            
            # 计算该因子的平均 IC 和 Rank IC
            if daily_ic:
                avg_ic = np.mean(daily_ic)           # 平均 IC
                std_ic = np.std(daily_ic)           # IC 标准差（衡量稳定性）
                pos_ic_ratio = np.mean(np.array(daily_ic) > 0)  # IC>0 的比例
                
                avg_rank_ic = np.mean(daily_rank_ic)             # 平均 Rank IC
                std_rank_ic = np.std(daily_rank_ic)             # Rank IC 标准差
                pos_rank_ic_ratio = np.mean(np.array(daily_rank_ic) > 0)  # Rank IC>0 的比例
                
                # 存储结果
                factor_ic_dict[factor] = daily_ic
                factor_rank_ic_dict[factor] = daily_rank_ic
                
                # 写入结果到文件
                f.write(f"{factor},{avg_ic:.4f},{std_ic:.4f},{pos_ic_ratio:.4f},{avg_rank_ic:.4f},{std_rank_ic:.4f},{pos_rank_ic_ratio:.4f}\n")
                print(f"因子 {factor} 分析完成: 平均 IC={avg_ic:.4f}, 平均 Rank IC={avg_rank_ic:.4f}")
            else:
                f.write(f"{factor},N/A,N/A,N/A,N/A,N/A,N/A\n")
                print(f"因子 {factor} 没有有效数据")
        
        # 写入分析完成信息
        f.write("\n## 分析完成\n")
        f.write("分析完成！\n")
    
    # 打印完成信息
    print(f"分析完成！结果已输出到文件: {output_file}")
