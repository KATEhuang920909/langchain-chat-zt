# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     logo_parse
   Author :       huangkai
   date：          2025/6/24
-------------------------------------------------
"""
import pandas as pd


def OperationLog(log_data):
    # 假设日志数据存储在CSV文件中
    # 提取主帐号名称、操作时间和操作内容

    log_data = log_data[['主帐号名称', '操作时间', '操作内容']]

    # 确保操作时间格式正确
    log_data['操作时间'] = pd.to_datetime(log_data['操作时间'], errors='coerce')
    log_data = log_data.dropna(subset=['操作时间'])

    # 按主帐号名称分类统计
    account_counts = log_data['主帐号名称'].value_counts().to_dict()

    log_data['日期'] = log_data['操作时间'].dt.date
    daily_counts = log_data.groupby('日期').size().to_dict()

    # 按操作内容分类统计
    operation_counts = log_data['操作内容'].value_counts().to_dict()

    # 多维度组合分析：按主账号和操作时间分组
    account_daily_counts = log_data.groupby(['主帐号名称', '日期']).size().unstack(fill_value=0).to_dict()

    error_time = log_data[((log_data['操作时间'].dt.hour >= 0) & (log_data['操作时间'].dt.hour < 6))]

    error_login_daily_counts = error_time.groupby(['主帐号名称', '日期']).size().unstack(fill_value=0).to_dict()
    error_login_daily_counts = {
        date: {k: v for k, v in inner_dict.items() if v != 0}
        for date, inner_dict in error_login_daily_counts.items()
    }
    error_login_counts = error_time['主帐号名称'].value_counts().to_dict()

    return account_counts, daily_counts, operation_counts, account_daily_counts, error_login_counts, error_login_daily_counts


def UseLog(log_data):
    # # 提取登录ID、操作内容和访问时间
    # log_data = log_data[['登录ID', '操作内容', '访问时间']]
    #
    # # 确保访问时间格式正确
    # log_data['访问时间'] = pd.to_datetime(log_data['访问时间'], format='%Y%m%d%H%M%S', errors='coerce')
    # log_data = log_data.dropna(subset=['访问时间'])
    #
    # # 按登录ID分类统计
    # login_id_counts = log_data['登录ID'].value_counts()
    #
    # # 按操作内容分类统计
    # operation_counts = log_data['操作内容'].value_counts()
    #
    # # 按访问时间分类统计
    # log_data['日期'] = log_data['访问时间'].dt.date
    # daily_counts = log_data.groupby('日期').size()
    #
    # # 多维度组合分析：按登录ID和操作内容分组
    # login_operation_counts = log_data.groupby(['登录ID', '操作内容']).size().unstack(fill_value=0)
    #
    # # 多维度组合分析：按登录ID和日期分组
    # login_daily_counts = log_data.groupby(['登录ID', '日期']).size().unstack(fill_value=0)
    #
    # error_time = log_data[(log_data['访问时间'].dt.hour >= 0) & (log_data['访问时间'].dt.hour < 6)]
    #
    # error_login_daily_counts = error_time.groupby(['登录ID', '日期']).size().unstack(fill_value=0).to_dict()
    # error_login_daily_counts = {
    #     date: {k: v for k, v in inner_dict.items() if v != 0}
    #     for date, inner_dict in error_login_daily_counts.items()
    # }
    # error_login_counts = error_time['登录ID'].value_counts()

    # 提取登录ID、操作内容和访问时间
    log_data = log_data[['登录ID', '操作内容', '访问时间']]

    # 确保访问时间格式正确
    log_data['访问时间'] = pd.to_datetime(log_data['访问时间'], format='%Y%m%d%H%M%S', errors='coerce')
    log_data = log_data.dropna(subset=['访问时间'])

    # 按登录ID分类统计
    login_id_counts = log_data['登录ID'].value_counts().to_dict()

    # 按操作内容分类统计
    operation_counts = log_data['操作内容'].value_counts().to_dict()

    # 按访问时间分类统计
    log_data['日期'] = log_data['访问时间'].dt.date
    daily_counts = log_data.groupby('日期').size().to_dict()
    # 多维度组合分析：按登录ID和操作内容分组
    login_operation_counts = log_data.groupby(['登录ID', '操作内容']).size().unstack(fill_value=0).to_dict()
    all_items = []
    for date, inner_dict in login_operation_counts.items():
        for key, value in inner_dict.items():
            all_items.append((date, key, value))

    # 按值降序排序
    all_items.sort(key=lambda x: x[2], reverse=True)

    # 获取前20个最大值
    login_operation_counts = all_items[:50]

    # 多维度组合分析：按登录ID和日期分组
    login_daily_counts = log_data.groupby(['登录ID', '日期']).size().unstack(fill_value=0)
    all_items = []
    for date, inner_dict in login_daily_counts.items():
        for key, value in inner_dict.items():
            all_items.append((date, key, value))

    # 按值降序排序
    all_items.sort(key=lambda x: x[2], reverse=True)

    # 获取前20个最大值
    login_daily_counts = all_items[:50]

    # +++++++++++++++++++++++++++++
    error_time = log_data[(log_data['访问时间'].dt.hour >= 0) & (log_data['访问时间'].dt.hour < 6)]
    # +++++++++++++++++++++++++++++
    error_login_daily_counts = error_time.groupby(['登录ID', '日期']).size().unstack(fill_value=0).to_dict()
    error_login_daily_counts = {
        date: {k: v for k, v in inner_dict.items() if v != 0}
        for date, inner_dict in error_login_daily_counts.items()
    }
    # ++++++++++++++++++++++++
    error_login_counts = error_time['登录ID'].value_counts().to_dict()

    return login_id_counts, operation_counts, daily_counts, login_operation_counts, login_daily_counts, error_login_counts, error_login_daily_counts


def EnterLog(log_data):
    # 提取登录ID、登录时间、登出时间和客户端浏览器
    log_data = log_data[['登录ID', '登录时间']]

    # 确保登录时间和登出时间格式正确
    log_data['登录时间'] = pd.to_datetime(log_data['登录时间'], format='%Y%m%d%H%M%S', errors='coerce')
    log_data = log_data.dropna(subset=['登录时间'])
    # 数据可视化：登录ID登录频率图
    login_counts = log_data['登录ID'].value_counts().to_dict()#[:30].to_dict()

    # 按登录时间分类统计
    log_data['日期'] = log_data['登录时间'].dt.date
    daily_counts = log_data.groupby('日期').size().to_dict()

    # 每个登录ID在一天之内的登录次数排序
    log_data = log_data[['登录ID', '登录时间']]

    # 确保登录时间格式正确
    log_data['登录时间'] = pd.to_datetime(log_data['登录时间'], format='%Y%m%d%H%M%S', errors='coerce')
    log_data = log_data.dropna(subset=['登录时间'])

    # 提取日期部分
    log_data['日期'] = log_data['登录时间'].dt.date

    # 按登录ID和日期分组，统计每个ID在每一天的登录次数
    daily_login_counts = log_data.groupby(['登录ID', '日期']).size().reset_index(name='登录次数')

    # 按登录次数降序排序
    sorted_daily_login_counts = daily_login_counts.sort_values(by='登录次数', ascending=False)
    pivot_table = sorted_daily_login_counts[:30].pivot(index='登录ID', columns='日期', values='登录次数').fillna(
        0).to_dict()
    pivot_table = {
        date: {k: v for k, v in inner_dict.items() if v != 0}
        for date, inner_dict in pivot_table.items()
    }

    error_time = log_data[(log_data['登录时间'].dt.hour >= 0) & (log_data['登录时间'].dt.hour < 6)]
    error_daily_login_counts = error_time.groupby(['登录ID', '日期']).size().reset_index(name='登录次数')
    error_sorted_daily_login_counts = error_daily_login_counts.sort_values(by='登录次数', ascending=False)
    error_pivot_table = error_sorted_daily_login_counts[:30].pivot(index='登录ID', columns='日期',
                                                                   values='登录次数').fillna(0).to_dict()

    error_login_counts = error_time['登录ID'].value_counts().to_dict()

    return login_counts, daily_counts, pivot_table, error_login_counts, error_pivot_table

# if __name__ == '__main__':
#     data =pd.read_excel(r"D:\code\langchain-chat-zt\tests\logparse\操作日志查验：应用侧近半年日志(四川).xlsx")
#     print(UseLog(data))

