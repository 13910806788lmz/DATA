import time
import json
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import numpy as np
from wordcloud import WordCloud
from warnings import filterwarnings
from datetime import datetime
# from concat import df_10G
# from concat import df_30G
filterwarnings('ignore')

# ================== 基础设置 ==================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ================== 记录并打印耗时NEW ==================
def log_time(task_name, start_time):
    elapsed = time.time() - start_time
    print(f"[{task_name}] 耗时: {elapsed:.2f}秒 | 内存使用: {psutil.virtual_memory().percent}%")
    return time.time()

# ================== 数据加载NEW ==================
def load_data(df_):
    start = time.time()
    df = df_
    print(f"\n{'='*40}\n正在加载数据: ")    
    print("数据概览：")
    df.info()
    print(f"初始数据量: {len(df):,} 条")
    print(f"内存占用: {df.memory_usage(deep=True).sum()/1024**3:.2f} GB")
    log_time("数据加载", start)
    return df

# ================== 数据匿名化处理NEW ==================
def anonymize_data(df):
    # 重复值分析
    print("\n[重复值统计]", df.duplicated(subset='user_name', keep='first').sum())
    df.drop_duplicates(subset='user_name', keep='first')
    print('[删除重复值后数据量统计]', len(df)) 
    
    sensitive_cols = ['user_name', 'fullname', 'email', 'phone_number']
    df = df.drop(columns=[col for col in sensitive_cols if col in df.columns])
    return df

# ================== 时间格式转换与特征提取处理NEW ==================
def process_time_features(df):
    time_cols = ['last_login', 'registration_date']
    for col in time_cols:
        if col in df.columns:
            # 转换为datetime类型
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # 提取时间特征
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
    return df

# ================== 数据预处理NEW ==================
def preprocess(df, dataset_name):
    start = time.time()
    print(f"\n{'='*40}\n{dataset_name}预处理开始")
    
    # 缺失值分析
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_report = pd.concat([missing, missing_pct], axis=1)
    missing_report.columns = ['缺失数量', '缺失比例(%)']
    print("\n[缺失值统计]")
    print(missing_report)

    
    
    # age异常值处理NEW
    if 'age' in df.columns:
        df = df[(df['age'] >= 18) & (df['age'] <= 100)]  
        # 分位数分组
        bins = [18, 25, 35, 45, 55, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '55+']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)    

    # income异常值处理NEW    
    if 'income' in df.columns:
        # 过滤负值
        df = df[df['income'] >= 0]
        # 分位数分组
        df['income_level'] = pd.qcut(
            df['income'], 
            q=4, 
            labels=['低', '中低', '中高', '高']
        )
   
    # gender异常值处理 存在‘未指定’‘其他’NEW
    if 'gender' in df.columns:
        # 统一格式
        df['gender'] = df['gender'].str.strip().replace({
            'male': '男', 
            'female': '女',
            'm': '男',
            'f': '女'
        })
        
        # 处理异常值
        valid_genders = ['男', '女']
        df['gender'] = df['gender'].where(
            df['gender'].isin(valid_genders), 
            '其他'
        )
        # df = df[(df['gender'] == '男') | (df['gender'] == '女')]
    
    # address异常值处理 
    if 'address' in df.columns:
        # 提取省份
        province_pattern = r'(北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|台湾|内蒙古|广西|西藏|宁夏|新疆|香港|澳门)'
        df['province'] = df['address'].str.extract(province_pattern)

    # purchase_history异常值处理 
    if 'purchase_history' in df.columns:
        def parse_record(record):
            try:
                fixed = record.replace('}{', '},{')
                return json.loads(f'[{fixed}]')
            except:
                return np.nan
        # 解析JSON
        df['parsed_purchases'] = df['purchase_history'].apply(parse_record)
        # 提取消费金额
        df['purchase_amounts'] = df['parsed_purchases'].apply(
            lambda x: [round(float(item.get('avg_price', 0)), 2) for item in x]
            if isinstance(x, list) else []
        )
    
    # is_active异常值处理
    if 'is_active' in df.columns:
        # 转换为布尔类型
        df['is_active'] = df['is_active'].astype(bool)    

    log_time("预处理完成", start)
    return df

# ================== 可视化分析NEW ==================
def visualize(df, dataset_name):
    def visualize_age_distribution(df):
        """年龄分布可视化"""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
        plt.title('用户年龄分布')
        plt.xlabel('年龄')
        plt.ylabel('用户数量')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('visualize_age_distribution.png')
        # plt.show()

    def visualize_gender_distribution(df):
        """性别分布可视化"""
        gender_counts = df['gender'].value_counts()
        
        plt.figure(figsize=(10, 6))
        # 饼图
        plt.subplot(1, 2, 1)
        gender_counts.plot.pie(autopct='%1.11f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('性别比例')
        plt.ylabel('')
        
        # 柱状图
        plt.subplot(1, 2, 2)
        sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='pastel')
        plt.title('性别分布')
        plt.xlabel('性别')
        plt.ylabel('用户数量')
        
        plt.tight_layout()
        plt.savefig('visualize_gender_distribution.png')
        # plt.show()

    def visualize_income_analysis(df):
        """收入分析可视化"""
        plt.figure(figsize=(12, 6))
        
        # 箱线图
        plt.subplot(1, 2, 1)
        sns.boxplot(y=df['income'], color='lightgreen')
        plt.title('收入分布箱线图')
        plt.ylabel('收入（元）')
        
        # 分箱柱状图
        plt.subplot(1, 2, 2)
        sns.countplot(x='income_level', data=df, order=['低','中低','中高','高'], palette='Blues')
        plt.title('收入等级分布')
        plt.xlabel('收入等级')
        plt.ylabel('用户数量')
        
        plt.tight_layout()
        plt.savefig('visualize_income_analysis.png')
        # plt.show()

    def visualize_geo_distribution(df):
        """地理分布可视化"""
        plt.figure(figsize=(12, 6))
        
        # 国家分布
        plt.subplot(1, 2, 1)
        top_countries = df['country'].value_counts().head(5)
        sns.barplot(y=top_countries.index, x=top_countries.values, palette='viridis')
        plt.title('Top 5 国家分布')
        plt.xlabel('用户数量')
        
        # 省份分布
        plt.subplot(1, 2, 2)
        top_provinces = df['province'].value_counts().head(5)
        sns.barplot(y=top_provinces.index, x=top_provinces.values, palette='magma')
        plt.title('Top 5 省份分布')
        plt.xlabel('用户数量')
        
        plt.tight_layout()
        plt.savefig('visualize_geo_distribution.png')
        # plt.show()

    def visualize_purchase_behavior(df):
        """消费行为可视化"""
        plt.figure(figsize=(16, 12))
        
        # 消费金额分布
        plt.subplot(2, 2, 1)
        sns.histplot(df['purchase_amounts'].explode().astype(float), bins=30, kde=True, color='purple')
        plt.title('单次消费金额分布')
        plt.xlabel('消费金额（元）')
        
        # 消费频率分布
        plt.subplot(2, 2, 2)
        purchase_counts = df['purchase_amounts'].apply(len)
        sns.histplot(purchase_counts, bins=15, kde=True, color='orange')
        plt.title('用户消费次数分布')
        plt.xlabel('消费次数')
        
        # 消费时间热力图
        plt.subplot(2, 2, 3)
        df['purchase_hour'] = df['last_login'].dt.hour
        hour_counts = df['purchase_hour'].value_counts().sort_index()
        sns.heatmap(hour_counts.values.reshape(1, -1), annot=True, fmt="d", cmap='YlGnBu', cbar=False)
        plt.title('消费时段分布')
        plt.xlabel('小时(0-23)')
        plt.yticks([])
        
        # 消费类别词云
        plt.subplot(2, 2, 4)
        categories = df['parsed_purchases'].apply(
            lambda x: [item.get('categories') for item in x] 
            if isinstance(x, list) else []
        ).explode().value_counts().head(5).index.tolist()

        text = ' '.join(categories)
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='msyh.ttc').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('消费类别词云')
        
        plt.tight_layout()
        plt.savefig('visualize_purchase_behavior.png')
        # plt.show()

    def visualize_activity_analysis(df):
        """活跃度分析可视化"""
        plt.figure(figsize=(10, 6))
        
        # 活跃用户占比
        active_ratio = df['is_active'].mean()
        labels = ['活跃用户', '非活跃用户']
        sizes = [active_ratio, 1-active_ratio]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66ff66','#ff6666'])
        plt.title('用户活跃度比例')
        plt.savefig('visualize_activity_analysis.png')
        # plt.show()
 
    # ================== 登录行为深度分析==================
    def visualize_login_analysis(df):
        """登录行为可视化（基于login_history字段）"""
        plt.figure(figsize=(14, 6))
        
        # 登录频率分析
        plt.subplot(1, 2, 1)
        login_counts = df['login_history'].apply(
            lambda x: len(json.loads(x)) if pd.notnull(x) else 0
        )
        sns.histplot(login_counts, bins=20, kde=True, color='navy')
        plt.title('用户登录次数分布')
        plt.xlabel('历史登录总次数')
        
        # 登录时段分析
        plt.subplot(1, 2, 2)
        def extract_login_hours(history):
            try:
                logins = json.loads(history)
                return [datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S').hour for entry in logins]
            except:
                return []
        
        all_hours = df['login_history'].apply(extract_login_hours).explode().dropna()
        sns.histplot(all_hours.astype(int), bins=24, discrete=True, color='darkorange')
        plt.title('登录时段分布')
        plt.xlabel('小时 (0-23)')
        plt.xticks(range(0,24))
        plt.tight_layout()
        plt.savefig('visualize_login_analysis.png')
        profile = {
            'avg_login_freq': login_counts.mean(),
        }
        print(f"- 平均登录频率: {profile['avg_login_freq']:.1f}次")
        # return login_counts

    # ================== 消费行为增强分析================== 
    def visualize_purchase_behavior_enhanced(df):
        """增强型消费行为分析"""
        plt.figure(figsize=(18, 12))
        
        # 消费周期分析（注册日期到最近购买日）
        # 确保列是 datetime 类型
        df['last_login'] = pd.to_datetime(df['last_login'])
        df['registration_date'] = pd.to_datetime(df['registration_date'])

        # 统一时区（假设需要去除时区）
        df['last_login'] = df['last_login'].dt.tz_localize(None)
        df['registration_date'] = df['registration_date'].dt.tz_localize(None)


        # 处理缺失值
        df = df.dropna(subset=['last_login', 'registration_date'])

        df['purchase_timedelta'] = (df['last_login'] - df['registration_date']).dt.days
        plt.subplot(2, 2, 1)
        sns.histplot(df[df['purchase_timedelta']>0]['purchase_timedelta'], bins=30, kde=True)
        plt.title('注册到最近消费的时间分布')
        plt.xlabel('天数')
        
        # 消费金额与收入关系
        plt.subplot(2, 2, 2)
        df['total_spent'] = df['purchase_amounts'].apply(lambda x: sum(map(float, x)) if x else 0)
        sns.scatterplot(x='income', y='total_spent', hue='age_group', data=df, alpha=0.6)
        plt.title('收入 vs 总消费金额（按年龄组）')
        plt.xlabel('收入')
        plt.ylabel('总消费金额')
        
        # 复购率分析
        plt.subplot(2, 2, 3)
        df['purchase_count'] = df['purchase_amounts'].apply(len)
        repurchase_rate = (df['purchase_count'] > 1).mean()
        plt.pie([repurchase_rate, 1-repurchase_rate], 
                labels=['复购用户', '单次购买用户'],
                autopct='%1.1f%%', 
                colors=['#4CAF50', '#FF5722'])
        plt.title('用户复购率分析')
        
        # 消费频率与活跃度关系
        plt.subplot(2, 2, 4)
        sns.boxplot(x='is_active', y='purchase_count', data=df)
        plt.title('活跃用户 vs 非活跃用户消费次数')
        plt.xticks([0, 1], ['非活跃', '活跃'])
        plt.ylabel('消费次数')
        
        plt.tight_layout()
        plt.savefig('visualize_purchase_behavior_enhanced.png')

        profile = {
            'repurchase_rate': repurchase_rate,
        }
        print(f"- 用户复购率: {profile['repurchase_rate']:.2%}")
        # return repurchase_rate

    # ================== 用户生命周期分析==================
    def analyze_user_lifecycle(df):
        """用户生命周期分析"""
        # 计算生命周期指标
        # 确保列是 datetime 类型
        df['last_login'] = pd.to_datetime(df['last_login'])
        df['registration_date'] = pd.to_datetime(df['registration_date'])

        # 统一时区（假设需要去除时区）
        df['last_login'] = df['last_login'].dt.tz_localize(None)
        df['registration_date'] = df['registration_date'].dt.tz_localize(None)

        # 处理缺失值
        df = df.dropna(subset=['last_login', 'registration_date'])
        df['lifetime_days'] = (df['last_login'] - df['registration_date']).dt.days
        df['recent_activity'] = (pd.to_datetime('today') - df['last_login']).dt.days
        
        plt.figure(figsize=(14, 6))
        
        # 生命周期分布
        plt.subplot(1, 2, 1)
        sns.histplot(df[df['lifetime_days']>0]['lifetime_days'], bins=30, kde=True)
        plt.title('用户生命周期分布')
        plt.xlabel('活跃天数')
        
        # 留存分析
        plt.subplot(1, 2, 2)
        retention_bins = [0, 7, 30, 90, 365, np.inf]
        retention_labels = ['<1周', '1周-1月', '1-3月', '3-12月', '>1年']
        df['retention_period'] = pd.cut(df['recent_activity'], bins=retention_bins, labels=retention_labels)
        sns.countplot(y='retention_period', data=df, order=retention_labels, palette='RdYlGn')
        plt.title('用户留存情况分析')
        plt.xlabel('用户数量')
        plt.tight_layout()
        plt.savefig('analyze_user_lifecycle.png')
        profile = {
            'median_lifetime': df['lifetime_days'].median(),
            'retention_distribution': df['retention_period'].value_counts(normalize=True).to_dict(),
        }
        print(f"- 中位生命周期: {profile['median_lifetime']:.1f}天")
        print(f"- 用户留存分布: {', '.join([f'{k}: {v:.4f}' for k, v in profile['retention_distribution'].items()])}")
        
    print('\n可视化分析', dataset_name)
    visualize_age_distribution(df)
    visualize_gender_distribution(df)
    visualize_income_analysis(df)
    visualize_geo_distribution(df)
    visualize_purchase_behavior(df)
    visualize_activity_analysis(df)
    visualize_login_analysis(df)
    visualize_purchase_behavior_enhanced(df)
    analyze_user_lifecycle(df)

# ================== 用户画像分析 ==================
def user_profiling(df):
    print(f"\n{'='*40}\n构建用户画像") 
    start = time.time()
    
    # 展平所有消费金额
    all_amounts = df['purchase_amounts'].explode()
    valid_amounts = pd.to_numeric(all_amounts, errors='coerce').dropna()


    profile = {
        'median_age': df['age'].median(),
        'age_distribution': df['age_group'].value_counts(normalize=True).to_dict(),
        'gender_dist': df['gender'].value_counts(normalize=True).to_dict(), 
        'top_countries': df['country'].value_counts().head(5).index.tolist(),
        'top_provinces': df['province'].value_counts().head(5).index.tolist(),
        'avg_income': df['income'].mean(),
        'income_distribution': df['income_level'].value_counts(normalize=True).to_dict(),
        'avg_purchase': valid_amounts.mean(),
        'purchase_freq': len(valid_amounts) / len(df),
        'top_categories': df['parsed_purchases'].apply(
            lambda x: [item.get('categories') for item in x] 
            if isinstance(x, list) else []
        ).explode().value_counts().head(5).index.tolist(),
        'active_ratio': df['is_active'].mean(),

    }

    print(f"- 年龄中位数: {profile['median_age']:.4f}岁")
    print(f"- 年龄分布: {{{', '.join(f'{k}: {v:.4f}' for k, v in profile['age_distribution'].items())}}}")
    print(f"- 性别分布: {{{', '.join(f'{k}: {v:.4f}' for k, v in profile['gender_dist'].items())}}}")
    print(f"- 高频国家: {', '.join(profile['top_countries'])}")
    print(f"- 高频城市: {', '.join(profile['top_provinces'])}")
    print(f"- 收入平均水平: {profile['avg_income']:.4f}元")
    print(f"- 收入分布: {{{', '.join(f'{k}: {v:.4f}' for k, v in profile['income_distribution'].items())}}}")
    print(f"- 消费平均水平: {profile['avg_purchase']:.4f}元")
    print(f"- 消费频率: {profile['purchase_freq']:.4f}")
    print(f"- 高频消费产品: {profile['top_categories']}")
    print(f"- 活跃度平均水平: {profile['active_ratio']:.4f}")
    
    log_time("\n画像分析完成", start)
    return profile

if __name__ == "__main__":
    total_start = time.time()

    # df_test = df_30G
    # df_test = df_10G   
    # df = load_data(df_test)

    # #test
    base_path = r'D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data_new\part-00000.parquet'
    dfs = pd.read_parquet(base_path)
    df = dfs
    # #test

    # D:\360MoveData\Users\Admin\OneDrive\桌面\数据挖掘\30G_data_new\part-00000.parquet
    df = anonymize_data(df)
    df = process_time_features(df)   
    df_clean = preprocess(df, "30G数据集") # 数据预处理

    visualize(df_clean, "30G数据集")
    profile_10g = user_profiling(df_clean)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*40}\n总耗时: {total_elapsed/60:.2f}分钟")