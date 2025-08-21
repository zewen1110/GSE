import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 1. 读取文件
file_path = r'96804用于机器学习_DEGS+WGCNA+ptz.xlsx'
data = pd.read_excel(file_path)

# 2. 数据预处理
# 最后一列是分组标签，第一列是样本名称，其他列是基因表达值
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
features = data.columns[1:-1]  # 获取特征名称

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 建立LASSO模型
lasso = LassoCV(cv=10)
lasso.fit(X_scaled, y)

# 4. 计算每个lambda值对应的非零系数数量
num_nonzero_coefs = [np.sum(LassoCV(alphas=[alpha], cv=10).fit(X_scaled, y).coef_ != 0) for alpha in lasso.alphas_]

# 5. 绘制图像
plt.figure(figsize=(12, 8))  # 增加图像高度

# 图A: 绘制Binomial Deviance
mean_mse = lasso.mse_path_.mean(axis=-1)
std_mse = lasso.mse_path_.std(axis=-1)
alphas = np.logspace(-5.5, -0.5, 100)
plt.errorbar(np.log(alphas), mean_mse, yerr=std_mse, fmt='o', ecolor='gray', capsize=5, markersize=4, elinewidth=1, label='Cross-validated MSE')
lambda_cv_log = np.log(lasso.alpha_)
plt.axvline(lambda_cv_log, linestyle='--', color='k', label='Lambda CV')
plt.xlabel('Log Lambda', fontsize=18)
plt.ylabel('Binomial Deviance', fontsize=18)
plt.title('LASSO Model - Deviance', fontsize=22)

# 标注Lambda.CV对应的特征数量
selected_features_count = np.sum(lasso.coef_ != 0)
plt.text(lambda_cv_log, max(mean_mse) * 1.1, f'   Features: {selected_features_count}', verticalalignment='top', fontsize=12)

# 添加上方坐标轴
ax1 = plt.gca()
ax2 = ax1.twiny()

# 设置新坐标轴的范围与主坐标轴相同
ax2.set_xlim(ax1.get_xlim())
selected_ticks = np.linspace(0, len(lasso.alphas_) - 1, 15, dtype=int)  # 增加到15个标签
ax2.set_xticks(np.log(lasso.alphas_[selected_ticks]))

# 设置新坐标轴的标签，避免过于拥挤
ax2.set_xticklabels(np.array(num_nonzero_coefs)[selected_ticks], rotation=45, ha='right', fontsize=10)
ax2.set_xlabel('Number of Features Selected', fontsize=14)

plt.legend()
plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9)  # 调整边距以确保内容显示完整

plt.savefig('LASSO Model-Deviance_PTZ.svg')
# 6. 获取LASSO选择的非零系数特征
non_zero_coefs = lasso.coef_ != 0
selected_features = features[non_zero_coefs]
selected_coefs = lasso.coef_[non_zero_coefs]

# 创建DataFrame以保存选择的特征和系数
selected_features_df = pd.DataFrame({
    'Feature': selected_features,
    'Coefficient': selected_coefs
})

# 保存DataFrame到CSV文件
selected_features_df.to_csv(r'selected_features_coefficients_LASSO_PTZ.txt', index=False)