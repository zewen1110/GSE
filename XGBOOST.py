import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, f1_score, recall_score, accuracy_score, matthews_corrcoef, precision_score
from sklearn.model_selection import StratifiedKFold

# 加载数据
data = pd.read_excel(r'96804用于机器学习_DEGS+WGCNA+ptz.xlsx')

# 分离特征和目标变量
X = data.iloc[:, 1:-1]  # 排除第一列和最后一列
y = data.iloc[:, -1]

# 创建XGBoost模型
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.5,
    learning_rate=0.05,
    n_estimators=200,
    reg_lambda=1.0,
    reg_alpha=1.0
)

# 进行10折分层交叉验证
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
auc_scores = []
f1_scores = []
recall_scores = []
accuracy_scores = []
mcc_scores = []
precision_scores = []
conf_matrices = []
class_reports = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 选择正类的概率
    y_pred = model.predict(X_test)

    # 计算AUC得分
    auc_score = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc_score)

    # 计算F1 score
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

    # 计算Recall
    recall = recall_score(y_test, y_pred)
    recall_scores.append(recall)

    # 计算Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # 计算Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_test, y_pred)
    mcc_scores.append(mcc)

    # 计算Precision
    precision = precision_score(y_test, y_pred)
    precision_scores.append(precision)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrices.append(conf_matrix)

    # 分类报告
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_reports.append(class_report)

# 输出平均AUC得分
print(f"Average AUC Score: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
# 输出F1、Recall、Accuracy、MCC、Precision及其标准差
print(f"Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Average Recall (REC): {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Average Accuracy (ACC): {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Average Matthews Correlation Coefficient (MCC): {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}")
print(f"Average Precision (PRE): {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")

#
# # 绘制ROC曲线（最后一折）
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([-0.05, 1.05])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic of XGB Classifier')
# plt.legend(loc="lower right")
# plt.savefig('roc曲线_XGB.svg')
# plt.show()
# # 用全部数据重新训练一个final模型
# final_model = xgb.XGBClassifier(
#     objective='binary:logistic',
#     eval_metric='auc',
#     use_label_encoder=False,
#     random_state=42,
#     max_depth=3,              # 降低树深度
#     subsample=0.8,            # 使用部分样本
#     colsample_bytree=0.5,     # 使用部分特征
#     learning_rate=0.05,
#     n_estimators=200,
#     reg_lambda=1.0,           # 增强正则化
#     reg_alpha=1.0
# )
#
# final_model.fit(X, y)
#
# # 再做SHAP解释
# explainer = shap.Explainer(final_model, X)
# shap_values = explainer(X)
#
# # # 后面保持不变即可
# #
# # # 使用SHAP计算解释性（基于全体数据）
# # explainer = shap.Explainer(model, X)
# # shap_values = explainer(X)
#
# # 计算并输出每个特征的平均绝对SHAP值（特征重要性得分）
# feature_importance = np.abs(shap_values.values).mean(0)
# feature_importance_df = pd.DataFrame(list(zip(X.columns, feature_importance)), columns=['Feature', 'SHAP Importance'])
# feature_importance_df = feature_importance_df[feature_importance_df['SHAP Importance'] > 0].sort_values(
#     by='SHAP Importance', ascending=False)
#
# # 筛选重要性大于0的特征
# important_features = feature_importance_df['Feature'].tolist()  # 将Pandas Series转换为列表
# filtered_shap_values = shap_values[:, important_features]  # 使用列表来索引
#
# # 绘制筛选后的SHAP总结图
# shap.summary_plot(filtered_shap_values, X[important_features], show=False)
#
# # 保存图像
# plt.savefig('SHAP_Summary_Plot_PTZ.svg')
# plt.show()  # 确保这一行在保存之后调用
#
# # 将SHAP values和基因名保存到文本文件中
# # 计算每个特征的平均SHAP值并保存到文本文件
# average_shap_values = np.abs(shap_values.values).mean(axis=0)
# average_shap_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Average SHAP Value': average_shap_values
# })
#
# # 保存到TXT文件
# average_shap_df.to_csv(r'average_shap_values_PTZ.txt', sep='\t', index=False)
