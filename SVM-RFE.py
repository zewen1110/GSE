import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, accuracy_score, matthews_corrcoef, precision_score
from sklearn.model_selection import cross_val_predict
import shap  # 导入SHAP库

# 设置随机种子以确保结果可重复
np.random.seed(44)

# 读取数据
data = pd.read_excel(r'96804用于机器学习_DEGS+WGCNA+ptz.xlsx')

# 分离特征和目标变量
X = data.iloc[:, 1:-1]  # 排除第一列和最后一列
y = data.iloc[:, -1]

# 实例化SVM分类器
clf = SVC(kernel='linear', probability=True, random_state=17315,C=50)

# 使用递归特征消除法进行10折交叉验证
cv = StratifiedKFold(10)  # 10折交叉验证
selector = RFECV(clf, step=1, cv=cv, scoring='accuracy', min_features_to_select=1)
selector = selector.fit(X, y)

# 输出特征选择结果
selected_features = [i for i in range(len(selector.support_)) if selector.support_[i]]

# 从cv_results_中提取平均得分（准确率）
mean_scores = selector.cv_results_['mean_test_score']
cv_errors = 1 - mean_scores  # 将准确率转换为错误率

print('10折交叉验证错误率随特征数量的变化：', cv_errors)

# 绘制交叉验证错误率图
plt.figure()
plt.xlabel("Number of Features")
plt.ylabel("10x CV Error")

# 绘制平均交叉验证错误率
plt.plot(range(1, len(cv_errors) + 1), cv_errors)

# 突出显示最优特征数量
optimal_idx = np.argmin(cv_errors)
plt.scatter(optimal_idx + 1, cv_errors[optimal_idx], color='red')
plt.text(optimal_idx + 1, cv_errors[optimal_idx], f'{optimal_idx + 1} - {cv_errors[optimal_idx]:.4f}', color='red')

# 设置x轴刻度
plt.xticks(range(1, len(cv_errors) + 1, 25))
plt.savefig('svm-rfe-feature_select_test.svg')

# 仅使用选择的特征
X_selected = X.iloc[:, selected_features]

# 使用选择的特征进行10折交叉验证并进行预测
y_pred_proba = cross_val_predict(clf, X_selected, y, cv=cv, method='predict_proba')[:, 1]
y_pred = cross_val_predict(clf, X_selected, y, cv=cv, method='predict')

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])  # 设置x轴范围
plt.ylim([0.0, 1.05])  # 设置y轴范围
plt.gca().set_aspect('auto')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver Operating Characteristic of SVM Classifier', fontsize=22)
plt.legend(loc="lower right")
plt.savefig('svm-rfe_roc_96804_PTZ.svg')

# 计算评估指标
accuracy = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
recall = recall_score(y, y_pred)
precision = precision_score(y, y_pred)
mcc = matthews_corrcoef(y, y_pred)

# 输出评估指标
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Recall (REC): {recall:.4f}')
print(f'Precision (PRE): {precision:.4f}')
print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')

# 计算标准差
# 我们可以在此处计算标准差，使用cross_val_predict来获取每折的预测结果。
# 获取每一折的预测结果
y_pred_all = cross_val_predict(clf, X_selected, y, cv=cv)

accuracy_std = np.std([accuracy_score(y[test], y_pred_all[test]) for _, test in cv.split(X_selected, y)])
f1_std = np.std([f1_score(y[test], y_pred_all[test]) for _, test in cv.split(X_selected, y)])
recall_std = np.std([recall_score(y[test], y_pred_all[test]) for _, test in cv.split(X_selected, y)])
precision_std = np.std([precision_score(y[test], y_pred_all[test]) for _, test in cv.split(X_selected, y)])
mcc_std = np.std([matthews_corrcoef(y[test], y_pred_all[test]) for _, test in cv.split(X_selected, y)])

# 输出标准差
print(f'Accuracy Std: {accuracy_std:.4f}')
print(f'F1 Score Std: {f1_std:.4f}')
print(f'Recall Std: {recall_std:.4f}')
print(f'Precision Std: {precision_std:.4f}')
print(f'MCC Std: {mcc_std:.4f}')

# 输出被选择的特征名称并保存到txt文件
selected_feature_names = data.columns[1:-1][selector.support_].tolist()
selected_features_txt = '\n'.join(selected_feature_names)

# # 保存到TXT文件
with open(r'selected_features_SVM_PTZ.txt', 'w') as file:
    file.write(selected_features_txt)

print("选择的特征已保存到文件中")

# 使用所有数据训练最终的SVM模型
final_clf = SVC(kernel='linear', probability=True, random_state=17315,C=50)
final_clf.fit(X_selected, y)
explainer = shap.Explainer(final_clf, X_selected)
shap_values = explainer(X_selected)
# 计算并输出每个特征的平均绝对SHAP值（特征重要性得分）
feature_importance = np.abs(shap_values.values).mean(0)
feature_importance_df = pd.DataFrame(list(zip(X.columns, feature_importance)), columns=['Feature', 'SHAP Importance'])
feature_importance_df = feature_importance_df[feature_importance_df['SHAP Importance'] > 0].sort_values(
    by='SHAP Importance', ascending=False)
shap.summary_plot(shap_values, X_selected, show=False)
plt.savefig('SHAP_Summary_Plot_SVM_PTZ.svg')
plt.show()  # 确保这一行在保存之后调用