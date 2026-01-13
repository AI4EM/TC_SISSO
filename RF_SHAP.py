import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 重新加载数据并训练模型（保持完整性）
data = pd.read_csv('/mnt/all.csv')

# 提取特征和目标变量
X = data.drop('K', axis=1)
y = data['K']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# 2. SHAP分析核心代码
# 创建SHAP解释器（针对树模型的优化版本）
explainer = shap.TreeExplainer(rf_model)

# 计算测试集的SHAP值（使用测试集进行解释）
shap_values = explainer.shap_values(X_test)

# 3. SHAP可视化分析
# 3.1 总体特征重要性（SHAP值的绝对值均值）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Absolute Mean)', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.2 特征影响散点图（Summary Plot）
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot (Feature Impact on Predictions)', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.3 单个特征的依赖图（以最重要的content为例）
plt.figure(figsize=(10, 6))
shap.dependence_plot(
    "content",  # 要分析的特征名
    shap_values, 
    X_test,
    show=False,
    alpha=0.5,
    color='#1f77b4'
)
plt.title('SHAP Dependence Plot for "content" Feature', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/shap_dependence_content.png', dpi=300, bbox_inches='tight')
plt.close()

# 3.4 单个预测样本的解释（以测试集第一个样本为例）
sample_idx = 0
plt.figure(figsize=(12, 8))
shap.force_plot(
    explainer.expected_value,  # 模型的基准预测值
    shap_values[sample_idx],   # 该样本的SHAP值
    features=X_test.iloc[sample_idx],  # 该样本的特征值
    matplotlib=True,
    show=False,
    figsize=(12, 4)
)
plt.title(f'SHAP Force Plot for Sample {sample_idx+1}', fontsize=14)
plt.tight_layout()
plt.savefig('/mnt/shap_force_plot_sample.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 输出关键SHAP统计信息
# 计算每个特征的平均绝对SHAP值（用于量化重要性）
shap_importance = pd.DataFrame({
    'Feature': X.columns,
    'Mean Absolute SHAP Value': np.abs(shap_values).mean(axis=0)
}).sort_values('Mean Absolute SHAP Value', ascending=False)

print("=== SHAP特征重要性（平均绝对SHAP值） ===")
print(shap_importance)

# 输出模型基准值（所有样本的平均预测值）
print(f"\n模型基准预测值（Expected Value）: {explainer.expected_value:.4f}")

# 输出第一个样本的详细解释
print(f"\n=== 第一个测试样本的SHAP解释 ===")
print(f"样本真实值: {y_test.iloc[sample_idx]:.4f}")
print(f"样本预测值: {rf_model.predict(X_test.iloc[sample_idx:sample_idx+1])[0]:.4f}")
print(f"各特征SHAP贡献值:")
for feat, shap_val in zip(X.columns, shap_values[sample_idx]):
    print(f"  {feat}: {shap_val:.6f}")