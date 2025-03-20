import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from contextlib import redirect_stdout
import shap

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 根据环境选择设备，有cuda就用cuda，否则用cpu
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device_name)

# 尝试加载数据，不存在则生成模拟数据
try:
    data = pd.read_excel('data.xlsx', header=0)
except FileNotFoundError:
    print("未找到 Excel 文件，生成模拟数据作为替代。")
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=5760, n_features=33, n_informative=20, random_state=42)
    data = pd.DataFrame(X_dummy, columns=[f'feature_{i}' for i in range(33)])
    data['是否舞弊'] = y_dummy

print("数据集列名：", data.columns.tolist())

target_column = '是否舞弊'
X = data.drop(target_column, axis=1)
y = data[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_mlp(X_train, y_train, X_test, y_test):
    # 用MLPClassifier训练简单神经网络模型
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("MLP evaluation:", metrics)
    return model, metrics


def train_tabnet(X_train, y_train, X_test, y_test):
    # 使用TabNet模型进行训练
    from pytorch_tabnet.tab_model import TabNetClassifier
    model = TabNetClassifier(device_name=device_name, verbose=0, seed=42)
    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_test.values, y_test.values)],
        max_epochs=50,
        patience=10,
        batch_size=256,
        virtual_batch_size=128
    )
    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("TabNet evaluation:", metrics)
    return model, metrics


class MeanPooling(nn.Module):
    def forward(self, x):
        # 对序列维度取均值，简单聚合特征
        return x.mean(dim=1)


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, n_classes, d_model=32, nhead=4, num_layers=1,
                 epochs=20, lr=0.001, batch_size=32, device='cpu'):
        # 设置模型超参，并调用构建模型函数
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self._build_model()

    def _build_model(self):
        # 利用TransformerEncoder提取特征，经过均值池化后输出类别概率
        self.model = nn.Sequential(
            nn.Unflatten(1, (self.input_dim, 1)),
            nn.Linear(1, self.d_model),
            nn.ReLU(),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, batch_first=True),
                num_layers=self.num_layers
            ),
            MeanPooling(),
            nn.Linear(self.d_model, self.n_classes)
        )
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model.train()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        predicted = torch.argmax(outputs, dim=1).cpu().numpy()
        return predicted

    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            proba = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        return proba


def train_ft_transformer(X_train, y_train, X_test, y_test):
    # 训练FT-Transformer模型
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = FTTransformerClassifier(input_dim=input_dim, n_classes=n_classes,
                                    epochs=20, device=device_name)
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("FT-Transformer evaluation:", metrics)
    return model, metrics


def plot_roc_curve(metrics, model_name):
    # 绘制ROC曲线并保存图像
    fpr, tpr, _ = metrics['roc_curve']
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["roc_auc"]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.savefig(f"{model_name}_roc_curve.png")
    plt.close()


def optimize_mlp(X_train, y_train, X_test, y_test):
    # 优化MLP模型并评估特征重要性
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("Optimized MLP evaluation:", metrics)
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test, y_test, scoring='roc_auc', n_repeats=10, random_state=42)
    feature_importances = result.importances_mean
    print("Optimized MLP feature importances:", feature_importances)
    return model, metrics, feature_importances


def optimize_tabnet(X_train, y_train, X_test, y_test):
    # 优化TabNet模型，并获取模型自带的特征重要性
    from pytorch_tabnet.tab_model import TabNetClassifier
    model = TabNetClassifier(device_name=device_name, verbose=0, seed=42)
    model.fit(
        X_train.values, y_train.values,
        eval_set=[(X_test.values, y_test.values)],
        max_epochs=100,
        patience=15,
        batch_size=256,
        virtual_batch_size=128
    )
    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("Optimized TabNet evaluation:", metrics)
    feature_importances = model.feature_importances_
    print("Optimized TabNet feature importances:", feature_importances)
    return model, metrics, feature_importances


def optimize_ft_transformer(X_train, y_train, X_test, y_test):
    # 优化FT-Transformer模型，并用置换重要性计算特征贡献
    input_dim = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    model = FTTransformerClassifier(input_dim=input_dim, n_classes=n_classes,
                                    epochs=40, lr=0.0005, device=device_name)
    model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    y_proba = model.predict_proba(X_test.values)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("Optimized FT-Transformer evaluation:", metrics)
    from sklearn.inspection import permutation_importance
    result = permutation_importance(model, X_test.values, y_test.values, scoring='roc_auc', n_repeats=10,
                                    random_state=42)
    feature_importances = result.importances_mean
    print("Optimized FT-Transformer feature importances:", feature_importances)
    return model, metrics, feature_importances


def ensemble_with_xgboost(mlp_model, tabnet_model, ft_model, X_train, y_train, X_test, y_test):
    # 将三个模型的预测概率作为新特征，训练XGBoost集成模型
    mlp_train_proba = get_positive_probability(mlp_model, X_train)
    tabnet_train_proba = get_positive_probability(tabnet_model, X_train.values)
    ft_train_proba = get_positive_probability(ft_model, X_train.values)
    X_train_meta = np.vstack([mlp_train_proba, tabnet_train_proba, ft_train_proba]).T

    mlp_test_proba = get_positive_probability(mlp_model, X_test)
    tabnet_test_proba = get_positive_probability(tabnet_model, X_test.values)
    ft_test_proba = get_positive_probability(ft_model, X_test.values)
    X_test_meta = np.vstack([mlp_test_proba, tabnet_test_proba, ft_test_proba]).T

    from xgboost import XGBClassifier
    ensemble_model = XGBClassifier(eval_metric='logloss', random_state=42)
    ensemble_model.fit(X_train_meta, y_train)

    y_pred = ensemble_model.predict(X_test_meta)
    y_proba = ensemble_model.predict_proba(X_test_meta)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("Ensemble XGBoost evaluation:", metrics)
    return ensemble_model, metrics, X_test_meta


def optimize_ensemble_xgboost(X_train_meta, y_train, X_test_meta, y_test):
    # 利用网格搜索找XGBoost集成模型的最佳参数
    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    grid = GridSearchCV(xgb_model, param_grid, scoring='roc_auc', cv=3, verbose=1)
    grid.fit(X_train_meta, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_meta)
    y_proba = best_model.predict_proba(X_test_meta)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    metrics = {
        'accuracy': acc,
        'f1_score': f1,
        'roc_auc': auc_score,
        'roc_curve': (fpr, tpr, thresholds)
    }
    print("Optimized Ensemble XGBoost evaluation:", metrics)
    feature_importances = best_model.feature_importances_
    print("Optimized Ensemble XGBoost feature importances:", feature_importances)
    return best_model, metrics, feature_importances


def get_positive_probability(model, X):
    # 返回模型预测中正类的概率值，兼容不同输入格式
    if hasattr(model, "feature_names_in_") and not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=model.feature_names_in_)
    proba = model.predict_proba(X)
    if hasattr(model, 'classes_'):
        if model.classes_[0] == 1:
            return proba[:, 0]
        elif model.classes_[1] == 1:
            return proba[:, 1]
        else:
            raise ValueError("模型的 classes_ 中没有包含标签1")
    else:
        return proba[:, 1]


def final_model_test(final_model, optimized_models, sample_data, sample_labels):
    # 用集成模型对10个样本进行预测，计算常用评估指标并绘制ROC曲线
    mlp_sample_proba = optimized_models[0].predict_proba(sample_data)[:, 1]
    tabnet_sample_proba = optimized_models[1].predict_proba(sample_data.values)[:, 1]
    ft_sample_proba = optimized_models[2].predict_proba(sample_data.values)[:, 1]
    X_sample_meta = np.vstack([mlp_sample_proba, tabnet_sample_proba, ft_sample_proba]).T

    y_pred = final_model.predict(X_sample_meta)
    y_proba = final_model.predict_proba(X_sample_meta)[:, 1]

    acc = accuracy_score(sample_labels, y_pred)
    f1 = f1_score(sample_labels, y_pred)
    auc_score = roc_auc_score(sample_labels, y_proba)
    fpr, tpr, thresholds = roc_curve(sample_labels, y_proba)

    print("Final model evaluation on 10 samples:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC AUC:", auc_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f'Final Model (AUC = {auc_score:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Final 10 Samples')
    plt.legend(loc='lower right')
    plt.savefig("Final_10_Samples_ROC_Curve.png")
    plt.close()


def compute_and_plot_combined_shap(ensemble_model, sub_models, X_train):
    # 计算各模型SHAP值并加权组合，得到综合特征重要性
    mlp_train_proba = get_positive_probability(sub_models[0], X_train)
    tabnet_train_proba = get_positive_probability(sub_models[1], X_train.values)
    ft_train_proba = get_positive_probability(sub_models[2], X_train.values)
    X_train_meta = np.vstack([mlp_train_proba, tabnet_train_proba, ft_train_proba]).T

    ensemble_explainer = shap.TreeExplainer(ensemble_model)
    shap_values_ensemble = ensemble_explainer.shap_values(X_train_meta)
    if isinstance(shap_values_ensemble, list):
        ensemble_shap = np.mean(np.abs(shap_values_ensemble[1]), axis=0)
    else:
        ensemble_shap = np.mean(np.abs(shap_values_ensemble), axis=0)

    sample_size = min(50, X_train.shape[0])
    background = X_train.values[:sample_size]
    submodel_shap_list = []
    # 对每个子模型计算SHAP值（为加快速度只选部分背景数据）
    for i, model in enumerate(sub_models):
        model_func = lambda X: model.predict_proba(X)[:, 1]
        explainer = shap.KernelExplainer(model_func, background)
        shap_values = explainer.shap_values(background, nsamples=100)
        submodel_shap = np.mean(np.abs(shap_values), axis=0)
        submodel_shap_list.append(submodel_shap)

    # 加权组合得到每个原始特征的综合重要性
    combined_importance = np.zeros(X_train.shape[1])
    for i in range(len(sub_models)):
        combined_importance += ensemble_shap[i] * submodel_shap_list[i]

    feature_importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': combined_importance
    })
    top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(10)

    plt.figure(figsize=(10, 8))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel("Combined SHAP Importance")
    plt.title("Top 10 Feature Importance from Combined SHAP Values")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("combined_feature_importance.png")
    plt.close()
    print("Combined feature importance bar chart saved as 'combined_feature_importance.png'")


def main():
    # 主流程：训练初始模型、优化、集成、评估及解释
    metrics_summary = {}

    print("Step 1: 训练初始模型")
    mlp_model, mlp_metrics = train_mlp(X_train, y_train, X_test, y_test)
    metrics_summary["MLP"] = mlp_metrics
    plot_roc_curve(mlp_metrics, "MLP")

    tabnet_model, tabnet_metrics = train_tabnet(X_train, y_train, X_test, y_test)
    metrics_summary["TabNet"] = tabnet_metrics
    plot_roc_curve(tabnet_metrics, "TabNet")

    ft_model, ft_metrics = train_ft_transformer(X_train, y_train, X_test, y_test)
    metrics_summary["FT-Transformer"] = ft_metrics
    plot_roc_curve(ft_metrics, "FT-Transformer")

    print("\nStep 2: 优化模型及特征重要性分析")
    opt_mlp_model, opt_mlp_metrics, mlp_feat_imp = optimize_mlp(X_train, y_train, X_test, y_test)
    metrics_summary["Optimized MLP"] = opt_mlp_metrics

    opt_tabnet_model, opt_tabnet_metrics, tabnet_feat_imp = optimize_tabnet(X_train, y_train, X_test, y_test)
    metrics_summary["Optimized TabNet"] = opt_tabnet_metrics

    opt_ft_model, opt_ft_metrics, ft_feat_imp = optimize_ft_transformer(X_train, y_train, X_test, y_test)
    metrics_summary["Optimized FT-Transformer"] = opt_ft_metrics

    plot_roc_curve(opt_mlp_metrics, "Optimized_MLP")
    plot_roc_curve(opt_tabnet_metrics, "Optimized_TabNet")
    plot_roc_curve(opt_ft_metrics, "Optimized_FT-Transformer")

    print("\nStep 3: 利用 XGBoost 对三个优化后的模型进行集成")
    ensemble_model, ensemble_metrics, X_test_meta = ensemble_with_xgboost(
        opt_mlp_model, opt_tabnet_model, opt_ft_model, X_train, y_train, X_test, y_test)
    metrics_summary["Ensemble XGBoost"] = ensemble_metrics
    plot_roc_curve(ensemble_metrics, "Ensemble_XGBoost")

    print("\nStep 4: 优化集成模型")
    mlp_train_proba = get_positive_probability(opt_mlp_model, X_train)
    tabnet_train_proba = get_positive_probability(opt_tabnet_model, X_train.values)
    ft_train_proba = get_positive_probability(opt_ft_model, X_train.values)
    X_train_meta = np.vstack([mlp_train_proba, tabnet_train_proba, ft_train_proba]).T
    optimized_ensemble_model, optimized_ensemble_metrics, ensemble_feat_imp = optimize_ensemble_xgboost(
        X_train_meta, y_train, X_test_meta, y_test)
    metrics_summary["Optimized Ensemble XGBoost"] = optimized_ensemble_metrics
    plot_roc_curve(optimized_ensemble_metrics, "Optimized_Ensemble_XGBoost")

    print("\nStep 5: 对10个样本数据进行预测并评估")
    sample_data = X_test.sample(10, random_state=42)
    sample_labels = y_test.loc[sample_data.index]
    try:
        actual_samples = pd.read_excel('data_test.xlsx', header=0)
        sample_data = actual_samples
    except FileNotFoundError:
        print("未找到 data_test.xlsx 文件，使用 sample_data 作为新样本数据。")
    final_model_test(optimized_ensemble_model,
                     [opt_mlp_model, opt_tabnet_model, opt_ft_model],
                     sample_data,
                     sample_labels)

    print("\nStep 6: 计算并绘制特征综合重要性 (SHAP)")
    compute_and_plot_combined_shap(optimized_ensemble_model,
                                   [opt_mlp_model, opt_tabnet_model, opt_ft_model],
                                   X_train)

    print("\n===== 所有模型的指标汇总 =====")
    for model_name, metrics in metrics_summary.items():
        print(f"\n模型：{model_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")


if __name__ == '__main__':
    # 重定向输出到文件，便于查看训练和评估结果
    with open('output.txt', 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            main()
