# 诈骗检测模型

本项目是一个基于多种机器学习和深度学习方法的诈骗检测系统，使用 `MLPClassifier`、`TabNetClassifier` 和 `FT-Transformer` 进行训练，并使用 `XGBoost` 进行模型融合。

## 📌 项目概述

该项目旨在利用机器学习方法检测欺诈行为，首先加载数据集，从 `data.xlsx` 读取数据，如果文件不存在，则自动生成模拟数据。随后对数据进行预处理，拆分为训练集和测试集，以便进行模型训练。项目采用 `MLPClassifier`（多层感知机）、`TabNetClassifier`（深度学习 TabNet 模型）和 `FT-Transformer`（基于 Transformer 的表格数据模型）进行训练，并对这些模型进行超参数优化，以提高预测能力。在此基础上，使用 `XGBoost` 进行模型融合，提升整体性能。最终，模型的效果通过 `Accuracy`、`F1 Score` 和 `ROC AUC` 进行评估，并绘制 `ROC` 曲线，以可视化模型的表现。此外，本项目采用 `SHAP` 进行特征重要性分析，以理解不同特征在欺诈检测中的贡献。最后，模型将被用于预测新的 10 个样本，并对其结果进行评估，以确保模型的可靠性和泛化能力。

---

## 📂 代码结构

```
├── main.py          # 主要代码文件
├── data.xlsx        # 训练数据
├── data_test.xlsx   # 测试数据
├── output.txt       # 训练和评估的日志输出
├── combined_feature_importance.png  # 特征重要性分析图
├── *_roc_curve.png  # 各模型的 ROC 曲线
└── README.md        # 本说明文档
```

---

## 📌 依赖环境

请确保安装以下 Python 库：

```bash
pip install numpy pandas matplotlib scikit-learn torch pytorch-tabnet shap xgboost openpyxl
```

---

## 🚀 快速开始

### 1️⃣ 运行模型训练和评估

在终端运行：

```bash
python main.py
```


### 2️⃣ 结果分析

执行完成后，关键结果将保存在：

- **`output.txt`**（日志文件，包含所有模型的训练和评估结果）
- **`*_roc_curve.png`**（各模型的 `ROC` 曲线）
- **`combined_feature_importance.png`**（综合特征重要性分析图）

---

## 📈 关键技术

- **多种模型训练**：使用 `MLP`、`TabNet` 和 `FT-Transformer` 进行诈骗检测。
- **模型融合**：利用 `XGBoost` 结合多个模型，提高准确性。
- **超参数优化**：对 `MLP`、`TabNet` 和 `FT-Transformer` 进行调优。
- **特征重要性分析**：利用 `SHAP` 计算特征贡献。

---

## 📌 评估指标

该项目主要采用以下指标评估模型表现：

- **准确率（Accuracy）**
- **F1 分数（F1 Score）**
- **ROC AUC 分数（ROC AUC）**
- **ROC 曲线**

---

## 📢 备注

- 如果 `data.xlsx` 或 `data_test.xlsx` 不存在，代码会自动生成模拟数据。
- 运行过程中，所有训练过程和评估结果都会保存到 `output.txt`。
- 该项目适用于二分类任务，可扩展到其他分类问题。

