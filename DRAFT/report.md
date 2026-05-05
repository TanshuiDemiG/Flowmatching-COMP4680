# COMP4680/8650 Flow Matching Report (Full High-Quality Template)

This document provides a **polished, submission-ready report template** for the Flow Matching assignment. All sections are written in formal academic English, with structured placeholders for results, figures, and analysis. Chinese annotations are provided under each section to explain writing requirements, format expectations, and related experiments.

---

# 1. Introduction

This report investigates flow matching models under different prediction parameterizations, including x-prediction, ε-prediction, and v-prediction. We aim to understand how different parameterizations affect model performance in both low-dimensional and high-dimensional settings. In particular, we study the scalability of these methods as ambient dimensionality increases, and analyze whether failures in v-prediction can be mitigated through architectural or training modifications.

Furthermore, we explore efficient sampling strategies based on MeanFlow, which extends standard flow matching by incorporating horizon-aware velocity modeling for improved few-step generation.

本报告研究了在不同预测参数化方案（包括x预测、ε预测和v预测）下的流匹配模型。我们的目标是了解不同的参数化方案如何影响模型在低维和高维场景下的性能。特别是，我们研究了随着环境维度增加这些方法的可扩展性，并分析了是否可以通过架构或训练调整来缓解v预测中的失效问题。

此外，我们还探索了基于MeanFlow的高效采样策略，该方法通过引入具有时域意识的速度建模来扩展标准流匹配，从而改善了多步流生成。

## 🟡 中文写作要求

### ✔ 内容要求

- 说明研究背景：flow matching + diffusion parameterization
- 明确三个研究问题：
  1. 高维空间行为差异
  2. v-prediction failure现象
  3. MeanFlow少步生成能力

### ✔ 写作格式

- 1段正式学术英语（100–150词）
- 不使用公式
- 不描述代码实现

### ✔ 关联实验

- Part 2 (核心分析)
- Part 3 (v-pred rescue)
- Part 4 (MeanFlow)

---

# 2. Implementation and Experimental Setup

We implement a flow matching framework based on a multilayer perceptron (MLP) conditioned on sinusoidal time embeddings. The model takes noisy inputs constructed via linear interpolation between clean data samples and Gaussian noise. Depending on the configuration, the model is trained to predict either clean data (x-prediction) or flow velocity (v-prediction), under corresponding loss formulations.

We conduct experiments on three synthetic datasets: Swiss roll, 8-mode Gaussian, and concentric circles. Each dataset is evaluated under multiple ambient dimensions, specifically D ∈ {2, 8, 32}, where higher-dimensional samples are generated via orthogonal projection from a 2D manifold. Optimization is performed using Adam with a learning rate of 1e-3 and batch size of 1024. Sampling is performed using Euler discretization of the probability flow ODE.

## 🟡 中文写作要求

### ✔ 必须说明四部分

#### 1. 模型结构

- MLP (5 layers, ReLU)
- sinusoidal time embedding (128-d)

#### 2. 数据设置

- swiss roll / gaussian / circle
- 2D → D=2/8/32 projection

#### 3. Training

- z_t = (1 - t)x + tε
- x-pred / v-pred
- MSE loss

#### 4. Sampling

- Euler ODE
- default 50 steps

### ✔ 写作风格

- concise but complete
- 不逐行解释代码

### ✔ 关联实验

- Part 1
- Part 2
- Part 3
- Part 4

---

# 3. Part 1: Warm-up Results (D=2 Sanity Check)

I first validate the correctness of our implementation using low-dimensional settings (D=2). Across all three toy datasets, the model successfully learns the underlying data distributions. The generated samples exhibit correct geometric structures, including spiral patterns, clustered Gaussian modes, and circular manifolds.

Additionally, we verify that high-dimensional samples projected back into 2D preserve the original structure, confirming the correctness of the projection pipeline.


我首先通过低维设置（D=2）验证了我们实现方案的正确性。在所有三个示例数据集上，模型都成功学习到了底层数据分布。生成的样本展现出了正确的几何结构，包括螺旋图案、簇状高斯模态以及圆形流形。

此外，我们验证了高维样本投影回2D后仍能保留原始结构，从而证实了投影流程的正确性。

## 📊 Results (Figures to be inserted)

- Swiss roll (GT vs Generated)
- Gaussian mixture (GT vs Generated)
- Circle (GT vs Generated)
- Projection consistency check (D=32 → 2D)

## 🟡 中文写作要求

### ✔ 必须包含

- 6张图（GT vs generated）
- D=2结果
- D=32 → 2D projection

### ✔ 写作重点

- 验证 implementation correctness
- 强调结构一致性

### ✔ 结论模板

- Model works correctly in low-dimensional regime

### ✔ 关联实验

- Part 1.1
- Part 1.2

---

# 4. Part 2: Parameterization Study

We evaluate different combinations of prediction parameterizations (x-prediction and v-prediction) and loss formulations (x-loss and v-loss) across varying ambient dimensions. Our results show a clear divergence in performance as dimensionality increases.

In low-dimensional settings (D=2), all configurations perform similarly. However, as the dimension increases to D=8 and D=32, x-prediction remains stable while v-prediction degrades significantly, often failing to capture the underlying data structure.

We further observe that the choice of loss function has a secondary effect compared to the choice of prediction parameterization.

## 📊 Results (Insert 36-grid visualization)

- Swiss roll: 3×4 grid
- Gaussian: 3×4 grid
- Circle: 3×4 grid

## 🟡 中文写作要求

### ✔ 必须写三层分析

#### 1. 现象

- D=2: all methods work
- D=8: partial degradation
- D=32: clear separation

#### 2. 对比分析

- x-pred: stable, preserves structure
- v-pred: unstable, collapses in high-D

#### 3. loss分析

- loss affects training stability
- does not change ranking fundamentally

### ✔ 核心结论必须写

- Prediction target dominates performance

### ✔ 关联实验

- 36 experiments grid

---

# 5. Part 3: v-prediction Rescue Study

We investigate whether the failure of v-prediction in high-dimensional settings can be mitigated through improved model capacity, training strategies, and numerical stabilization techniques.

We evaluate several approaches, including increasing model width, extending training duration, and modifying the time sampling distribution.

While certain improvements are observed, v-prediction remains less efficient and less stable compared to x-prediction under comparable computational budgets.

## 📊 Results

- Capacity scaling experiments
- Training step scaling
- t-clipping ablations

## 🟡 中文写作要求

### ✔ 必须写

- 至少3种rescue方法

#### 方法建议

- increase hidden dimension
- increase training steps
- clip t range

### ✔ 必须分析

- 是否有效
- 计算成本是否增加
- 是否稳定

### ✔ 核心结论

- partial recovery possible
- but inefficient compared to x-pred

### ✔ 关联实验

- Part 3 ablations

---

# 6. Part 4: Sampling Efficiency and MeanFlow

We evaluate the sampling efficiency of standard flow matching under different numbers of ODE integration steps and compare it with MeanFlow.

Standard flow matching requires a relatively large number of sampling steps to achieve high-quality generation. In contrast, MeanFlow introduces horizon-conditioned velocity modeling, enabling improved performance under few-step or even one-step generation settings.

## 📊 Results

- 1-step sampling
- 2-step sampling
- 5-step sampling
- 50-step baseline
- MeanFlow comparison

## 🟡 中文写作要求

### ✔ 必须写两部分

#### 1. Sampling efficiency

- step越多越好
- 1-step usually poor

#### 2. MeanFlow

- introduces horizon h
- learns averaged flow

### ✔ 必须解释

- why standard FM needs many steps
- why MeanFlow improves efficiency

### ✔ 关联实验

- Part 4 sampling curves
- MeanFlow experiments

---

# 7. Conclusion

This study demonstrates that prediction parameterization plays a critical role in flow matching performance, especially in high-dimensional settings. x-prediction consistently outperforms v-prediction due to its alignment with low-dimensional manifold structures, while v-prediction suffers from high-dimensional regression difficulty.

Furthermore, MeanFlow improves sampling efficiency by enabling better few-step generation, although one-step generation remains challenging for complex distributions.

## 🟡 中文写作要求

### ✔ 必须总结三点

- x vs v核心差异
- high-dim failure原因
- MeanFlow作用

### ✔ 写作风格

- concise academic conclusion
- 不引入新实验

### ✔ 关联实验

- 全部Part 1–4

---

# END OF REPORT TEMPLATE
