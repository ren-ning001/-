# 项目概述
本项目针对文本-图像配对数据完成情感三分类任务，支持 positive（积极）、neutral（中性）、negative（消极）三类标签预测。通过设计多模态融合模型，整合文本与图像的语义信息，完成数据预处理、训练集-验证集划分、超参数调优、模型训练及测试集预测全流程，最终输出测试集情感标签结果。
## 多模态情感分类模型（MPS）
## 目录结构
```
AI第五次作业（MPS）
├── AI第五次作业（MPS）.ipynb #MPS部分代码文件
├── requirements（MPS）.txt  # 项目依赖清单
├── best_model_clip_balanced.pth  # 默认最佳模型权重文件（代码指定路径）
├── best_model_exp_xxxxxx.pth     # 按实验ID保存的最佳模型权重（自动生成）
├── ablation_study_results.json   # 消融实验结果（JSON格式，记录各配置性能）
├── ablation_study_results.csv    # 消融实验结果（CSV格式，便于表格分析）
├── ablation_study_results.png    # 消融实验可视化图表（F1/准确率对比）
├── ablation_detailed_comparison.png  # 详细对比图表（单模态vs多模态、融合策略对比）
├── training_curves_exp_xxxxxx.png    # 训练曲线（Loss/F1/Precision/Recall）
├── bad_cases_exp_xxxxxx.txt      # Bad Case分析文件（错误案例详情）
├── submission_exp_xxxxxx_2025xxxx.txt # 测试集预测结果（带时间戳）
└── results_exp_xxxxxx_2025xxxx.json  # 实验完整结果（含配置/指标/预测分布）
```

## 环境依赖
执行以下命令安装依赖：
```bash
pip install -r requirements（MPS）.txt
```

## 数据说明
### 数据规模与分布
- 训练集：4000条样本，含 positive（2388条，59.7%）、neutral（419条，10.5%）、negative（1193条，29.8%）三类标签，存在标签不平衡现象
- 测试集：511条样本，仅含guid字段，无情感标签
- 数据完整性：训练集与测试集guid无重叠，文本-图像配对完整率100%（每个guid对应1个.txt文本文件和1个.jpg图像文件）
- 验证集：从训练集里面按照一定比例进行划分
  
### 数据格式规范
- 训练集文件（train.txt）：逗号分隔，字段为`guid,tag`
- 测试集文件（test_without_label.txt）：逗号分隔，字段为`guid`
- 文本文件：命名格式为`{guid}.txt`，编码为UTF-8，平均长度97字符，长度范围37-139字符
- 图像文件：命名格式为`{guid}.jpg`，格式统一为JPEG，平均尺寸575×526像素，宽高比约1.20

## 核心功能模块
### 1. 数据预处理
#### 文本预处理
- 清洗规则：移除URL、@提及、#话题标签、emoji表情，扩展英文缩写（如"won't"→"will not"）、替换网络俚语（如"lol"→"laugh out loud"）
- 分词与编码：使用DeBERTa/BERTTokenizer分词，最大序列长度64（基于95%分位数统计），padding/truncation处理
- 停用词处理：可选移除英文停用词（如"the"、"and"），提升语义提取效率

#### 图像预处理
- 训练集增强：随机水平翻转（p=0.3）、随机旋转（±10°）、颜色抖动（亮度/对比度/饱和度±0.2）、随机裁剪（224×224）
- 测试集标准化：Resize至224×224，归一化（均值[0.485, 0.456, 0.406]，标准差[0.229, 0.224, 0.225]）
- 异常处理：缺失图像填充默认值，避免训练中断

#### 数据集划分
- 划分比例：训练集:验证集比例不等，会有所变化，采用分层抽样（stratify=train_df['tag']），保证标签分布一致
- 随机种子：固定seed=42，确保实验可复现

### 2. 多模态融合模型
#### （1）基础融合模型（MultimodalFusionModel）
- 文本编码器：BERT-base-uncased（输出768维特征）
- 图像编码器：ResNet50（移除分类头，输出2048维特征）
- 融合策略：特征拼接（768+2048）→ 全连接层（256维）→ Dropout（0.3）
- 分类头：全连接层→输出3类概率

#### （2）增强融合模型（ImprovedFusionModel）
- 文本编码器：DeBERTa-v3-small（冻结基础参数，微调顶层）
- 图像编码器：自定义卷积网络（32→64→128维特征）
- 融合策略：特征拼接→双全连接层（256→128）→ Dropout（0.3）
- 优化点：减少参数量，提升训练速度，适配服务器资源

#### （3）CLIP融合模型（SimpleCLIPFusionModel）
- 文本编码器：CLIP文本模型（openai/clip-vit-base-patch32，输出512维特征）
- 图像编码器：CLIP视觉模型（输出768维特征）
- 融合策略：特征投影至同一维度→拼接→注意力融合→分类
- 优势：预训练模型具备更强的跨模态对齐能力，标签不平衡场景表现更优

#### （4）平衡CLIP模型（BalancedCLIPFusionModel）
- 针对标签不平衡优化：中性样本权重提升1.5倍，采用Focal Loss降低易分类样本权重
- 融合策略：支持late fusion（特征拼接）、early fusion（特征相加）、attention fusion（注意力加权）三种模式
- 后处理：低置信度预测（置信度<0.8）自动调整为中性标签，提升预测稳定性

### 3. 训练与优化
#### 训练配置
- 优化器：AdamW（学习率2e-5，权重衰减0.01）
- 学习率调度：余弦退火调度（warmup步数=1个epoch）
- 损失函数：基础版用CrossEntropyLoss，增强版用带类别权重的CrossEntropyLoss/Focal Loss
- 早停机制：每一段代码早停不一样

#### 超参数调优
支持自动调优的核心参数：
| 参数                | 可选范围          | 最优配置推荐 |
|---------------------|-------------------|--------------|
| batch_size          | 8/16/32           | 16（GPU）/8（CPU） |
| learning_rate       | 1e-5/2e-5/3e-5    | 2e-5         |
| dropout_rate        | 0.2/0.3/0.4       | 0.3          |
| max_text_length     | 32/64/128         | 64           |
| image_size          | 224/256           | 224          |

#### 训练监控
- 实时输出：每轮训练损失、训练集F1/准确率，验证集F1/准确率/精确率/召回率
- 可视化：训练曲线（损失、F1、准确率）

### 4. 测试集预测
- 加载最优模型权重，批量预测测试集样本
- 后处理：低置信度样本（置信度<0.6）自动修正为中性，平衡预测分布

## 核心功能与结果文件说明
### 1. 核心实验能力
| 功能模块                | 核心特性                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| 数据处理                | 文本清洗、文本/图像增强、类别平衡、缺失数据处理                          |
| 模型架构                | CLIP预训练+多模态融合（Late/Early/Attention）、LayerNorm优化            |
| 训练策略                | 平衡损失函数、学习率调度、早停、梯度裁剪、权重衰减                       |
| 评估分析                | F1/准确率/精确率/召回率、混淆矩阵、Bad Case分析、错误类型分布            |
| 可视化                  | 消融实验对比图、训练曲线、多维度性能雷达图                              |
| 结果输出                | 测试集预测文件、实验配置/指标文件、模型权重、错误案例分析文件            |

### 2. 关键结果文件详解
| 文件名                                  | 生成时机               | 核心内容                                                                 |
|-----------------------------------------|------------------------|--------------------------------------------------------------------------|
| `ablation_study_results.json/csv`       | 消融实验完成后         | 所有模态/融合策略的性能指标（F1/准确率/样本数）                          |
| `ablation_study_results.png`            | 消融实验可视化后       | 所有实验的F1分数、准确率横向对比条形图                                   |
| `ablation_detailed_comparison.png`      | 消融实验可视化后       | 单模态vs多模态对比、融合策略对比、性能雷达图                             |
| `best_model_exp_xxxxxx.pth`             | 训练过程（早停触发）   | 最佳模型权重（含epoch/优化器/调度器状态/配置）                           |
| `training_curves_exp_xxxxxx.png`        | 训练完成后             | 训练/验证集的Loss/F1/Precision/Recall曲线                                |
| `bad_cases_exp_xxxxxx.txt`              | 训练完成后             | 错误案例详情（GUID/真实标签/预测标签/置信度/文本/概率分布）              |
| `submission_exp_xxxxxx_2025xxxx.txt`    | 测试集预测完成后       | 测试集最终预测结果（格式：guid,tag），带时间戳避免覆盖                   |
| `results_exp_xxxxxx_2025xxxx.json`      | 测试集预测完成后       | 完整实验结果（配置/设备/指标/预测分布/文件路径）                        |

## 完整执行流程
### 环境准备
1. 确保数据目录`data/`下包含训练集、测试集标签文件及对应文本/图像文件
2. 安装依赖：`pip install -r requirements（MPS）.txt`

### 全流程执行
启动 Jupyter Notebook：
```bash
jupyter notebook
```
2. 在浏览器中打开 `AI第五次作业（MPS）.ipynb`，选择内核「Python」；
3. 运行方式：
   - 一键运行：菜单栏「Kernel」→「Restart & Run All」，自动完成所有实验步骤（数据处理、模型训练、可视化、数据分析）；
   - 分步运行：按「Shift + Enter」逐单元格执行，观察每一步输出结果。

### 结果查看
- 模型权重：`best_model.pth`
- 测试集预测结果：`test_predictions.txt`
- 错误案例分析：`bad_cases_*.txt`（仅CLIP模型支持）

## 模型性能指标
### 验证集最优性能（CLIP平衡版模型）
- 加权F1值：0.7621
- 准确率：0.7650
- 各类别F1：positive(0.8411)、neutral(0.4545)、negative(0.7124)
- 训练耗时：15轮约1.5小时（GPU环境）

### 性能对比（不同模型架构）
| 模型类型          | 加权F1  | 准确率  | 训练速度 | 适用场景               |
|-------------------|---------|---------|----------|------------------------|
| 基础融合模型      | 0.6329  | 0.6613  | 快       | 资源有限，快速验证      |
| 增强融合模型      | 0.6974  | 0.6967  | 中       | 平衡性能与速度         |
| CLIP融合模型      | 0.7391  | 0.7333  | 中-慢    | 追求高准确率           |
| CLIP平衡模型      | 0.7621  | 0.7650  | 中-慢    | 标签不平衡场景（推荐） |

# 多模态情感分类模型（服务器部分）
## 该服务器部分模型链接为 https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
## 目录结构
```
AI第五次作业（服务器）/
├── main.py               # 主执行入口（整合数据处理、训练、预测逻辑）
├── dataset.py            # 数据加载、预处理、训练集划分验证集
├── model.py              # 多模态融合模型定义
├── requirements（服务器）.txt  # 项目依赖
├── results/              # 预测结果与模型权重保存目录
│   ├── test_predictions.txt  # 测试集情感标签预测结果
│   └── best_model.pth    # 训练得到的最优模型权重
```

## 环境依赖
执行以下命令安装依赖：
```bash
pip install -r requirements（服务器）.txt
```
## 完整执行流程
### 在终端输入
```bash
python main.py
```
#### 代码文件说明
#### （1）数据预处理与验证集划分（dataset.py 实现）
`dataset.py`中`MultimodalDataset`类负责数据加载，`split_train_val`函数完成训练集/验证集划分：
- 加载训练集文本+图像数据，完成文本分词（BERTTokenizer）、图像resize/归一化；
- 按比例随机划分训练/验证集，生成索引文件；
- 测试集数据加载适配`test_without_label.txt`格式，关联图像文件路径。

#### （2）模型训练与超参数调优（model.py + main.py 实现）
- `model.py`定义多模态融合模型（BERT文本编码器 + ResNet50图像编码器 + 拼接融合层 + 分类头）；
- `main.py`训练逻辑：
  1. 初始化模型、AdamW优化器、交叉熵损失函数；
  2. 加载`dataset.py`生成的DataLoader，迭代训练；
  3. 每轮验证集评估准确率，保存最优模型至`results/best_model.pth`；
  4. 内置超参数调优（学习率、批大小、dropout），支持早停机制（patience=5）。

#### （3）测试集情感标签预测（main.py 实现）
- 加载`results/best_model.pth`权重，批量预测`test_without_label.txt`对应样本；
- 预测结果保存至`results/test_predictions.txt`，格式为`样本ID,预测标签`。

# 模型迭代优化性能总结表
| 模型版本                  | 加权 F1  | 准确率   | positive F1 | neutral F1 | negative F1 | 核心优化点                     |
|---------------------------|----------|----------|-------------|------------|-------------|--------------------------------|
| 基础融合模型              | 0.6329   | 0.6613   | 0.76        | 0.16       | 0.51        | 基础文本 + 图像拼接融合        |
| 增强融合模型              | 0.6974   | 0.6967   | 0.77        | 0.39       | 0.67        | 交叉注意力 + 残差连接          |
| CLIP融合模型              | 0.7391   | 0.7333   | 0.82        | 0.36       | 0.73        | CLIP 预训练特征提取           |
| 平衡 CLIP 模型            | 0.7732   | 0.7717   | 0.84        | 0.48       | 0.74        | 类别权重 + 置信度后处理        |
| 自适应融合模型            | 0.7865   | 0.7833   | 0.85        | 0.52       | 0.75        | 多融合策略 + Bad Case 优化     |
| Qwen/Qwen2-VL-2B-Instruct模型 | -        | 0.8250   | -           | -          | -           | 大模型跨模态直接推理           |

## 参考资源
### 参考仓库
1. Hugging Face Transformers：https://github.com/huggingface/transformers  （文本编码器实现）
2. PyTorch Vision：https://github.com/pytorch/vision  （图像编码器实现）
4. CLIP官方仓库：https://github.com/openai/CLIP  （跨模态预训练模型参考）
   
### 参考论文
1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.（文本特征提取）
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.（图像特征提取）
3. Zadeh, A., et al. (2017). Tensor Fusion Network for Multimodal Sentiment Analysis. EMNLP.（跨模态融合策略参考）
4. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML.（CLIP模型架构）
