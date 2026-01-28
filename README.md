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

## 参考资源
### 参考仓库
1. Hugging Face Transformers：https://github.com/huggingface/transformers  （文本编码器实现）
2. PyTorch Vision：https://github.com/pytorch/vision  （图像编码器实现）

### 参考论文
1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.（文本特征提取）
2. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.（图像特征提取）
3. Zadeh, A., et al. (2017). Tensor Fusion Network for Multimodal Sentiment Analysis. EMNLP.（跨模态融合策略参考）

## 结果说明
- 训练过程中，验证集准确率会实时打印，最优模型权重保存至`results/best_model.pth`；
- 测试集预测结果保存在`results/test_predictions.txt`，每行格式为`样本ID,positive/neutral/negative`；
- 可在`main.py`中调用`evaluate`函数，输出训练/验证集的准确率、F1值等评估指标。
