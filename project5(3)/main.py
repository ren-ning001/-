import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from dataset import get_data_loaders, LABEL_MAP
from model import QwenClassifier

os.environ['TRANSFORMERS_OFFLINE'] = '0'

BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-4
DATA_DIR = './data' 
TRAIN_FILE = './train.txt'
TEST_FILE = './test_without_label.txt'
SAVE_MODEL_NAME = 'best_qwen_lora_optimized.pth'
RESULT_DIR = './results'

os.makedirs(RESULT_DIR, exist_ok=True)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    if torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        scaler = None

    loop = tqdm(dataloader, desc='Train')
    for batch in loop:
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            image_grid_thw = batch['image_grid_thw'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask, pixel_values, image_grid_thw)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids, attention_mask, pixel_values, image_grid_thw)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=correct/total)
            
        except Exception as e:
            print(f"❌ 训练batch失败: {e}")
            continue
        
    return total_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                for batch in dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    pixel_values = batch['pixel_values'].to(device)
                    image_grid_thw = batch['image_grid_thw'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(input_ids, attention_mask, pixel_values, image_grid_thw)
                    loss = criterion(outputs, labels)
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        else:
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                image_grid_thw = batch['image_grid_thw'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, pixel_values, image_grid_thw)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
    return total_loss / len(dataloader), correct / total

def plot_history(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(RESULT_DIR, 'training_curves.png'))
    print("Curves saved.")

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        print("\n1. 准备数据...")
        train_loader, val_loader, test_loader = get_data_loaders(
            DATA_DIR, TRAIN_FILE, TEST_FILE, BATCH_SIZE, mode='multimodal'
        )
        print(f"  训练集: {len(train_loader.dataset)} 个样本")
        print(f"  验证集: {len(val_loader.dataset)} 个样本")
        print(f"  测试集: {len(test_loader.dataset)} 个样本")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        print("\n2. 初始化模型...")
        model = QwenClassifier(num_classes=3)
        model.to(device)
        print("模型初始化成功")
        
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
        scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1*len(train_loader)*EPOCHS), len(train_loader)*EPOCHS)
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    
    print("\n3. 开始训练...")
    print(f"   总轮数: {EPOCHS}")
    print(f"   批大小: {BATCH_SIZE}")
    print(f"   学习率: {LEARNING_RATE}")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        try:
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            v_loss, v_acc = evaluate(model, val_loader, criterion, device)
            
            train_losses.append(t_loss); val_losses.append(v_loss)
            train_accs.append(t_acc); val_accs.append(v_acc)
            
            print(f"   Train Loss: {t_loss:.4f} | Acc: {t_acc:.4f}")
            print(f"   Val   Loss: {v_loss:.4f} | Acc: {v_acc:.4f}")
            
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                trainable_state = {k: v for k, v in model.state_dict().items() if v.requires_grad}
                torch.save(trainable_state, os.path.join(RESULT_DIR, SAVE_MODEL_NAME))
                print(f"    Best Model Saved (Acc: {v_acc:.4f})")
                
        except Exception as e:
            print(f"❌ 训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            break

    try:
        plot_history(train_losses, val_losses, train_accs, val_accs)
    except Exception as e:
        print(f" 绘制训练曲线失败: {e}")
    print("\n4. 加载最佳模型进行预测...")
    try:
        if os.path.exists(os.path.join(RESULT_DIR, SAVE_MODEL_NAME)):
            checkpoint = torch.load(os.path.join(RESULT_DIR, SAVE_MODEL_NAME))
            model.load_state_dict(checkpoint, strict=False) 
            model.eval()
            print(" 最佳模型加载成功")
        else:
            print("  未找到保存的模型，使用当前模型")
            model.eval()
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        model.eval()
    
    idx_to_label = {v: k for k, v in LABEL_MAP.items()}
    predictions = []
    
    try:
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for batch in tqdm(test_loader, desc="预测中"):
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        pixel_values = batch['pixel_values'].to(device)
                        image_grid_thw = batch['image_grid_thw'].to(device)
                        
                        outputs = model(input_ids, attention_mask, pixel_values, image_grid_thw)
                        _, preds = torch.max(outputs, 1)
                        
                        for guid, pred_idx in zip(batch['guid'], preds):
                            predictions.append(f"{guid},{idx_to_label[pred_idx.item()]}")
            else:
                for batch in tqdm(test_loader, desc="预测中"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    pixel_values = batch['pixel_values'].to(device)
                    image_grid_thw = batch['image_grid_thw'].to(device)
                    
                    outputs = model(input_ids, attention_mask, pixel_values, image_grid_thw)
                    _, preds = torch.max(outputs, 1)
                    
                    for guid, pred_idx in zip(batch['guid'], preds):
                        predictions.append(f"{guid},{idx_to_label[pred_idx.item()]}")
        
        output_file = os.path.join(RESULT_DIR, 'submit_qwen_optimized.txt')
        with open(output_file, 'w') as f:
            f.write("guid,tag\n")
            f.write("\n".join(predictions))
        print(f"预测完成，结果保存到: {output_file}")
        print(f"   总预测数量: {len(predictions)}")
        
    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n程序执行完成!")

if __name__ == '__main__':
    main()
