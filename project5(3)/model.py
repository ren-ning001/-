import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration
from peft import get_peft_model, LoraConfig
import os

class QwenClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(QwenClassifier, self).__init__()
        model_id = "/home/bygpu/Downloads/qwen2-vl-2b-instruct-local"
        
        if not os.path.exists(model_id):
            print(f"警告: 模型路径不存在: {model_id}")
            print("请确保模型已下载到该路径")
            raise FileNotFoundError(f"模型路径不存在: {model_id}")
        
        print(f"找到模型路径: {model_id}")
        print(f"Loading model in BFloat16...")

        os.environ['TRANSFORMERS_OFFLINE'] = '0'
        
        try:
            self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            print(" 模型加载成功")
            
        except Exception as e:
            print(f"BFloat16加载失败: {e}")
            print("尝试使用 float16...")
            
            try:
                self.qwen = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                print("模型加载成功 (使用 float16)")
                
            except Exception as e2:
                print(f"❌ 所有加载方式都失败: {e2}")
                raise RuntimeError(f"无法加载模型: {e2}")
        
        # LoRA 配置
        peft_config = LoraConfig(
            r=16, 
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        self.qwen = get_peft_model(self.qwen, peft_config)
        self.qwen.print_trainable_parameters() 

        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        ).to(torch.float32) 

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = self.qwen(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden_state = outputs.hidden_states[-1]
        
        batch_size = input_ids.shape[0]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        sentence_emb = last_hidden_state[batch_indices, sequence_lengths]

        logits = self.classifier(sentence_emb.to(torch.float32))
        return logits
