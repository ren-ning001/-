import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

LABEL_MAP = {
    'negative': 0,
    'neutral': 1,
    'positive': 2,
    'null': -1 
}

class QwenDataset(Dataset):
    def __init__(self, df, data_dir, processor, mode='multimodal'):
        """
        mode: 'multimodal' (默认), 'text_only', 'image_only'
        """
        self.df = df
        self.data_dir = data_dir
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        guid = str(row['guid']).split('.')[0] 

        txt_path = os.path.join(self.data_dir, f"{guid}.txt")
        text_content = ""
        if self.mode != 'image_only' and os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read().strip()
            except: 
                text_content = "unknown"
        
        if self.mode == 'image_only':
            text_content = "Analyze the sentiment of the image." 

        image = None
        if self.mode != 'text_only':
            img_path = os.path.join(self.data_dir, f"{guid}.jpg")
            try:
                image = Image.open(img_path).convert('RGB')
            except:
                image = Image.new('RGB', (256, 256), (0, 0, 0))
        else:
            image = Image.new('RGB', (28, 28), (0, 0, 0))

        prompt_text = f"Analyze the sentiment. Text content: {text_content}"
        
        content_list = []
        if image is not None:
            content_list.append({"type": "image", "image": image})
        content_list.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content_list}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        label_id = LABEL_MAP.get(str(row['tag']), -1)

        return {
            'guid': guid,
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'], 
            'image_grid_thw': inputs['image_grid_thw'], 
            'label': torch.tensor(label_id, dtype=torch.long)
        }

def qwen_collate_fn(batch):
    guids = [item['guid'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=151643) 
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    image_grid_thw = torch.cat([item['image_grid_thw'] for item in batch], dim=0)

    return {
        'guid': guids,
        'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw,
        'label': labels
    }

def get_data_loaders(data_dir, train_file, test_file, batch_size=16, mode='multimodal'):
    os.environ['TRANSFORMERS_OFFLINE'] = '0'  
    
    try:
        model_path = "/home/bygpu/Downloads/qwen2-vl-2b-instruct-local"
        processor = Qwen2VLProcessor.from_pretrained(
            model_path,
            min_pixels=256*256,
            max_pixels=1024*1024,
            trust_remote_code=True
        )
        print("Processor loaded from local files")
    except Exception as e:
        print(f"Local processor loading failed: {e}")
        print("Trying online loading...")
        
        processor = Qwen2VLProcessor.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            min_pixels=256*256,
            max_pixels=1024*1024
        )
        print(" Processor loaded online")
    
    full_train_df = pd.read_csv(train_file, dtype={'guid': str})
    test_df = pd.read_csv(test_file, dtype={'guid': str})
    
    train_df, val_df = train_test_split(full_train_df, test_size=0.1, random_state=42, stratify=full_train_df['tag'])

    train_dataset = QwenDataset(train_df, data_dir, processor, mode=mode)
    val_dataset = QwenDataset(val_df, data_dir, processor, mode=mode)
    test_dataset = QwenDataset(test_df, data_dir, processor, mode=mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=qwen_collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=qwen_collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=qwen_collate_fn, num_workers=2)

    return train_loader, val_loader, test_loader
