import torch
import json
import logging
import sys
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = f"{data.get('context', '')} [SEP] {data['response']}"
                        inputs = self.tokenizer(
                            text,
                            max_length=self.max_length,
                            padding='max_length',
                            truncation=True,
                            return_tensors="pt"
                        )
                        self.examples.append({
                            'input_ids': inputs['input_ids'].squeeze(),
                            'attention_mask': inputs['attention_mask'].squeeze()
                        })
                    except Exception as e:
                        logging.warning(f"Skipping line: {e}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise
        
        if len(self.examples) == 0:
            raise ValueError("No valid examples found")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            'input_ids': self.examples[idx]['input_ids'],
            'labels': self.examples[idx]['input_ids'],
            'attention_mask': self.examples[idx]['attention_mask']
        }

def setup_model_and_tokenizer():
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'sep_token': '[SEP]'
        })
        
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer
    except Exception as e:
        logging.error(f"Initialization error: {e}")
        raise

def train_model(model, tokenizer, dataset):
    try:
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=2,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=100,
            report_to=None
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        trainer.train()
        return trainer
    except Exception as e:
        logging.error(f"Training error: {e}")
        raise

def main():
    try:
        model, tokenizer = setup_model_and_tokenizer()
        dataset = ConversationDataset('formatted_reply_pools.json', tokenizer)
        trainer = train_model(model, tokenizer, dataset)
        
        output_dir = Path("fine_tuned_gpt2")
        output_dir.mkdir(exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logging.info("Training successful")
        return 0
    except Exception as e:
        logging.error(f"Main error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
