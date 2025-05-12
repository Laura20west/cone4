import torch
import json
import logging
import sys
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Configure logging
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
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            logging.info(f"Successfully loaded dataset from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise
        
        self.examples = []
        self._prepare_examples()
    
    def _prepare_examples(self):
        valid_examples = 0
        for category in self.data:
            for example in self.data[category]:
                try:
                    if not isinstance(example, dict):
                        raise ValueError("Example is not a dictionary")
                    if 'context' not in example or 'response' not in example:
                        raise ValueError("Missing required fields")
                    
                    text = f"{example['context']} [SEP] {example['response']}"
                    tokenized = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors="pt"
                    )
                    self.examples.append({
                        'input_ids': tokenized['input_ids'].squeeze(),
                        'attention_mask': tokenized['attention_mask'].squeeze()
                    })
                    valid_examples += 1
                except Exception as e:
                    logging.warning(f"Skipping invalid example in category {category}: {e}")
        
        logging.info(f"Prepared {valid_examples} valid training examples")
        if valid_examples == 0:
            raise ValueError("No valid training examples found")
    
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
        logging.info("Initializing tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({
            'pad_token': '[PAD]',
            'sep_token': '[SEP]',
            'eos_token': '[EOS]'
        })
        
        logging.info("Initializing model...")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"Model/tokenizer initialization failed: {e}")
        raise

def train_model(model, tokenizer, dataset):
    try:
        logging.info("Configuring training arguments...")
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
            no_cuda=not torch.cuda.is_available(),
            report_to=None  # Changed from "none" to None for compatibility
        )
        
        logging.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )
        
        logging.info("Starting training...")
        trainer.train()
        
        return trainer
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

def main():
    try:
        logging.info("System Information:")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        
        model, tokenizer = setup_model_and_tokenizer()
        dataset = ConversationDataset('formatted_reply_pools.json', tokenizer)
        trainer = train_model(model, tokenizer, dataset)
        
        logging.info("Saving model...")
        output_dir = Path("fine_tuned_gpt2")
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logging.info("Training completed successfully!")
        return 0
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
