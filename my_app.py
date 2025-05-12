from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import json
from pathlib import Path
import logging
from typing import List, Optional
import os

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Configuration
DATASET_PATH = "conversation_dataset.jsonl"
MODEL_DIR = "fine_tuned_gpt2"
BASE_MODEL = "gpt2"

# Load model and tokenizer
try:
    tokenizer = GPT2Tokenizer.from_pretrained(
        MODEL_DIR if os.path.exists(MODEL_DIR) else BASE_MODEL
    )
    tokenizer.add_special_tokens({
        'pad_token': '[PAD]',
        'sep_token': '[SEP]',
        'eos_token': '[EOS]'
    })
    
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_DIR if os.path.exists(MODEL_DIR) else BASE_MODEL
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

class TrainingRequest(BaseModel):
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    overwrite: bool = False

class ConversationExample(BaseModel):
    context: str
    response: str

class ConversationDataset(Dataset):
    def __init__(self, file_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    self.examples.append({
                        'context': data.get('context', ''),
                        'response': data['response']
                    })
            logging.info(f"Loaded {len(self.examples)} examples from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise
&& python -m spacy download en_core_web_md && python -m nltk.downloader wordne
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = f"{self.examples[idx]['context']} [SEP] {self.examples[idx]['response']}"
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze(),
            'labels': tokenized['input_ids'].squeeze()
        }

@app.get("/dataset", response_model=List[ConversationExample])
async def get_dataset(limit: int = 10):
    """Endpoint to view the training dataset"""
    try:
        examples = []
        with open(DATASET_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                data = json.loads(line)
                examples.append(ConversationExample(**data))
        return examples
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model")
async def train_model(request: TrainingRequest):
    """Endpoint to trigger model training"""
    try:
        # Check if model exists and overwrite flag
        if os.path.exists(MODEL_DIR) and not request.overwrite:
            return JSONResponse(
                status_code=400,
                content={"message": "Model already exists. Set overwrite=True to retrain."}
            )

        # Load dataset
        dataset = ConversationDataset(DATASET_PATH, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./training_results',
            num_train_epochs=request.epochs,
            per_device_train_batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            save_steps=500,
            save_total_limit=2,
            logging_steps=100,
            report_to=None,
            no_cuda=not torch.cuda.is_available()
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        # Train and save
        trainer.train()
        trainer.save_model(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)

        return {"message": "Training completed successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_response(context: str, max_length: int = 100):
    """Generate a response given a context"""
    try:
        input_text = f"{context} [SEP]"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        full_text = tokenizer.decode(output[0], skip_special_tokens=True)
        response = full_text.split('[SEP]')[-1].strip()
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
