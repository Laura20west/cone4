import os
import io
import json
import boto3
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import logging
from dotenv import load_dotenv
import smart_open

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 Configuration - IMPORTANT: Never hardcode credentials!
S3_BUCKET = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX', 'model-weights/')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')  # Changed from hardcoded value
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')  # Changed from hardcoded value

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Initialize model and tokenizer as None (will be loaded later)
model = None
tokenizer = None

class S3Dataset(Dataset):
    def __init__(self, s3_uri, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Stream directly from S3
        try:
            with smart_open.open(s3_uri, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        self.examples.append({
                            'context': data.get('context', ''),
                            'response': data['response']
                        })
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {line}. Error: {e}")
        except Exception as e:
            logger.error(f"Error loading dataset from S3: {e}")
            raise

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = f"{self.examples[idx]['context']} [SEP] {self.examples[idx]['response']}"
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': inputs['input_ids'].squeeze()
        }

def upload_to_s3(local_path, s3_key):
    try:
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        logger.info(f"Successfully uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False

def download_from_s3(s3_key, local_path):
    try:
        s3.download_file(S3_BUCKET, s3_key, local_path)
        logger.info(f"Successfully downloaded s3://{S3_BUCKET}/{s3_key} to {local_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return False

class TrainingRequest(BaseModel):
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 5e-5

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

@app.on_event("startup")
async def startup_event():
    """Load model when starting the application"""
    global model, tokenizer
    try:
        # Check if model files exist locally, if not download from S3
        os.makedirs("./model", exist_ok=True)
        
        if not os.path.exists("./model/model.safetensors"):
            download_from_s3(f"{S3_PREFIX}model.safetensors", "./model/model.safetensors")
        
        if not os.path.exists("./model/tokenizer.json"):
            download_from_s3(f"{S3_PREFIX}tokenizer.json", "./model/tokenizer.json")
        
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("./model")
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        
        # Load the fine-tuned weights
        model.load_state_dict(torch.load("./model/model.safetensors"))
        model.eval()
        
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

@app.post("/start-training")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training as a background task"""
    background_tasks.add_task(run_training, request.dict())
    return {"status": "training_started", "message": "Training started in the background"}

def run_training(params):
    """Run the training process"""
    global model, tokenizer
    try:
        logger.info("Starting training process")
        
        # Initialize tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})
        
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))

        # Stream dataset from S3
        dataset_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}conversation_dataset.jsonl"
        dataset = S3Dataset(dataset_uri, tokenizer)

        # Training with checkpointing to S3
        training_args = TrainingArguments(
            output_dir="./output",
            num_train_epochs=params['epochs'],
            per_device_train_batch_size=params['batch_size'],
            learning_rate=params['learning_rate'],
            save_strategy="steps",
            save_steps=100,
            save_total_limit=2,
            gradient_accumulation_steps=4,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            report_to=None
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info("Starting model training")
        trainer.train()
        logger.info("Training completed successfully")

        # Save model parts to S3
        model_path = "./output/model.safetensors"
        torch.save(model.state_dict(), model_path)
        upload_to_s3(model_path, f"{S3_PREFIX}model.safetensors")

        tokenizer.save_pretrained("./output")
        upload_to_s3("./output/tokenizer.json", f"{S3_PREFIX}tokenizer.json")

        logger.info("Model artifacts uploaded to S3 successfully")
        
        # Reload the model for inference
        startup_event()

    except Exception as e:
        logger.error(f"Training failed: {e}")

@app.get("/load-model")
async def load_model():
    """Explicitly load or reload the model"""
    global model, tokenizer
    try:
        startup_event()
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-reply")
async def generate_reply(request: GenerateRequest):
    """Generate a reply using the fine-tuned model"""
    global model, tokenizer
    
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the response
        reply = reply.replace(request.prompt, "").strip()
        reply = reply.split('\n')[0]  # Take only the first line
        
        return {
            "status": "success",
            "reply": reply,
            "prompt": request.prompt
        }
    except Exception as e:
        logger.error(f"Error generating reply: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
