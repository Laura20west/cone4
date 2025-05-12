import os
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

# S3 Configuration
S3_BUCKET = os.getenv('S3_BUCKET')
S3_PREFIX = os.getenv('S3_PREFIX', 'model/')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Initialize model and tokenizer
model = None
tokenizer = None

class S3Dataset(Dataset):
    def __init__(self, s3_uri, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        transport_params = {
            'session': boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY,
                aws_secret_access_key=AWS_SECRET_KEY
            )
        }
        
        try:
            with smart_open.open(s3_uri, 'r', encoding='utf-8', transport_params=transport_params) as f:
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
        logger.info(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False

def download_from_s3(s3_key, local_path):
    try:
        s3.download_file(S3_BUCKET, s3_key, local_path)
        logger.info(f"Downloaded s3://{S3_BUCKET}/{s3_key} to {local_path}")
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
def startup_event():
    """Load model on startup"""
    global model, tokenizer
    try:
        os.makedirs("./model", exist_ok=True)
        
        # List and download all model files from S3
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)
        if 'Contents' not in response:
            logger.error("No files found in S3 bucket.")
            raise Exception("No model files in S3.")
        
        for obj in response['Contents']:
            s3_key = obj['Key']
            local_path = os.path.join("./model", os.path.relpath(s3_key, S3_PREFIX))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not download_from_s3(s3_key, local_path):
                raise Exception(f"Failed to download {s3_key}")
        
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained("./model")
        model = GPT2LMHeadModel.from_pretrained("./model")
        model.eval()
        logger.info("Model and tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

@app.post("/start-training")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start training as a background task"""
    background_tasks.add_task(run_training, request.dict())
    return {"status": "training_started", "message": "Training started in the background"}

def run_training(params):
    """Run training and upload model to S3"""
    global model, tokenizer
    try:
        logger.info("Initializing training...")
        
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({'pad_token': '[PAD]', 'sep_token': '[SEP]'})
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.resize_token_embeddings(len(tokenizer))

        # Load dataset from S3
        dataset_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}conversation_dataset.json"
        dataset = S3Dataset(dataset_uri, tokenizer)

        # Training arguments
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

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        # Train and save
        logger.info("Training model...")
        trainer.train()
        
        # Save model and tokenizer
        output_dir = "./output"
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Upload all files to S3
        for file_name in os.listdir(output_dir):
            local_path = os.path.join(output_dir, file_name)
            s3_key = f"{S3_PREFIX}{file_name}"
            upload_to_s3(local_path, s3_key)
        
        logger.info("Model updated and uploaded to S3")
        
        # Reload model
        startup_event()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

@app.get("/load-model")
async def load_model():
    """Manually reload the model from S3"""
    try:
        startup_event()
        return {"status": "success", "message": "Model reloaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-reply")
async def generate_reply(request: GenerateRequest):
    """Generate a reply using the model"""
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        outputs = model.generate(
            inputs.input_ids,
            max_length=request.max_length,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )
        
        reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reply = reply.replace(request.prompt, "").strip()
        reply = reply.split('\n')[0]
        
        return {"status": "success", "reply": reply}
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
