from transformers import GPT2LMHeadModel, GPT2Tokenizer
from app.s3_utils import S3Manager
from app.config import Config
import torch
import os
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.s3 = S3Manager()
    
    def load_model(self):
        try:
            # Check if model exists locally
            local_model_dir = os.path.join(Config.LOCAL_MODEL_DIR, Config.MODEL_VERSION)
            
            if not os.path.exists(local_model_dir):
                # Try to download from S3
                if not self.s3.download_model():
                    logger.warning("No model found in S3, will need to train first")
                    return False
            
            # Load from local
            self.tokenizer = GPT2Tokenizer.from_pretrained(local_model_dir)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = GPT2LMHeadModel.from_pretrained(local_model_dir)
            self.model.eval()
            
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        if not self.model or not self.tokenizer:
            return None
            
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None