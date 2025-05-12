import boto3
from app.config import Config
import os
import logging

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=Config.AKIA5GU2673EBGPIRSFW ,
            aws_secret_access_key=Config.6u3E93qvJft/Q4UV41soVsVgK8uGdHoL7e0/BeTx
        )
    
    def download_model(self):
        try:
            os.makedirs(Config.LOCAL_MODEL_DIR, exist_ok=True)
            
            # Download all files in the model directory
            objects = self.s3.list_objects_v2(
                Bucket=Config.S3_BUCKET,
                Prefix=Config.model_s3_path()
            )
            
            if 'Contents' not in objects:
                return False
                
            for obj in objects['Contents']:
                file_key = obj['Key']
                local_path = os.path.join(Config.LOCAL_MODEL_DIR, os.path.basename(file_key))
                self.s3.download_file(Config.S3_BUCKET, file_key, local_path)
            
            return True
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return False
    
    def upload_model(self, local_dir):
        try:
            for root, _, files in os.walk(local_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_path = os.path.join(
                        Config.model_s3_path(),
                        os.path.relpath(local_path, local_dir)
                    )
                    self.s3.upload_file(local_path, Config.S3_BUCKET, s3_path)
            return True
        except Exception as e:
            logger.error(f"Error uploading model: {e}")
            return False
