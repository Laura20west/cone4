from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import json
from datetime import datetime
from pathlib import Path
import uuid
import random
from collections import defaultdict, deque, Counter
import nltk
from nltk.corpus import wordnet as wn
from typing import Dict, List, Optional, Tuple
import asyncio

# Initialize NLP
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_md"])
    nlp = spacy.load("en_core_web_md")

nltk.download('wordnet')

app = FastAPI()

# Configuration
AUTHORIZED_OPERATORS = {
        "cone221", "cone412", "admin@company.com"
    }
DATASET_PATH = Path("conversation_dataset.jsonl")
UNCERTAIN_PATH = Path("uncertain_responses.jsonl")
REPLY_POOLS_PATH = Path("reply_pools_augmented.json")

# Precompute for faster access
CATEGORY_NAMES = set()
TRIGGER_CACHE = defaultdict(list)

# Security dependency
async def verify_operator(request: Request):
    operator_email = request.headers.get("X-Operator-Email")
    if operator_email in AUTHORIZED_OPERATORS:
        return operator_email
    raise HTTPException(status_code=403, detail="Unauthorized operator")

# Load or initialize reply pools
if REPLY_POOLS_PATH.exists():
    with open(REPLY_POOLS_PATH, "r") as f:
        REPLY_POOLS = json.load(f)
    for category, data in REPLY_POOLS.items():
        data.setdefault("triggers", [])
        data.setdefault("responses", [])
        data.setdefault("questions", [])
        CATEGORY_NAMES.add(category)
        for trigger in data["triggers"]:
            TRIGGER_CACHE[category].append((trigger, nlp(trigger.lower())))
else:
    REPLY_POOLS = {
        "general": {
            "triggers": [],
            "responses": ["Honey, let's talk about something more exciting..."],
            "questions": ["What really gets you going?"]
        }
    }
    CATEGORY_NAMES.add("general")

class ResponseSelector:
    def __init__(self):
        self.used_combinations = defaultdict(set)
        self.available_combinations = defaultdict(deque)
        self._initialize_combinations()
    
    def _initialize_combinations(self):
        for category, data in REPLY_POOLS.items():
            responses = data["responses"]
            questions = data["questions"]
            combinations = [(r_idx, q_idx) 
                          for r_idx in range(len(responses))
                          for q_idx in range(len(questions))]
            random.shuffle(combinations)
            self.available_combinations[category] = deque(combinations)
    
    def get_unique_pair(self, category: str) -> Tuple[Optional[int], Optional[int]]:
        if category not in self.available_combinations:
            return (None, None)
            
        # Get next available combination
        while self.available_combinations[category]:
            r_idx, q_idx = self.available_combinations[category].popleft()
            if (r_idx, q_idx) not in self.used_combinations[category]:
                self.used_combinations[category].add((r_idx, q_idx))
                return (r_idx, q_idx)
        
        # If we've used all combinations, reset and try again
        self._reset_category(category)
        if self.available_combinations[category]:
            r_idx, q_idx = self.available_combinations[category].popleft()
            self.used_combinations[category].add((r_idx, q_idx))
            return (r_idx, q_idx)
        return (None, None)
    
    def _reset_category(self, category: str):
        self.used_combinations[category].clear()
        self._initialize_combinations()

response_selector = ResponseSelector()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

class UserMessage(BaseModel):
    message: str

class SallyResponse(BaseModel):
    matched_word: str
    matched_category: str
    confidence: float
    replies: List[str]

async def log_to_dataset(user_input: str, response_data: dict, operator: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "matched_category": response_data["matched_category"],
        "response": response_data["replies"][0] if response_data["replies"] else None,
        "question": response_data["replies"][1] if len(response_data["replies"]) > 1 else None,
        "operator": operator,
        "confidence": response_data["confidence"],
        "embedding": nlp(user_input).vector.tolist()
    }
    
    with open(DATASET_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

async def store_uncertain(user_input: str):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "user_input": user_input,
        "reviewed": False
    }
    
    with open(UNCERTAIN_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")

def augment_dataset():
    if not DATASET_PATH.exists():
        return
    
    with open(DATASET_PATH, "r") as f:
        entries = [json.loads(line) for line in f]
    
    category_counts = Counter(entry["matched_category"] for entry in entries)
    avg_count = sum(category_counts.values()) / len(category_counts) if category_counts else 0
    
    for category in [k for k, v in category_counts.items() if v < avg_count * 0.5]:
        if category not in REPLY_POOLS:
            continue
            
        base_triggers = REPLY_POOLS[category]["triggers"]
        new_triggers = set()
        
        for trigger in base_triggers:
            doc = nlp(trigger)
            new_triggers.add(" ".join([token.lemma_ for token in doc]))
            
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    syns = [syn.lemmas()[0].name() for syn in wn.synsets(token.text)]
                    if syns:
                        new_triggers.add(trigger.replace(token.text, syns[0]))
        
        REPLY_POOLS[category]["triggers"] = list(set(REPLY_POOLS[category]["triggers"]) | new_triggers)
        TRIGGER_CACHE[category].extend(
            (trigger, nlp(trigger.lower()))
            for trigger in new_triggers
            if trigger not in {t[0] for t in TRIGGER_CACHE[category]}
        )
    
    with open(REPLY_POOLS_PATH, "w") as f:
        json.dump(REPLY_POOLS, f, indent=2)
    
    response_selector._initialize_combinations()

def get_best_match(doc) -> Tuple[str, str, float]:
    best_match = ("general", None, 0.0)
    
    for category in CATEGORY_NAMES:
        for trigger, trigger_doc in TRIGGER_CACHE[category]:
            if trigger.lower() in doc.text:
                return (category, trigger, 1.0)
    
    for category in CATEGORY_NAMES:
        for trigger, trigger_doc in TRIGGER_CACHE[category]:
            similarity = doc.similarity(trigger_doc)
            if similarity > best_match[2]:
                best_match = (category, trigger, similarity)
                if similarity > 0.9:
                    return best_match
    
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            for category in CATEGORY_NAMES:
                for trigger, _ in TRIGGER_CACHE[category]:
                    if token.lemma_ in trigger.lower():
                        current_sim = 0.7 + (0.3 * (token.pos_ == "NOUN"))
                        if current_sim > best_match[2]:
                            best_match = (category, token.text, current_sim)
    
    return best_match

@app.post("/1E59I6F1O5R1C8O3N87E5145ID", response_model=SallyResponse)
async def analyze_message(
    request: Request,
    user_input: UserMessage,
    operator: str = Depends(verify_operator)
):
    message = user_input.message.strip()
    doc = nlp(message.lower())
    
    category, matched_word, confidence = get_best_match(doc)
    
    response = {
        "matched_word": matched_word or "general",
        "matched_category": category,
        "confidence": round(confidence, 2),
        "replies": []
    }
    
    if REPLY_POOLS[category]["responses"] and REPLY_POOLS[category]["questions"]:
        r_idx, q_idx = response_selector.get_unique_pair(category)
        if r_idx is not None and q_idx is not None:
            response["replies"].append(REPLY_POOLS[category]["responses"][r_idx])
            response["replies"].append(REPLY_POOLS[category]["questions"][q_idx])
    
    if not response["replies"]:
        response["replies"] = [
            "Honey, let's take this somewhere more private...",
            "What's your deepest, darkest fantasy?"
        ]
    
    asyncio.create_task(log_to_dataset(message, response, operator))
    
    if confidence < 0.6:
        asyncio.create_task(store_uncertain(message))
        if len(response["replies"]) > 1:
            response["replies"][1] += " Could you rephrase that, baby?"
    
    return response

@app.get("/dataset/analytics")
async def get_analytics(request: Request, operator: str = Depends(verify_operator)):
    analytics = {
        "total_entries": 0,
        "common_categories": {},
        "confidence_stats": {}
    }
    
    if DATASET_PATH.exists():
        try:
            with open(DATASET_PATH, "r") as f:
                entries = [json.loads(line) for line in f]
            
            analytics["total_entries"] = len(entries)
            analytics["common_categories"] = Counter(entry["matched_category"] for entry in entries)
            
            if entries:
                confidences = [entry["confidence"] for entry in entries]
                analytics["confidence_stats"] = {
                    "average": round(sum(confidences)/len(confidences), 2),
                    "min": round(min(confidences), 2),
                    "max": round(max(confidences), 2)
                }
        except Exception as e:
            print(f"Error reading analytics: {e}")
    
    return analytics

@app.post("/augment")
async def trigger_augmentation(request: Request, operator: str = Depends(verify_operator)):
    augment_dataset()
    return {"status": "Dataset augmented", "new_pools": REPLY_POOLS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
