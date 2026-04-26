
import os
os.environ["HOME"] = "/mnt/nas/shuvranshu"
os.environ["HF_HOME"] = "/mnt/nas/shuvranshu/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/nas/shuvranshu/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/nas/shuvranshu/huggingface_cache"
os.environ["XDG_CACHE_HOME"] = "/mnt/nas/shuvranshu/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/mnt/nas/shuvranshu/huggingface_cache"
os.makedirs("/mnt/nas/shuvranshu/huggingface_cache", exist_ok=True)

# ============================================================
# Dataset
# ============================================================
from datasets import load_dataset

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

# Check categories
print("Unique categories:", dataset["train"].unique("category"))

# Map to your system categories
CATEGORY_MAP = {
    "ACCOUNT"          : "account",
    "BILLING"          : "billing",
    "CANCELLATION_FEE" : "billing",
    "REFUND"           : "billing",
    "PAYMENT"          : "billing",
    "DELIVERY"         : "shipping",
    "ORDER"            : "order",
    "TECHNICAL_SUPPORT": "technical",
    "CONTACT"          : "other",
    "FEEDBACK"         : "complaint",
    "COMPLAINT"        : "complaint",
}

def map_category(row):
    raw = row["category"].upper().replace(" ", "_")
    for key in CATEGORY_MAP:
        if key in raw:
            row["mapped_category"] = CATEGORY_MAP[key]
            return row
    row["mapped_category"] = "other"
    return row

dataset = dataset.map(map_category)

# Format into prompt
SYSTEM_PROMPT = """You are a customer support query classifier.
Classify the user query into exactly one of these categories:
[billing, account, technical, shipping, order, complaint, other]
Respond with ONLY the category name, nothing else."""

def format_example(example):
    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['mapped_category']}<|im_end|>"
    )
    return {"text": text}

dataset = dataset.map(format_example)

# Train/val split
dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
print(f"Train: {len(dataset['train'])}  Val: {len(dataset['test'])}")

# ============================================================
# Model & Tokenizer
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "Qwen/Qwen2.5-3B-Instruct"
print("qwen downloaded")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

import torch
print(f"GPU name: {torch.cuda.get_device_name(0)}")


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,    # ~6GB, no quantization
    device_map={"": 0},           # GPU 0 only
    trust_remote_code=True
)

print(f"Model loaded on: {next(model.parameters()).device}")
# ============================================================
# LoRA
# ============================================================
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters()

model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.config.use_cache = False

# ============================================================
# Training
# ============================================================
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="/mnt/nas/shuvranshu/qwen3b_planner",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    fp16=True,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,    # correct for your version
    args=training_args,
    peft_config=lora_config,       # pass LoRA config here directly
)
trainer.train()

# ============================================================
# Save adapter + merge
# ============================================================
SAVE_DIR = "/mnt/nas/shuvranshu/finetune"

model.save_pretrained(f"{SAVE_DIR}/qwen_lora_adapter")
tokenizer.save_pretrained(f"{SAVE_DIR}/qwen_lora_adapter")
print("Adapter saved!")

# Merge on CPU
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True
)
merged = PeftModel.from_pretrained(base, f"{SAVE_DIR}/qwen_lora_adapter")
merged = merged.merge_and_unload()
merged.save_pretrained(f"{SAVE_DIR}/qwen_merged_model")
tokenizer.save_pretrained(f"{SAVE_DIR}/qwen_merged_model")
print("Merged model saved to", f"{SAVE_DIR}/qwen_merged_model")