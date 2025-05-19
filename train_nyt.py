import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, AdamW, get_cosine_schedule_with_warmup
import json
import re
from tqdm import tqdm

# Special tokens and configs (from your input)
SPECIAL_TOKENS = {
    "user_token": "<|UserToken|>",
    "assistant_focus": "<|AssistantFocus|>",  # New token
    "headline": "<|NYT_Headline|>",
    "byline": "<|NYT_Byline|>",
    "body": "<|NYT_Body|>"
}

transformer_config = {
    "vocab_size": 50257 + len(SPECIAL_TOKENS),
    "max_position_embeddings": 2048,
    "n_layers": 24,
    "n_heads": 16,
    "d_model": 1024,
    "d_ff": 4096,
    "activation_function": "threshold_relu",
    "attention": {
        "type": "sparse_threshold",
        "threshold_strategy": "adaptive",
        "threshold_init": 0.1,
        "value_sparsity": True,
        "logit_sparsity": True
    },
    "ffn_threshold_strategy": "adaptive",
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "layer_norm_eps": 1e-5,
    "initializer_range": 0.02,
}

training_config = {
    "batch_size": 128,
    "micro_batch_size": 8,
    "learning_rate": 3e-4,
    "lr_scheduler": "cosine",
    "warmup_steps": 2000,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "adam_eps": 1e-8,
    "gradient_clipping": 1.0,
    "num_epochs": 30,
    "checkpoint_every": 500,
    "eval_every": 1000,
    "log_every": 100,
    "fp16": True,
    "device": "cuda",
    "use_flash_attention": True
}

# Custom components from your input
class AdaptiveThresholdReLU(nn.Module):
    def __init__(self, init_threshold=0.1, adaptive=True, anneal_rate=0.01):
        super().__init__()
        self.adaptive = adaptive
        self.threshold = nn.Parameter(torch.tensor(init_threshold), requires_grad=adaptive)
        self.anneal_rate = anneal_rate  # Add annealing capability

    def forward(self, x):
        if self.training:  # Only anneal during training
            self.threshold.data.clamp_(min=0.01, max=0.99)
            self.threshold.data += self.anneal_rate * torch.randn_like(self.threshold.data)
        return torch.where(x > self.threshold, x, torch.zeros_like(x))

def sparse_threshold_attention(q, k, v, threshold):
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
    mask = attn_scores >= threshold
    sparse_scores = attn_scores.masked_fill(~mask, float('-inf'))
    attn_probs = torch.nn.functional.softmax(sparse_scores, dim=-1)
    return torch.matmul(attn_probs, v)

# Dataset Preparation
class NYTDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path) as f:
            self.articles = [self._format_article(a) for a in json.load(f)]
        
    def _format_article(self, article):
        return f"{SPECIAL_TOKENS['user_token']} {article['User token']} "\
               f"{SPECIAL_TOKENS['assistant_focus']} "\  # Added focus token
               f"{SPECIAL_TOKENS['headline']} {article['NYT-style article']['headline']} "\
               f"{SPECIAL_TOKENS['byline']} {article['NYT-style article']['byline']} "\
               f"{SPECIAL_TOKENS['body']} {article['NYT-style article']['body']}"
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        text = self.articles[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze()
        }

# Model Architecture
class SparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.head_dim = self.d_model // self.n_heads
        self.threshold = nn.Parameter(torch.tensor(config["attention"]["threshold_init"]))
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_output = sparse_threshold_attention(q, k, v, self.threshold)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = SparseAttention(config)
        self.ffn = nn.Sequential(
            nn.Linear(config["d_model"], config["d_ff"]),
            AdaptiveThresholdReLU(config["ffn_threshold_strategy"] == "adaptive"),
            nn.Linear(config["d_ff"], config["d_model"])
        )
        self.norm1 = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])
        self.norm2 = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class NYTTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config["vocab_size"], config["d_model"])
        self.pos_emb = nn.Embedding(config["max_position_embeddings"], config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layers"])])
        self.ln_f = nn.LayerNorm(config["d_model"], eps=config["layer_norm_eps"])
        self.head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        return self.head(x)

# Training Setup
def train():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.values())})
    
    # Prepare dataset
    dataset = NYTDataset(tokenizer, "nytimes_articles.json")
    train_loader = DataLoader(dataset, batch_size=training_config["micro_batch_size"], shuffle=True)
    
    # Initialize model
    model = NYTTransformer(transformer_config).to(training_config["device"])
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=training_config["learning_rate"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_config["warmup_steps"],
        num_training_steps=len(train_loader)*training_config["num_epochs"]
    )
    
    # Training loop
    model.train()
    accum_steps = training_config["batch_size"] // training_config["micro_batch_size"]
    
    for epoch in range(training_config["num_epochs"]):
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress):
            inputs = batch["input_ids"].to(training_config["device"])
            targets = inputs.clone()
            
            outputs = model(inputs)
            loss = custom_loss(outputs, targets, tokenizer)
            
            loss.backward()
            
            if (i + 1) % accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), training_config["gradient_clipping"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            if (i + 1) % training_config["log_every"] == 0:
                progress.set_postfix({"loss": loss.item()})
                
            if (i + 1) % training_config["checkpoint_every"] == 0:
                torch.save(model.state_dict(), f"checkpoint_{epoch+1}_{i+1}.pt")

def custom_loss(logits, targets, tokenizer):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_targets = targets[..., 1:].contiguous()
    
    # Create segment weights
    special_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in SPECIAL_TOKENS.values()]
    weights = torch.ones_like(shift_targets).float()
    for tok_id in special_token_ids:
        weights[shift_targets == tok_id] = 2.0  # Weight special tokens higher
    
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1))
    return (loss.view(weights.size()) * weights).mean()

if __name__ == "__main__":
    # First run the data preparation
    # main() from your scraping code should be executed first
    # Then train the model
    train()