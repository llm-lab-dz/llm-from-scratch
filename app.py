"""
Gradio web chat UI for a custom-trained GPT-style LLM.

Deployable on Render (or any host) - downloads the checkpoint and tokenizer
from your private Hugging Face repos at startup using an HF_TOKEN env var.

Before running:
  pip install -r requirements.txt
"""

import os
import torch
import torch.nn.functional as F
import gradio as gr
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

from model.model import GPT  # your GPT class lives in model/model.py

# ----------------------------------------------------------------------
# 1. CONFIG
# ----------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")  # set this in Render's Environment tab
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_REPO_ID = "llm-lab-dz/LLM_From_Scratch"
CKPT_FILENAME = "ckpt.pt"
TOKENIZER_REPO_ID = "llm-lab-dz/LLM_From_Scratch_Data"
TOKENIZER_FILENAME = "tokenizer.json"

DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.3

# ----------------------------------------------------------------------
# 2. DOWNLOAD MODEL FILES FROM YOUR PRIVATE HF REPOS
# ----------------------------------------------------------------------
print("Downloading checkpoint...")
ckpt_path = hf_hub_download(
    repo_id=CKPT_REPO_ID,
    filename=CKPT_FILENAME,
    token=HF_TOKEN,
)
print("Downloading tokenizer...")
tokenizer_path = hf_hub_download(
    repo_id=TOKENIZER_REPO_ID,
    filename=TOKENIZER_FILENAME,
    repo_type="dataset",
    token=HF_TOKEN,
)

# ----------------------------------------------------------------------
# 3. LOAD MODEL + TOKENIZER (mirrors your generate.py script)
# ----------------------------------------------------------------------
print("Loading checkpoint into model...")
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
config = ckpt["config"]

model = GPT(config).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = Tokenizer.from_file(tokenizer_path)

print(f"Model loaded on {DEVICE} with {model.num_params():,} parameters.")

# ----------------------------------------------------------------------
# 4. SAMPLING FUNCTION (ported directly from your generate.py)
# ----------------------------------------------------------------------
@torch.no_grad()
def sample(model, idx, max_new_tokens, block_size, temperature=0.8, top_k=50,
           top_p=0.9, repetition_penalty=1.3):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # repetition penalty: discourage tokens already generated
        for batch_index in range(idx.size(0)):
            for token_id in set(idx[batch_index].tolist()):
                logits[batch_index, token_id] /= repetition_penalty

        # top-k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # top-p (nucleus)
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_probs[mask] = 0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_idx.gather(-1, next_token_sorted)

        idx = torch.cat([idx, next_token], dim=1)
    return idx


# ----------------------------------------------------------------------
# 5. CHAT WRAPPER - Gradio calls this on every message
# ----------------------------------------------------------------------
def chat_response(message, history, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    ids = tokenizer.encode(message).ids
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    out = sample(
        model, idx,
        max_new_tokens=int(max_new_tokens),
        block_size=config.block_size,
        temperature=temperature,
        top_k=int(top_k) if top_k else None,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    full_text = tokenizer.decode(out[0].tolist())
    prompt_text = tokenizer.decode(ids)

    # Strip the echoed prompt so only the new generation is shown
    if full_text.startswith(prompt_text):
        full_text = full_text[len(prompt_text):]

    return full_text.strip()


# ----------------------------------------------------------------------
# 6. BUILD THE UI (with generation-parameter sliders)
# ----------------------------------------------------------------------
demo = gr.ChatInterface(
    fn=chat_response,
    title="My From-Scratch LLM",
    description="A simple chat UI running on my own custom-trained GPT model.",
    additional_inputs=[
        gr.Slider(1, 500, value=DEFAULT_MAX_NEW_TOKENS, step=1, label="Max new tokens"),
        gr.Slider(0.1, 2.0, value=DEFAULT_TEMPERATURE, step=0.05, label="Temperature"),
        gr.Slider(0, 200, value=DEFAULT_TOP_K, step=1, label="Top-k"),
        gr.Slider(0.05, 1.0, value=DEFAULT_TOP_P, step=0.05, label="Top-p"),
        gr.Slider(1.0, 2.0, value=DEFAULT_REPETITION_PENALTY, step=0.05, label="Repetition penalty"),
    ],
)

# ----------------------------------------------------------------------
# 7. LAUNCH
# ----------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )