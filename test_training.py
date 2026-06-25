"""快速验证 -- 添加 Windows pyarrow/torch 兼容处理"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    # Windows pyarrow/torch DLL conflict workaround (issue #771)
    import datasets  # noqa: F401
except ModuleNotFoundError as exc:
    print("Missing dependency: datasets")
    print("Install project dependencies first, for example:")
    print("  python -m pip install -r requirements.txt")
    raise SystemExit(1) from exc

import torch
sys.path.insert(0, '.')
from model.model_dronemind import DroneMindConfig, DroneMindForCausalLM
from transformers import AutoTokenizer

device = "cpu"
print(f"Device: {device} | PyTorch: {torch.__version__}")

# ============ 1. 初始化模型 ============
print("\n[1] Init model (tiny 128x2)...")
config = DroneMindConfig(hidden_size=128, num_hidden_layers=2, use_moe=False)
model = DroneMindForCausalLM(config).to(device)
tokenizer = AutoTokenizer.from_pretrained('./model')
print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"  Vocab: {len(tokenizer)}")

# ============ 2. 数据集测试 ============
print("\n[2] Test PretrainDataset...")
from dataset.lm_dataset import PretrainDataset, SFTDataset
ds = PretrainDataset('./data/pretrain_drone_sample.jsonl', tokenizer, max_length=64)
print(f"  Size: {len(ds)} samples")
x, y = ds[0]
print(f"  Shape: {x.shape}, labels: {y.shape}")
print("  OK!")

print("\n[3] Test SFTDataset...")
ds_sft = SFTDataset('./data/sft_drone_sample.jsonl', tokenizer, max_length=256)
print(f"  Size: {len(ds_sft)} samples")
x, y = ds_sft[0]
print(f"  Shape: {x.shape}, labels: {y.shape}")
print("  OK!")

# ============ 3. 训练几步 ============
print("\n[4] Training 3 steps...")
from torch import optim
from torch.utils.data import DataLoader

optimizer = optim.AdamW(model.parameters(), lr=5e-4)
loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

model.train()
for step, (input_ids, labels) in enumerate(loader):
    res = model(input_ids, labels=labels)
    loss = res.loss + res.aux_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"  Step {step+1}: loss={loss.item():.4f}")
    if step >= 2:
        break

# ============ 4. 生成 ============
print("\n[5] Generation test...")
model.eval()
test_prompt = tokenizer.bos_token + "无人机飞行前需要"
inputs = tokenizer(test_prompt, return_tensors="pt")
with torch.no_grad():
    gen = model.generate(input_ids=inputs["input_ids"], max_new_tokens=30,
                         do_sample=True, temperature=0.85, eos_token_id=tokenizer.eos_token_id)
out_text = tokenizer.decode(gen[0], skip_special_tokens=True)
print(f"  Prompt:  {test_prompt}")
print(f"  Output: {out_text.encode('utf-8', errors='replace')[:100]}")

print("\n" + "="*50)
print("All checks passed! Pipeline is ready for training.")
print("="*50)
