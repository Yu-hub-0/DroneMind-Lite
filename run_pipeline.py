"""
全自动训练流水线：SFT生成 → 预训练 → SFT训练 → 推理测试
用法：python run_pipeline.py
"""
import os, sys, time, json, subprocess

PYTHON = r'C:\Program Files\Python312\python.exe'
BASE = os.path.dirname(os.path.abspath(__file__))
TRAINER = os.path.join(BASE, 'trainer')
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'out')
SFT_FILES = [
    'sft_drone_sample.jsonl',
    'sft_drone.jsonl',
    'sft_drone_batch2.jsonl',
    'sft_drone_batch3.jsonl',
]

def log(msg):
    print(f'[Pipeline {time.strftime("%H:%M:%S")}] {msg}', flush=True)

def run_step(name, cmd, cwd, timeout=None):
    log(f'START: {name}')
    log(f'  cmd: {cmd[:120]}...')
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    start = time.time()
    output_lines = []
    for line in proc.stdout:
        line = line.strip()
        # 过滤tensorflow噪声
        if 'tensorflow' in line.lower() or 'onednn' in line.lower():
            continue
        if line:
            output_lines.append(line)
            # 只打印训练关键行
            if any(kw in line for kw in ['loss:', 'Epoch:', 'Model Params', 'Done:', 'FAILED', 'error', 'Error']):
                log(f'  {line[:150]}')
    proc.wait(timeout=timeout)
    elapsed = (time.time() - start) / 60
    if proc.returncode == 0:
        log(f'DONE: {name} ({elapsed:.0f}min)')
    else:
        log(f'FAILED: {name} (exit={proc.returncode}, {elapsed:.0f}min)')
        log(f'  Last output: {output_lines[-3:] if output_lines else "none"}')
    return proc.returncode == 0

def sft_count():
    total = 0
    for fname in SFT_FILES:
        fp = os.path.join(DATA, fname)
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8') as f:
                total += sum(1 for _ in f)
    return total

def merge_sft():
    """合并所有SFT数据"""
    all_convos = []
    seen = set()
    for fname in SFT_FILES:
        fp = os.path.join(DATA, fname)
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    key = json.dumps(data.get('conversations',[]), ensure_ascii=False)
                    if key not in seen:
                        seen.add(key)
                        all_convos.append(data)
                except: pass
    merged_path = os.path.join(DATA, 'sft_merged.jsonl')
    with open(merged_path, 'w', encoding='utf-8') as f:
        for c in all_convos:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    log(f'Merged SFT: {len(all_convos)} records -> sft_merged.jsonl')
    return len(all_convos)


if __name__ == '__main__':
    log('=== DroneMind Pipeline Started ===')

    # ====== Phase 1: Wait for SFT generation ======
    log('Phase 1: Waiting for SFT data generation...')
    target = 500
    while sft_count() < target:
        current = sft_count()
        log(f'  SFT data: {current}/{target}...')
        time.sleep(300)  # check every 5 min

    # Kill any lingering ollama to free GPU
    log('SFT data ready! Freeing GPU from Ollama...')
    subprocess.run(['taskkill', '/f', '/im', 'ollama_llama_server.exe'], capture_output=True)
    time.sleep(10)

    # ====== Phase 2: Pretraining ======
    log('Phase 2: Pretraining on 1.27M records')
    os.makedirs(OUT, exist_ok=True)
    ok = run_step('Pretrain', [
        PYTHON, '-X', 'utf8', 'train_pretrain.py',
        '--data_path', '../data/pretrain_t2t_mini.jsonl',
        '--epochs', '1', '--batch_size', '32', '--accumulation_steps', '8',
        '--hidden_size', '768', '--num_hidden_layers', '8',
        '--max_seq_len', '340', '--log_interval', '500', '--save_interval', '10000',
        '--save_dir', '../out', '--from_weight', 'none', '--device', 'cuda:0',
    ], cwd=TRAINER, timeout=43200)  # 12h timeout

    if not ok:
        log('Pretrain failed! Aborting.')
        sys.exit(1)

    # ====== Phase 3: Merge SFT data ======
    log('Phase 3: Merging SFT data')
    n_sft = merge_sft()

    # ====== Phase 4: SFT training ======
    log('Phase 4: SFT training')
    ok = run_step('SFT', [
        PYTHON, '-X', 'utf8', 'train_full_sft.py',
        '--data_path', '../data/sft_merged.jsonl',
        '--epochs', '5', '--batch_size', '4', '--accumulation_steps', '1',
        '--hidden_size', '768', '--num_hidden_layers', '8',
        '--max_seq_len', '768', '--log_interval', '20', '--save_interval', '1000',
        '--save_dir', '../out', '--from_weight', 'pretrain', '--device', 'cuda:0',
    ], cwd=TRAINER, timeout=3600)

    # ====== Phase 5: Test ======
    log('Phase 5: Quick inference test')
    test_script = '''
import os,sys,torch,warnings,time,json
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM']='false'
import datasets
sys.path.insert(0,'.')
from model.model_dronemind import DroneMindConfig,DroneMindForCausalLM
from transformers import AutoTokenizer
device='cuda:0'
config=DroneMindConfig(hidden_size=768,num_hidden_layers=8)
model=DroneMindForCausalLM(config).half().to(device)
model.load_state_dict(torch.load('out/full_sft_768.pth',map_location=device),strict=False)
tokenizer=AutoTokenizer.from_pretrained('./model')
model.eval()
for q in [
    '无人机起飞前需要做哪些检查？',
    '无人机电池怎么保养？',
    '无人机在强风环境飞行需要注意什么？',
]:
    conv=[{'role':'user','content':q}]
    prompt=tokenizer.apply_chat_template(conv,tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(prompt,return_tensors='pt').to(device)
    st=time.time()
    with torch.no_grad():
        gen=model.generate(input_ids=inputs['input_ids'],max_new_tokens=200,do_sample=True,temperature=0.85,top_p=0.9,eos_token_id=tokenizer.eos_token_id)
    elapsed=time.time()-st
    tokens=len(gen[0])-len(inputs['input_ids'][0])
    resp=tokenizer.decode(gen[0][len(inputs['input_ids'][0]):],skip_special_tokens=True)
    print(f'Q: {q}')
    print(f'A: {resp[:300]}')
    print(f'  [{tokens} tok, {tokens/elapsed:.0f} tok/s]')
    print('-'*50)
'''
    with open(os.path.join(BASE, '_test_inference.py'), 'w', encoding='utf-8') as f:
        f.write(test_script)
    subprocess.run([PYTHON, '-X', 'utf8', os.path.join(BASE, '_test_inference.py')], cwd=BASE)

    log('=== Pipeline Complete! ===')
    log(f'Model: {os.path.join(OUT, "full_sft_768.pth")}')
