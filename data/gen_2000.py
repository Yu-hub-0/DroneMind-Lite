"""生成2000条SFT数据 -- 循环调用generate_drone_data"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from generate_drone_data import QA_TOPICS, call_ollama
import time

OUTFILE = os.path.join(os.path.dirname(__file__), 'sft_drone_batch3.jsonl')
TARGET = 2000
EXISTING = 460

need = TARGET - EXISTING
start = time.time()

batch_prompt = '你是一位资深无人机专家和飞行教练。请为以下问题提供专业、详细、实用的回答（300-600字）：\n「{topic}」\n直接回答，不要加"作为AI"、"根据我的知识"这类前缀。包含具体数据和建议。\n回答：'

generated = 0
topic_idx = 0

# Count existing in batch3
if os.path.exists(OUTFILE):
    with open(OUTFILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                generated += 1
    print(f'Resuming from {generated} existing records in batch3')

with open(OUTFILE, 'a' if generated > 0 else 'w', encoding='utf-8') as f:
    while generated < need:
        topic = QA_TOPICS[topic_idx % len(QA_TOPICS)]
        topic_idx += 1
        try:
            result = call_ollama(batch_prompt.format(topic=topic), 'qwen2.5:7b', max_tokens=1500)
            if result and len(result.strip()) > 50:
                conv = {'conversations': [
                    {'role': 'user', 'content': topic},
                    {'role': 'assistant', 'content': result.strip()}
                ]}
                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                f.flush()
                generated += 1
                elapsed = max((time.time() - start) / 60, 0.01)
                rate = generated / elapsed
                eta = (need - generated) / max(rate, 0.01)
                if generated % 10 == 0:
                    print(f'[{generated}/{need}] {generated+EXISTING}/2000, {rate:.1f} rec/min, ETA {eta:.0f}min', flush=True)
            else:
                print(f'SKIP: empty/bad response for topic {topic_idx}', flush=True)
        except Exception as e:
            print(f'ERROR: {e}', flush=True)
            time.sleep(5)
        time.sleep(0.1)

print(f'DONE: {generated} records -> {OUTFILE}', flush=True)
