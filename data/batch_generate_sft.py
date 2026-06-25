"""批量生成SFT数据 -- 每轮API调用生成5个QA对"""
import json, os, time, requests, sys
sys.path.insert(0, os.path.dirname(__file__))
from generate_drone_data import QA_TOPICS, call_ollama

outfile = os.path.join(os.path.dirname(__file__), 'sft_drone_batch3.jsonl')
existing = 460
target = 2000
need = target - existing
start = time.time()

# 每批5个问题
batch_size = 5
batches = [QA_TOPICS[i:i+batch_size] for i in range(0, len(QA_TOPICS), batch_size)]

batch_prompt_tpl = '''你是无人机专家。请为以下{num}个问题分别提供专业详细的回答（每题200-400字），用JSON格式输出：
{questions}

输出格式：
[{{"q": "问题1", "a": "回答1"}}, {{"q": "问题2", "a": "回答2"}}, ...]
不要加任何额外的解释文字，直接输出JSON数组。'''

generated = 0
with open(outfile, 'w', encoding='utf-8') as f:
    while generated < need:
        for batch in batches:
            if generated >= need:
                break
            questions = '\n'.join([f'{i+1}. {q}' for i, q in enumerate(batch)])
            prompt = batch_prompt_tpl.format(num=len(batch), questions=questions)
            result = call_ollama(prompt, 'qwen3-coder:30b', max_tokens=2048)
            if result:
                try:
                    # 尝试解析JSON数组
                    json_str = result.strip()
                    if '```' in json_str:
                        json_str = json_str.split('```')[1]
                        if json_str.startswith('json'):
                            json_str = json_str[4:]
                    if '[' in json_str and ']' in json_str:
                        json_str = json_str[json_str.find('['):json_str.rfind(']')+1]
                        pairs = json.loads(json_str)
                        for pair in pairs:
                            if isinstance(pair, dict) and 'q' in pair and 'a' in pair:
                                conv = {'conversations': [
                                    {'role': 'user', 'content': pair['q']},
                                    {'role': 'assistant', 'content': pair['a']}
                                ]}
                                f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                                generated += 1
                except:
                    pass  # JSON parse fail, skip this batch
                f.flush()
                if generated % 25 == 0:
                    elapsed = max((time.time() - start) / 60, 0.1)
                    rate = generated / elapsed
                    eta = (need - generated) / max(rate, 0.01)
                    print(f'[{generated}/{need}] total={generated+existing}/2000, {rate:.1f} rec/min, ETA {eta:.0f}min', flush=True)
            time.sleep(0.3)

print(f'Done: {generated} new records -> {outfile}')
