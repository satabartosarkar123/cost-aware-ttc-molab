import time
from core.ollama_client import OllamaClient

try:
    print('Checking 3b...')
    client3b = OllamaClient()
    client3b.default_model = 'qwen2.5:3b'
    t0 = time.time()
    res3b = client3b.generate('What is 2+2?', max_tokens=100)
    t3b = time.time() - t0
    tokens3b = res3b.get("completion_tokens", 1)
    print(f'3b took {t3b:.2f}s, speed: {tokens3b/t3b:.2f} t/s')

    print('Checking 7b...')
    client7b = OllamaClient()
    client7b.default_model = 'qwen2.5:7b-instruct-q4_K_M'
    t0 = time.time()
    res7b = client7b.generate('What is 2+2?', max_tokens=100)
    t7b = time.time() - t0
    tokens7b = res7b.get("completion_tokens", 1)
    print(f'7b took {t7b:.2f}s, speed: {tokens7b/t7b:.2f} t/s')

    speed_ratio = (tokens3b/t3b) / (tokens7b/t7b)
    print(f"7b is {speed_ratio:.2f}x slower per token.")
    
    if speed_ratio > 3.0:
        print("SKIP_7B")
    else:
        print("KEEP_7B")
except Exception as e:
    print("Failed or OOM:", e)
    print("SKIP_7B")
