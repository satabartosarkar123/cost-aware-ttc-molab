import requests, time
class OllamaClient:
    def __init__(self, model="qwen2.5:3b", base_url="http://localhost:11434"):
        self.model = model; self.base_url = base_url
    def generate(self, prompt, temperature=0.7, max_tokens=512, stop=None):
        payload = {"model": self.model, "prompt": prompt, "stream": False,
                   "options": {"temperature": temperature,
                               "num_predict": max_tokens,
                               "stop": stop or []}}
        t0 = time.time()
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
        r.raise_for_status(); d = r.json(); lat = time.time() - t0
        return {"text": d.get("response",""), "latency_seconds": lat,
                "prompt_tokens": d.get("prompt_eval_count",0),
                "completion_tokens": d.get("eval_count",0),
                "total_tokens": d.get("prompt_eval_count",0)+d.get("eval_count",0)}
