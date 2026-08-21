import json

with open('molab_run_fixed.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

diag_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'source': [
        '# [DIAGNOSTICS] Restart Ollama and Verify GPU Acceleration\n',
        'import os, subprocess, time, requests, json\n',
        'print("=== SYSTEM DIAGNOSTICS ===")\n',
        '\n',
        '# 1. Check GPU\n',
        'try:\n',
        '    nvidia_smi = subprocess.check_output("nvidia-smi", shell=True, text=True)\n',
        '    print("GPU DETECTED:")\n',
        '    for line in nvidia_smi.split(\'\\n\')[:10]:\n',
        '        print("  " + line)\n',
        'except Exception as e:\n',
        '    print("NO GPU DETECTED or nvidia-smi failed!", e)\n',
        '\n',
        '# 2. Restart Ollama\n',
        'print("\\n=== RESTARTING OLLAMA ===")\n',
        'os.system("pkill ollama")\n',
        'time.sleep(2)\n',
        'subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n',
        'time.sleep(5)\n',
        '\n',
        '# 3. Test Latency\n',
        'print("\\n=== TESTING LATENCY (qwen2.5:3b) ===")\n',
        'try:\n',
        '    payload = {"model": "qwen2.5:3b", "prompt": "What is 2+2? Answer in one word.", "stream": False}\n',
        '    start = time.time()\n',
        '    res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)\n',
        '    latency = time.time() - start\n',
        '    data = res.json()\n',
        '    print(f"Response: {data.get(\'response\')} {data.get(\'text\', \'\')}")\n',
        '    print(f"Latency: {latency:.2f} seconds")\n',
        '    if latency > 5.0:\n',
        '        print("\\n[!!!] WARNING: Latency is incredibly high. Ollama is running on CPU!")\n',
        '        print("[!!!] Stop the sweep. Reboot the Molab container instance.")\n',
        '    else:\n',
        '        print("\\n[OK] Latency is good. GPU is active! You may proceed with the sweep.")\n',
        'except Exception as e:\n',
        '    print("Ollama test failed:", e)\n'
    ]
}

nb['cells'].insert(0, diag_cell)

with open('molab_run_fixed.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Diagnostics cell added at the top.')
