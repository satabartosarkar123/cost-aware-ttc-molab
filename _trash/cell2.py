def execute_cell_29():
    # Day3-Fetch-Llama3.2-3B — Pull llama3.2:3b into Ollama

    print("=" * 60)
    print("  Day 3 — Fetching llama3.2:3b")
    print("=" * 60)

    subprocess.run("ollama pull llama3.2:3b", shell=True, check=True)
    time.sleep(3)

    r = requests.get("http://localhost:11434/api/tags", timeout=10)
    r.raise_for_status()
    models = [m["name"] for m in r.json().get("models", [])]
    assert any("llama3.2:3b" in m for m in models), f"llama3.2:3b not found! Available: {models}"
    print(f"llama3.2:3b confirmed. Available models: {models}")

execute_cell_29()