def run_bon_k5(client, task, question):
    prompt = get_prompt("greedy_cot", task, question)
    rationales = []; lat = 0; tok = 0
    parser = get_parser(task)
    for _ in range(5):
        r = client.generate(prompt, temperature=0.7)
        lat += r["latency_seconds"]; tok += r["total_tokens"]
        p = parser(r["text"])
        rationales.append({"text": r["text"], "answer": p["final_answer"]})
    best_ans = None; best_score = -1; best_rat = ""; judge_texts = []
    for rat in rationales:
        jp = get_prompt("best_of_n", task="", question=question, candidate=rat["text"])
        jr = client.generate(jp, temperature=0.0)
        lat += jr["latency_seconds"]; tok += jr["total_tokens"]; judge_texts.append(jr["text"])
        score = 0.5
        sm = re.search(r"confidence:\s*(\d+)", jr["text"].lower())
        if sm: score = float(sm.group(1)) / 100.0
        elif "yes" in jr["text"].lower(): score = 1.0
        elif "no" in jr["text"].lower(): score = 0.0
        if score > best_score: best_score = score; best_ans = rat["answer"]; best_rat = rat["text"]
    return {"selected_answer": best_ans,
            "raw_response": f"Selected:\n{best_rat}\n\nJudge:\n" + "\n---\n".join(judge_texts),
            "latency_seconds_total": lat, "total_tokens": tok, "model_calls": 10,
            "parse_success": best_ans is not None, "parse_method": "best_of_n_self_eval",
            "raw_paths": [r["text"] for r in rationales]}
