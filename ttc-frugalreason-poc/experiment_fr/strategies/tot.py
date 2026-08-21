import re
import time
from collections import Counter
from core.parsers import get_parser

def zero_shot_tot_k3(client, task, question, input_metadata=None, k=3, temperature=0.7):
    """
    Lightweight zero-shot ToT-BFS from ToT paper Appendix B.1.
    k = number of strategies and solutions (default 3)
    """
    parser = get_parser(task)
    t0 = time.time()
    tokens = 0
    calls = 0
    
    def gen(prompt, temp):
        nonlocal tokens, calls
        r = client.generate(prompt, temperature=temp)
        tokens += r.get("total_tokens", 0)
        calls += 1
        return r.get("text", "")
    
    # Step 1: Propose k strategies
    strategy_prompt = f"""Question: {question}

Write a short step-by-step PLAN to solve this problem.
Only the plan, no solution."""
    
    plans = []
    for _ in range(k):
        plans.append(gen(strategy_prompt, temperature))
    
    # Step 2: Vote for the best strategy
    choices = "\n".join(f"{i+1}. {p.strip()[:300]}" for i, p in enumerate(plans))
    vote_prompt = f"""Question: {question}

Candidate plans:
{choices}

Which plan is most promising? 
End with 'The best choice is X' where X is the plan number (1-{k})."""
    
    vote_response = gen(vote_prompt, 0.0)
    
    # Parse the vote
    match = re.search(r"best choice is\s*\(?\s*([1-9])", vote_response, re.I)
    if match:
        best_idx = min(max(int(match.group(1)) - 1, 0), k - 1)
    else:
        best_idx = 0  # Default to first plan if parsing fails
    
    winning_plan = plans[best_idx]
    
    # Step 3: Generate k solutions based on the winning plan
    solution_prompt = f"""Question: {question}

Follow this plan:
{winning_plan}

Now solve it step by step."""
    
    raw_responses = []
    parsed_answers = []
    
    for _ in range(k):
        response = gen(solution_prompt, temperature)
        raw_responses.append(response)
        parsed = parser(response)
        if parsed.get("final_answer") is not None:
            parsed_answers.append(parsed.get("final_answer"))
    
    # Step 4: Majority vote on parsed answers
    valid_answers = [a for a in parsed_answers if a is not None]
    
    if valid_answers:
        counter = Counter(valid_answers)
        selected = counter.most_common(1)[0][0]
    else:
        selected = None
    
    return {
        "raw_response": "\n---\n".join(raw_responses),
        "selected_answer": selected,
        "latency_seconds_total": time.time() - t0,
        "total_tokens": tokens,
        "model_calls": calls,
        "parse_success": selected is not None,
        "parse_method": "tot_majority_vote",
        "early_exit": False,
        "raw_paths": raw_responses,
        "tot_plans": plans,
        "tot_winning_plan": winning_plan,
        "tot_vote_response": vote_response
    }
