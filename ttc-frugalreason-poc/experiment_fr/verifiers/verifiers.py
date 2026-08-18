import re
import ast
import operator
from fractions import Fraction
from core.prompt_manager import get_prompt

# --- Game24 Verifier ---
def _eval_ast(node):
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg
    }
    
    if isinstance(node, ast.Num):
        return Fraction(node.n)
    elif isinstance(node, ast.BinOp):
        return operators[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    elif isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](_eval_ast(node.operand))
    else:
        raise TypeError("Unsupported operation")

def verify_game24(equation_text: str, input_numbers_str: str) -> bool:
    """
    Evaluates if equation_text strictly uses exactly the inputs and equals 24.
    Returns True if sound.
    """
    if not equation_text: return False
    
    try:
        # Extract inputs
        inputs = sorted([int(x) for x in re.findall(r'\d+', input_numbers_str)])
        
        # Extract numbers from equation
        eq_clean = re.sub(r'[^0-9\+\-\*\/\(\)\s]', '', equation_text)
        eq_nums = sorted([int(x) for x in re.findall(r'\d+', eq_clean)])
        
        if inputs != eq_nums:
            return False
            
        tree = ast.parse(eq_clean, mode='eval')
        result = _eval_ast(tree.body)
        
        return result == 24
    except Exception:
        return False

# --- GSM8K Step Verifier ---
def verify_gsm8k_steps(rationale_text: str, candidate_answer: str) -> dict:
    """
    Regex-extracts 'X op Y = Z'. Recomputes exactly.
    Returns dict with pass status.
    """
    result = {
        "all_steps_pass": False,
        "n_steps_parsed": 0,
        "n_steps_pass": 0,
        "final_matches": False
    }
    
    if not rationale_text: return result
    
    # Very crude step extractor for X [+-*/] Y = Z
    # We look for equations like: 20 + 30 = 50
    steps = re.findall(r'(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)', rationale_text)
    
    if not steps:
        return result
        
    result["n_steps_parsed"] = len(steps)
    passed = 0
    for s in steps:
        x, op, y, z = float(s[0]), s[1], float(s[2]), float(s[3])
        computed = 0
        try:
            if op == '+': computed = x + y
            elif op == '-': computed = x - y
            elif op == '*': computed = x * y
            elif op == '/': computed = x / y if y != 0 else float('inf')
            
            if abs(computed - z) < 1e-5:
                passed += 1
        except Exception:
            pass
            
    result["n_steps_pass"] = passed
    result["all_steps_pass"] = (passed == len(steps))
    
    # Check if final candidate answer is in the last step's Z
    if candidate_answer and steps:
        last_z = steps[-1][3]
        if candidate_answer in last_z or str(float(candidate_answer)) == str(float(last_z)):
            result["final_matches"] = True
            
    return result

# --- LLM Judge ---
def llm_judge(client, question: str, solution: str) -> float:
    """
    Uses best-of-n evaluation prompt to act as proxy RM.
    Returns score 0.0 to 1.0.
    """
    prompt = get_prompt("best_of_n", task="", question=question, candidate=solution)
    resp = client.generate(prompt, max_tokens=256, temperature=0.0)
    
    # Parse confidence/score
    text = resp.get("text", "").lower()
    
    # Look for "confidence: 100" or similar
    score_match = re.search(r'confidence:\s*(\d+)', text)
    if score_match:
        return float(score_match.group(1)) / 100.0
        
    # Look for raw number
    raw_nums = re.findall(r'\b(100|[1-9]?[0-9])\b', text)
    if raw_nums:
        return float(raw_nums[-1]) / 100.0
        
    # fallback
    if "yes" in text or "correct" in text:
        return 1.0
        
    return 0.0
