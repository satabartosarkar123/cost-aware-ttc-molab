import re

def parse_gsm8k(response: str) -> dict:
    """Parses GSM8K output. Strict match requires '#### <num>'. Lenient extracts last number."""
    result = {"strict_answer": None, "lenient_answer": None, "final_answer": None, "parse_method": "failed", "parse_success": False}
    if not response: return result
    
    # Try strict match first
    strict_match = re.search(r"####\s*(?:=\s*)?(-?\$?\d+(?:,\d+)*(?:\.\d+)?)", response)
    if strict_match:
        val = strict_match.group(1).replace("$", "").replace(",", "")
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result
        
    # Try common prefixes
    match = re.search(r"(?i)answer\s*(?:is|:)?\s*(-?\$?\d+(?:,\d+)*(?:\.\d+)?)", response)
    if match:
        val = match.group(1).replace("$", "").replace(",", "")
        result["lenient_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "lenient"
        result["parse_success"] = True
        return result

    # Try any numbers
    nums = re.findall(r"-?\$?\d+(?:,\d+)*(?:\.\d+)?", response)
    if nums:
        val = nums[-1].replace("$", "").replace(",", "")
        result["lenient_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "lenient"
        result["parse_success"] = True
        
    return result

def parse_strategyqa(response: str) -> dict:
    """Parses StrategyQA (yes/no). Strict looks for final 'yes' or 'no'."""
    result = {"strict_answer": None, "lenient_answer": None, "final_answer": None, "parse_method": "failed", "parse_success": False}
    if not response: return result
    
    text = response.lower()
    if text.endswith("yes.") or text.endswith("yes") or "#### yes" in text:
        result["strict_answer"] = "yes"
        result["final_answer"] = "yes"
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result
    elif text.endswith("no.") or text.endswith("no") or "#### no" in text:
        result["strict_answer"] = "no"
        result["final_answer"] = "no"
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result
        
    words = re.findall(r"\b(yes|no)\b", text)
    if words:
        ans = words[-1]
        result["lenient_answer"] = ans
        result["final_answer"] = ans
        result["parse_method"] = "lenient"
        result["parse_success"] = True
        
    return result

def parse_game24(response: str) -> dict:
    """Parses Game24 output. Extracts the mathematical expression."""
    result = {"strict_answer": None, "lenient_answer": None, "final_answer": None, "parse_method": "failed", "parse_success": False}
    if not response: return result
    
    lines = response.split('\n')
    for line in reversed(lines):
        if "=" in line and "24" in line:
            eq = line.split("=")[0].strip()
            result["strict_answer"] = eq
            result["final_answer"] = eq
            result["parse_method"] = "strict"
            result["parse_success"] = True
            return result
            
    # Lenient: just find any line with an operator
    for line in reversed(lines):
        if any(op in line for op in ['+', '-', '*', '/']):
            # attempt to clean
            eq = line.split("=")[0].strip()
            result["lenient_answer"] = eq
            result["final_answer"] = eq
            result["parse_method"] = "lenient"
            result["parse_success"] = True
            return result
            
    return result

def parse_math(response: str) -> dict:
    """Parses MATH output looking for \\boxed{...} and normalizes."""
    result = {"strict_answer": None, "lenient_answer": None, "final_answer": None, "parse_method": "failed", "parse_success": False}
    if not response: return result
    
    # Try to find \boxed{...}
    # It might have nested braces, but regex for nested braces is hard. 
    # Let's extract everything inside the last \boxed{
    boxed_idx = response.rfind("\\boxed{")
    if boxed_idx != -1:
        start_idx = boxed_idx + 7
        brace_count = 1
        end_idx = start_idx
        while end_idx < len(response) and brace_count > 0:
            if response[end_idx] == '{': brace_count += 1
            elif response[end_idx] == '}': brace_count -= 1
            end_idx += 1
        
        if brace_count == 0:
            val = response[start_idx:end_idx-1]
            
            # sqrt normalization \sqrt{X} -> numeric if perfect square
            sqrt_match = re.search(r"\\sqrt\{([^{}]+)\}", val)
            if sqrt_match:
                inner = sqrt_match.group(1).strip()
                try:
                    inner_val = float(inner)
                    import math as _math
                    root = _math.sqrt(inner_val)
                    if root == int(root):
                        val = str(int(root))
                    else:
                        val = str(root)
                except (ValueError, TypeError):
                    # Leave as sqrt(X) for sympy to handle
                    val = val.replace("\\sqrt{" + inner + "}", f"sqrt({inner})")

            # fraction normalization \frac{a}{b} -> a/b -> eval
            frac_match = re.search(r"\\frac{([^{}]+)}{([^{}]+)}", val)
            if frac_match:
                try:
                    num = float(frac_match.group(1).strip())
                    den = float(frac_match.group(2).strip())
                    if den != 0:
                        val = str(num / den)
                except:
                    pass

            # normalize remaining
            val = val.replace("\\text{", "").replace("}", "")
            val = val.replace(",", "")
            if val.endswith('.0'): val = val[:-2]

            result["strict_answer"] = val
            result["final_answer"] = val
            result["parse_method"] = "strict"
            result["parse_success"] = True
            return result
            
    # Lenient fallback: find last number or fraction
    # Not required for math strictness, but good for lenient
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", response)
    if nums:
        val = nums[-1]
        result["lenient_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "lenient"
        result["parse_success"] = True
        
    return result

def parse_aqua(response: str) -> dict:
    """Parses AQuA multiple choice letters (a)-(e)."""
    result = {"strict_answer": None, "lenient_answer": None, "final_answer": None, "parse_method": "failed", "parse_success": False}
    if not response: return result
    
    # Try case-insensitive matching of answer patterns first
    match = re.search(r"(?i)answer\s*(?:is|:)?\s*\(?([a-e])\)?", response)
    if match:
        val = match.group(1).upper()
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result
        
    # Check other patterns like "Option X", "choose (X)"
    match = re.search(r"(?i)(?:option|choose)\s*\(?([a-e])\)?", response)
    if match:
        val = match.group(1).upper()
        result["strict_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "strict"
        result["parse_success"] = True
        return result

    # Lenient fallback: just the last freestanding letter a-e
    matches = re.findall(r"(?i)\b([a-e])\b", response)
    if matches:
        val = matches[-1].upper()
        result["lenient_answer"] = val
        result["final_answer"] = val
        result["parse_method"] = "lenient"
        result["parse_success"] = True
        
    return result

def get_parser(task: str):
    if task in ["gsm8k", "gsm_hard", "svamp"]: return parse_gsm8k
    elif task == "strategyqa": return parse_strategyqa
    elif task == "game24": return parse_game24
    elif task == "math": return parse_math
    elif task == "aqua": return parse_aqua
    else: raise ValueError(f"Unknown task {task}")
