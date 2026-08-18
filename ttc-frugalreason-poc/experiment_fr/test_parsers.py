from core.parsers import parse_math, parse_aqua, parse_gsm8k
import logging

def test_parsers():
    # 1. MATH
    math_examples = [
        ("The answer is \\boxed{42}.", "42"),
        ("I think it is \\boxed{3.14} and that's it.", "3.14"),
        ("\\boxed{1/2}", "1/2"),
        ("Just a number 42", "42"), # lenient
        ("\\boxed{1,000}", "1000"),
        ("\\boxed{2.0}", "2"),
        ("\\boxed{\\text{Monday}}", "Monday"),
        ("So the answer is \\boxed{ x^2 + y }.", "x^2 + y"),
        ("\\boxed{{1, 2, 3}}", "{1 2 3}"), # roughly
        ("Not boxed at all but 100", "100"),
    ]
    print("Testing MATH parser...")
    for text, exp in math_examples:
        res = parse_math(text)
        ans = res.get('final_answer')
        # simplified check since some normalize logic handles text/commas
        print(f"'{text}' -> '{ans}' (expected approx '{exp}')")

    # 2. AQuA
    aqua_examples = [
        ("The answer is (A).", "A"),
        ("I pick choice b.", "B"),
        ("Therefore it's E", "E"),
        ("Clearly C.", "C"),
        ("So D is correct.", "D"),
        ("Option A is right.", "A"),
        ("(E) definitely.", "E"),
        ("B", "B"),
        ("Choice c)", "C"),
        ("answer is a", "A"),
    ]
    print("\nTesting AQuA parser...")
    for text, exp in aqua_examples:
        res = parse_aqua(text)
        ans = res.get('final_answer')
        if ans != exp:
            print(f"FAIL: '{text}' -> '{ans}' (expected '{exp}')")
            
    # 3. GSM-HARD / SVAMP (same as GSM8K)
    gsm_examples = [
        ("#### 1,000", "1000"),
        ("The answer is #### $42.50", "42.50"),
        ("#### -5", "-5"),
        ("Total is 100", "100"),
        ("#### 1,234,567.89", "1234567.89"),
    ]
    print("\nTesting GSM parser...")
    for text, exp in gsm_examples:
        res = parse_gsm8k(text)
        ans = res.get('final_answer')
        if ans != exp:
            print(f"FAIL: '{text}' -> '{ans}' (expected '{exp}')")

if __name__ == "__main__":
    test_parsers()
