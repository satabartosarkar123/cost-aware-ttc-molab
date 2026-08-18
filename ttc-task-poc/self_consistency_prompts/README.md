# Self-Consistency Prompt Examples
# (Wang et al., ICLR 2023)
#
# These prompts follow the "sample-and-marginalize" method:
#   1. Use a Chain-of-Thought prompt
#   2. Sample multiple reasoning paths at temperature > 0
#   3. Parse each answer and take majority vote
#
# The CoT prompt template is managed in experiment/prompts/raw_text/cot.txt
# This directory exists for reference and any additional raw prompt variants.
