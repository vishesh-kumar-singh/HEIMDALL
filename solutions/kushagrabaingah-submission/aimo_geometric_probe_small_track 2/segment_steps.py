import re
import numpy as np

def segment_steps(
    cot_text: str, 
    tokenizer=None, 
    prompt_prefix: str = "",
    min_steps: int = 4, 
    max_steps: int = 60,
    fallback_window_tokens: int = 96
) -> tuple[list[str], list[tuple[int, int]]]:
    """
    Segments a Chain-of-Thought string into reasoning steps and computes token spans.
    
    Returns:
        steps: list of string contents for each step [s_0, s_1, ..., s_{T-1}]
        step_spans: list of (start_token_idx, end_token_idx) relative to the full prompt+CoT sequence
    """
    cot_text = cot_text.strip()
    
    # 1. Primary: Structural / Regex Segmentation
    # Split on double newlines, explicit step markers, and reasoning transition anchors
    pattern = r'(?:\n\s*\n|(?<=\n)(?=Step \d+:|### Step|\*\*Step \d+:)|(?<=\n)(?=Wait,|Let me check|So,|Therefore,))'
    raw_splits = re.split(pattern, cot_text)
    steps = [s.strip() for s in raw_splits if len(s.strip()) > 5]
    
    # 2. Fallback: Fixed-window chunking if too few or too many steps
    if len(steps) < min_steps or len(steps) > max_steps:
        sentences = re.split(r'(?<=[.!?\n])\s+', cot_text)
        merged_steps = []
        curr = ""
        for sent in sentences:
            curr += " " + sent
            if len(curr.split()) >= fallback_window_tokens:
                merged_steps.append(curr.strip())
                curr = ""
        if curr.strip():
            merged_steps.append(curr.strip())
        if len(merged_steps) >= min_steps:
            steps = merged_steps

    if len(steps) < min_steps:
        return [], []

    # 3. Compute exact token spans for each step
    step_spans = []
    if tokenizer is not None:
        full_text = prompt_prefix + cot_text
        encoded_full = tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = encoded_full.get("offset_mapping", None)
        
        prefix_len = len(prompt_prefix)
        char_cursor = prefix_len
        
        for step_str in steps:
            step_start_char = full_text.find(step_str, char_cursor)
            if step_start_char == -1:
                step_start_char = char_cursor
            step_end_char = step_start_char + len(step_str)
            char_cursor = step_end_char
            
            if offset_mapping is not None:
                offsets = offset_mapping[0].tolist()
                tok_start, tok_end = None, None
                for t_idx, (c_start, c_end) in enumerate(offsets):
                    if c_end > step_start_char and tok_start is None:
                        tok_start = t_idx
                    if c_end >= step_end_char and tok_end is None:
                        tok_end = t_idx
                        break
                if tok_start is None:
                    tok_start = 0
                if tok_end is None:
                    tok_end = len(offsets) - 1
                step_spans.append((tok_start, tok_end))
            else:
                step_spans.append((0, 0))
    else:
        step_spans = [(i, i) for i in range(len(steps))]

    return steps, step_spans
