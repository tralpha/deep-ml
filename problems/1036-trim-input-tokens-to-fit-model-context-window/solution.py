import numpy as np

def trim_to_context(input_ids, context_len: int, max_new_tokens: int) -> list:
    """
    Trim input_ids from the left so that seq_len + max_new_tokens <= context_len.

    Args:
        input_ids: 2D array-like of token IDs, shape (batch_size, seq_len).
        context_len: model's maximum context length.
        max_new_tokens: number of tokens that will be generated.

    Returns:
        A nested Python list representing the trimmed token IDs.
    """
    input_ids = np.array(input_ids, int)
    # VALIDATE
    assert context_len > 0 and max_new_tokens >= 0, "Context length must be strictly greater than 0; max_new_tokens must not be negative"
    # BUDGET
    allowed = context_len - max_new_tokens 
    if allowed <= 0:
        return [[] for _ in range(input_ids.shape[0])]
    # TRIM
    return input_ids[:, -allowed:].tolist()