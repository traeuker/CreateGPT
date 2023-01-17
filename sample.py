import torch

def sample_tokens(
    model,
    tokenizer,
    initial_text: str,
    max_tokens_generated: int = 30,
    **kwargs
) -> str:
    '''
    Sample tokens until the model outputs `tokenizer.eos_token_id` or the specified token limit is reached.

    Return: the prompt and continuation concatenated
    '''
    assert max_tokens_generated < model.config.max_seq_len
    model.eval()
    input_ids: list = tokenizer.encode(initial_text)
    generated = []
    device = next(model.parameters()).device

    for _ in range(max_tokens_generated):
        new_input_ids = torch.tensor(
            input_ids + generated, dtype=torch.int64, device=device)
        new_input_ids_truncated = new_input_ids[-min(
            tokenizer.model_max_length, new_input_ids.shape[0]):].unsqueeze(0)
        output = model(new_input_ids_truncated)
        all_logits = output if isinstance(output, torch.Tensor) else output.logits
        logits = all_logits[0, -1]
        new_token = apply_sampling_methods(new_input_ids, logits, **kwargs)
        generated.append(new_token)
        if new_token == getattr(tokenizer, "eos_token_id", None):
            break
    return tokenizer.decode(input_ids + generated)



def apply_sampling_methods(
    input_ids: torch.Tensor, logits: torch.Tensor, temperature=1.0, freq_penalty=0.0, top_k=0, top_p=0.0
) -> int:
    '''
    Return the next token, sampled from the model's probability distribution with modifiers.
    input_ids: shape (seq,)
    '''
    assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
    assert temperature >= 0, "Temperature should be non-negative"
    assert 0 <= top_p <= 1.0, "Top-p must be a probability"
    assert 0 <= top_k, "Top-k must be non-negative"
    assert not (top_p != 0 and top_k !=
                0), "At most one of top-p and top-k supported"

    if temperature == 0:
        return greedy_search(logits)
    if temperature != 1.0:
        logits = apply_temperature(logits, temperature)
    if freq_penalty != 0.0:
        logits = apply_freq_penalty(input_ids, logits, freq_penalty)
    if top_k > 0:
        return sample_top_k(logits, top_k)
    if top_p > 0:
        return sample_top_p(logits, top_p)
    return sample_basic(logits)


def greedy_search(logits: torch.Tensor) -> int:
    '''
    logits: shape (vocab_size, )

    Return: the most likely token (as an integer)
    '''
    return torch.argmax(logits)


def sample_basic(logits: torch.Tensor) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    sampled_token = torch.distributions.categorical.Categorical(logits=logits)
    return sampled_token.sample()


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    '''
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    assert temperature > 0
    logits = logits / temperature
    return logits


def apply_freq_penalty(input_ids: torch.Tensor, logits: torch.Tensor, freq_penalty: float) -> torch.Tensor:
    '''
    input_ids: shape (seq, )
    logits: shape (vocab_size, )

    Return: shape (vocab_size, )
    '''
    count = torch.bincount(input_ids)
    for idx, i in enumerate(count):
        # Probably very inefficient: do better.
        if i > 1:
            logits[idx] = logits[idx] - i * freq_penalty

    return logits


def sample_top_k(logits: torch.Tensor, top_k: int) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities
    top_k: only consider this many of the most likely tokens for sampling

    Return: a sampled token
    '''
    top, top_idx = torch.topk(logits, top_k)

    sampled_token = torch.distributions.categorical.Categorical(logits=top)

    return top_idx[sampled_token.sample().item()]


def sample_top_p(logits: torch.Tensor, top_p: float, min_tokens_to_keep: int = 1) -> int:
    '''
    logits: shape (vocab_size, ) - unnormalized log-probabilities

    Return: a sampled token
    '''
    sorted, indices = torch.sort(logits, descending=True)
    cum_probs = sorted.softmax(-1).cumsum(-1)
    sorted_cum_probs = torch.searchsorted(
        cum_probs, top_p, side="right").item() + 1
    if sorted_cum_probs < min_tokens_to_keep:
        sorted_cum_probs = min_tokens_to_keep
    idx = indices[:sorted_cum_probs]
    keep_logits = logits[idx]
    sample_token = torch.distributions.categorical.Categorical(
        logits=keep_logits).sample()

    return idx[sample_token].item()
