import torch
import transformers

def copy_weights(model):
    '''Copy over the weights of `GPT2` to the transformer.'''

    transformer_dict = model.named_parameters()

    GPT = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    pretrained_dict = GPT.named_parameters()

    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict_to_load = {}

    for (key, value), (pretrained_key, pretrained_value) in zip(transformer_dict, pretrained_dict):
        if "embedding" in key:
            state_dict_to_load[key] = pretrained_value
        else: 
            # The weights are transposed because the GPT2 weights are in the shape (hidden_size, num_heads)
            state_dict_to_load[key] = torch.transpose(pretrained_value, dim0=0, dim1=-1)

    model.load_state_dict(state_dict_to_load)
    return model


def get_tokenizer():
    '''Get the tokenizer for the GPT2 model.'''
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    return tokenizer