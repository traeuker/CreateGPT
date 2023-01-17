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
        if len(value.shape) == 2 and value.shape == pretrained_value.T.shape:
            state_dict_to_load[key] = pretrained_value.T
            # print(f"Copied params.T: {pretrained_key} -> {key}")
        elif value.shape == pretrained_value.shape:
            state_dict_to_load[key] = pretrained_value
            # print(f"Copied params:   {pretrained_key} -> {key}")
        else:
            raise Exception(f"Parameter shapes don't match: {key} with {value.shape} vs {pretrained_key} with {pretrained_value.shape}")


    model.load_state_dict(state_dict_to_load)
    return model


def get_tokenizer():
    '''Get the tokenizer for the GPT2 model.'''
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    return tokenizer