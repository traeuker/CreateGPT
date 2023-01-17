import torch
from torch.utils.data import Dataset
import re

class WordsDataset(Dataset):
    def __init__(self, words, max_seq_len=1024, fraction=0.1):
        self.words = words
        self.vocab_size = len(set(words))
        self.seq_len = max_seq_len
        self.fraction = fraction
        self.max_len = int(len(self.words)  - (self.seq_len + 1))

        # if we call set on words we reduce the corpus for words that appear double 
        # sorted (alphabetically) enumerated from 1 to dict_length 
        # we get a set of words with their corresponding number
        self.words_to_token_idx = {word: idx for (idx, word) in enumerate(sorted(set(words)))}
        
        # that dict but reversed, getting a set of numbers with their corresponding words
        self.token_idx_to_words = {idx: word for (word, idx) in self.words_to_token_idx.items()}
        assert len(self.token_idx_to_words) == (self.vocab_size)

        # this is our corpus of words in tokens 
        self.tokens = torch.tensor([self.words_to_token_idx[word] for word in words]).to(dtype=torch.long)

    def __len__(self):
        return int(len(self.words) * self.fraction)

    def __getitem__(self, idx):
        
        next_tokens = self.tokens[idx +1: idx + self.seq_len+1] # last token of sequenth length
        tokens = self.tokens[idx: idx + self.seq_len] # all token before that 

        return tokens, next_tokens


class WordsTokenizer():
    def __init__(self, wordsdataset: WordsDataset):
        self.words_to_token_idx = wordsdataset.words_to_token_idx
        self.token_idx_to_words = wordsdataset.token_idx_to_words
        self.model_max_length = wordsdataset.max_len
      
    def encode(self, initial_text: str, return_tensors=None):
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        split_text = re.split(r"\b", initial_text)
        token_list = []
        for t in split_text:
            if len(t)>0:
                token_list.append( self.words_to_token_idx[t] )
        if return_tensors is None:
            return token_list
        elif return_tensors == 'pt':
            return torch.tensor(token_list)
        else:
            raise Exception("Invalid return_tensor, either _pt_ or None")

    def decode(self, list_of_ids) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        sentence = ''
        for t in list_of_ids:
            sentence += '' + str(self.token_idx_to_words[int(t)])
        return sentence


class WordsDatasetTokenized(Dataset):
    def __init__(self, tokenizer, dataset, max_seq_len=1024, fraction=0.1):
        
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.vocab_size = tokenizer.vocab_size
        self.seq_len = max_seq_len
        self.fraction = fraction

        # tokenize the dataset with the tokenizer
        # The tokenizer thinks the max length allowed is 1024, of course this 
        # is a much bigger dataset, so it throws an warning 
        tokens = self.tokenizer.encode(dataset, return_tensors='pt')[0]
        self.max_len = int(len(tokens)  - (self.seq_len + 1))
        self.tokens = tokens

        # convert the tokens to a list of string tokens with tokenizer
        # iterative mit for [x in list bal]
        self.words = [tokenizer.decode([x]) for x in self.tokens]

    def __len__(self):
        return int(len(self.tokens) * self.fraction)

    def __getitem__(self, idx):
        
        next_tokens = self.tokens[idx +1: idx + self.seq_len+1] # last token of sequenth length
        tokens = self.tokens[idx: idx + self.seq_len] # all token before that 

        return tokens, next_tokens
    

def get_dataloaders(dataset, batch_size=32):
    '''
    Returns a tuple of dataloaders for training and validation.
    '''
    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader
