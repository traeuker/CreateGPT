import os
from tqdm import tqdm
import wandb
import time
import torch
import torch.nn as nn
from fancy_einsum import einsum
import einops
from dataclasses import dataclass


device = torch.device("cpu")


@dataclass(frozen=False)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05


class MLP(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.linear_1 = nn.Linear(self.hidden_size, 4*self.hidden_size)
        self.linear_2 = nn.Linear(4*self.hidden_size, self.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.linear_1(x))
        x = self.dropout(self.linear_2(x))
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, max_sequence_length: int = 32, hidden_dim: int = 128, load_existing_model=False):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
        maximal_pe = self.get_maximal_pe()

        if load_existing_model:
            self.maximal_pe = nn.Linear(max_sequence_length, hidden_dim)
        else:
            self.register_buffer('maximal_pe', maximal_pe)

    def get_maximal_pe(self):

        def PE(delta):
            hidden_dim = self.hidden_dim

            sin_vec = torch.sin(
                delta / 10000**(2 * torch.arange(hidden_dim // 2) / hidden_dim))
            cos_vec = torch.cos(
                delta / 10000**(2 * torch.arange(hidden_dim // 2) / hidden_dim))

            pe = torch.zeros(hidden_dim)
            pe[::2] = sin_vec
            pe[1::2] = cos_vec

            return pe

        pe = torch.stack([PE(i) for i in range(self.max_sequence_length)])
        return pe

    def forward(self, x):
        '''
        x: shape (n, seq_len, hidden_dim)
        '''
        x = einops.rearrange(x, '')
        return x + self.maximal_pe[:x.size(1), :].to(device)


def multihead_masked_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                               num_heads: int):
    '''
    Implements multihead masked attention on the matrices Q, K and V.
    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)
    Return: shape (batch, seq_len, nheads*headsize)
    '''

    emb_len = Q.shape[-1]
    seq_len = Q.shape[-2]
    headsize = emb_len // num_heads

    Q_ = einops.rearrange(Q, 'b s (nh h) -> b nh s h', nh=num_heads)
    K_ = einops.rearrange(K, 'b s (nh h) -> b nh s h', nh=num_heads)
    V_ = einops.rearrange(V, 'b s (nh h) -> b nh s h', nh=num_heads)

    QKT = einsum('b nh s_q h, b nh s_k h -> b nh s_q s_k', Q_, K_)
    QKT = QKT / (headsize**0.5)
    tri = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1)*(-10 ** 4)
    QKT_masked = (QKT + tri)
    attention_probs = torch.softmax(QKT_masked, dim=-1)

    attention_values_ = einsum('b nh s_q s_k, b nh s_k h -> b nh s_q h',
                               attention_probs, V_)
    # b hn s_q h -->e = n*h --> b s_q e
    attention_values = einops.rearrange(attention_values_,
                                        ' b hn s_q h ->  b s_q (hn h)')

    return attention_values


class MultiheadMaskedAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert (hidden_size % num_heads) == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        Return: shape (batch, seq, hidden_size)
        '''
        # QKV = einsum('b s hs, hs h2 -> b (s h2) hs',x, self.W_QKV) # h2 = 3*emb_len
        QKV = self.W_QKV(x)
        Q, K, V = einops.rearrange(QKV, 'b hs (n es) -> n b hs es', n=3)
        av = multihead_masked_attention(Q, K, V, self.num_heads)
        # av shape: b s_q emb
        # W_O shape: emb, emb
        out = self.W_O(av)
        return out


class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.layer_norm_1 = nn.LayerNorm(
            self.config.hidden_size, eps=config.layer_norm_epsilon)
        self.multiheaded_self_attention = MultiheadMaskedAttention(
            config.hidden_size, config.num_heads)
        self.layer_norm_2 = nn.LayerNorm(
            self.config.hidden_size, eps=config.layer_norm_epsilon)
        self.MLP = MLP(self.config.hidden_size, self.config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.layer_norm_1(x)
        x = self.multiheaded_self_attention(x)
        x = x + res
        res = x
        x = self.layer_norm_2(x)
        x = self.MLP(x)
        x = x + res
        return x


class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, embedding_dim=config.hidden_size)
        self.positional_embedding = PositionalEncoding(
            config.max_seq_len, hidden_dim=config.hidden_size, load_existing_model=True)
        self.positional_embedding = nn.Embedding(
            config.max_seq_len, embedding_dim=config.hidden_size,)
        self.dropout = nn.Dropout(p=config.dropout)
        self.decoder_blocks = nn.Sequential(
            *[DecoderBlock(config) for _ in range(config.num_layers)])
        self.layer_norm_final = nn.LayerNorm(
            config.hidden_size, config.layer_norm_epsilon)
        self.pos_vector = torch.arange(0, config.max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: shape (batch, seq)
        Return: shape (batch, seq, vocab_size)  
        '''
        x1 = self.token_embedding(x)
        x2 = self.positional_embedding(self.pos_vector[:x.shape[-1]])
        x = x1 + x2
        x = self.dropout(x)
        x = self.decoder_blocks(x)
        x = self.layer_norm_final(x)
        x = einsum('word emb, b seq emb -> b seq word',
                   self.token_embedding.weight, x)
        return x


def train(model,
          optimizer,
          trainloader,
          testloader,
          loss_fn,
          num_epochs=3,
          save_dir=None,
          device=device,
          WANDB=False):

    since = time.time()
    model.to(device)

    print("Beginning Training")

    if WANDB:
        wandb.watch(model, log="all", log_freq=20, log_graph=True)

    for epoch in range(num_epochs):

        model.train()

        training_loss, running_loss = 0.0, 0.0

        progress_bar = tqdm(trainloader)
        for (x, y) in progress_bar:

            x = x.to(device)
            y = y.to(device)

            preds = model(x)
            preds_rearranged = einops.rearrange(preds, "b s v -> (b s) v")

            y_rearranged = einops.rearrange(y, "b s -> (b s)")

            training_loss = loss_fn(preds_rearranged, y_rearranged)

            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += training_loss.item() * x.size(0)  # scale to n in batch
            progress_bar.set_description(
                "Epoch {} Training Loss {:.4f}".format(epoch, training_loss))
            if WANDB:
                wandb.log({"training_loss": training_loss.item()})

        epoch_loss = running_loss / len(trainloader.dataset)
        print('Epoch {} Loss: {:.4f}'.format(epoch, epoch_loss))

        x_test, y_test = next(iter(testloader))

        x_test = x_test.to(device)
        y_test = y_test.to(device)

        model.eval()
        preds_test = model(x_test)

        def accuracy(preds, y):
            preds = preds.argmax(dim=-1)
            return (preds == y).float().mean().item()

        test_acc = accuracy(preds_test, y_test)
        print('Test Accuracy: {:.4f}'.format(test_acc))

        # save model
        if save_dir is not None:
            torch.save(model.state_dict(), os.path.join(
                save_dir, "model_{}.pt".format(epoch)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model
