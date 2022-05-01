import transformers
import torch
import torch.nn as nn
import math

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()
        
    def forward(self, x):
        return 0

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.pe[:x.size(1)]
        return self.dropout(x)
    
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = PositionalEncoding(d_model=config.hidden_size,
                                                      max_len=config.max_position_embeddings)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        
    def forward(self, inputs_embeds, input_ids=None, position_ids=None, token_type_ids=None,
               past_key_values_length=None):
        embeddings = self.word_embeddings(inputs_embeds)
        position_ids = self.position_ids
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings += position_embeddings.swapaxes(0, 1)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertEmbeddings2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.position_embeddings = PositionalEncoding(d_model=config.hidden_size,
                                                      max_len=config.max_position_embeddings)
        if config.add_pos_embed:
            self.position_embeddings2 = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        else:
            self.position_embeddings2 = None
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        
    def forward(self, inputs_embeds, input_ids=None, position_ids=None, token_type_ids=None,
               past_key_values_length=None):
        embeddings = self.word_embeddings(inputs_embeds)
        position_ids = self.position_ids
        position_embeddings = self.position_embeddings(self.position_ids)
        if self.position_embeddings2 is not None:
            position_embeddings2 = self.position_embeddings2(self.position_ids)
        embeddings += position_embeddings.swapaxes(0, 1)
        embeddings += position_embeddings2
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertEmbeddings3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        # self.position_embeddings = PositionalEncoding(d_model=config.hidden_size,
        #                                               max_len=config.max_position_embeddings)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.position_ids = torch.arange(config.max_position_embeddings).expand((1, -1))
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        
    def forward(self, inputs_embeds, input_ids=None, position_ids=None, token_type_ids=None,
               past_key_values_length=None):
        embeddings = self.word_embeddings(inputs_embeds)
        position_ids = self.position_ids
        position_embeddings = self.position_embeddings(self.position_ids)
        embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output

class PreLNBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states
    
class PreLNBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self = transformers.models.bert.modeling_bert.BertSelfAttention(config)
        self.output = transformers.models.bert.modeling_bert.BertSelfOutput(config)
        self.pruned_heads = set()
        
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, dim=1)
        
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
        
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        hidden_states = self.LayerNorm(hidden_states)
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs
    
class PreLNBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = transformers.activations.ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
            
    def forward(self, hidden_states):
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    
class PreLNBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_tensor
        return hidden_states

class DenoisingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output

class VelpredHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vel_size)
        self.predictions.decoder = Identity()
        self.vel_min = config.vel_min
        self.vel_max = config.vel_max
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        output = torch.mean(output[:, :1, :], dim=1)
        output = self.vel_min + (output + 1) * (self.vel_max - self.vel_min) * 0.5
        
        return output

class LowFreqHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output

class FirstBreakHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        
        return output

class FirstBreakHead2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, 1)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        output = output.squeeze()
        
        return output

# class FirstBreakHead3(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.act_fn = nn.Sigmoid()
#         self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
#         self.predictions.decoder = Identity()
        
#     def forward(self, sequence_output):
#         output = self.predictions(sequence_output)
#         output = self.act_fn(output.swapaxes(1, 2))

#         return output

# class FirstBreakHead3(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.act_fn = nn.Sigmoid()
#         self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
#         self.predictions.decoder = Identity()
        
#     def forward(self, sequence_output):
#         output = self.predictions(sequence_output)
#         output = self.act_fn(output)
#         output = output.swapaxes(1, 2)

#         return output

class FirstBreakHead3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn = nn.Sigmoid()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.act_fn(sequence_output)
        output = self.predictions(output)
        output = output.swapaxes(1, 2)

        return output

class FirstBreakHead4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = nn.Linear(config.hidden_size, config.vocab_size)
        self.predictions.decoder = Identity()
        
    def forward(self, sequence_output):
        output = self.predictions(sequence_output)
        output = output.swapaxes(1, 2)
        
        return output