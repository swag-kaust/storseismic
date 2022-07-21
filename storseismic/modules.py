import transformers
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
import torch
import torch.nn as nn
from typing import Optional, Tuple
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

class DenseSynthesizerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.max_length)
        )

    def forward(self, x):
        output = self.dense(x)

        return output

class RandomSynthesizerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.fixed:
            self.attention = nn.Parameter(torch.empty(config.max_length, config.max_length), requires_grad=False)
        else:
            self.attention = nn.Parameter(torch.empty(config.max_length, config.max_length), requires_grad=True)
        nn.init.xavier_uniform_(self.attention)

    def forward(self):
        output = self.attention

        return output

class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.max_length = config.max_length
        self.attention_type = config.attention_type

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        if self.attention_type != "default":
            for params in self.query.parameters():
                 params.requires_grad = False
            for params in self.key.parameters():
                 params.requires_grad = False
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.attention_type == "dense_synth":
            self.head = nn.ModuleList([DenseSynthesizerHead(config) for _ in range(config.num_attention_heads)])
        elif self.attention_type == "rand_synth":
            self.head = nn.ModuleList([RandomSynthesizerHead(config) for _ in range(config.num_attention_heads)])

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        if self.attention_type == "default":
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        elif self.attention_type == "dense_synth":
            scores_shape =  (hidden_states.size()[0], self.num_attention_heads, self.max_length, self.max_length)
            attention_scores = torch.empty(scores_shape, device=hidden_states.device)
            for i, head_module in enumerate(self.head):
                attention_scores[:, i] = head_module(hidden_states)
        elif self.attention_type == "rand_synth":
            scores_shape =  (self.num_attention_heads, self.max_length, self.max_length)
            attention_scores = torch.empty(scores_shape, device=hidden_states.device)
            for i, head_module in enumerate(self.head):
                attention_scores[i] = head_module()
            attention_scores = attention_scores.unsqueeze(0).repeat(hidden_states.size()[0], 1, 1, 1)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
