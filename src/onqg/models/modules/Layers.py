''' Define the Layers '''
import torch
import torch.nn as nn
from onqg.models.modules.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from onqg.models.modules.Attention import GatedSelfAttention, GraphAttention
import onqg.dataset.Constants as Constants


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, slf_attn, d_inner, n_head, d_k, d_v, 
                 dropout=0.1, attn_dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = slf_attn
        if slf_attn == 'multi-head':
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=attn_dropout)
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        else:
            self.gated_slf_attn = GatedSelfAttention(d_model, d_k, dropout=attn_dropout)

    def forward(self, enc_input, src_seq, non_pad_mask=None, slf_attn_mask=None, layer_id=-1):
        if self.slf_attn == 'gated':
            mask = (src_seq == Constants.PAD).unsqueeze(2) if slf_attn_mask is None else slf_attn_mask
            enc_output, enc_slf_attn = self.gated_slf_attn(enc_input, mask)
        else:
            enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
            enc_output *= non_pad_mask

            enc_output = self.pos_ffn(enc_output)
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class GGNNEncoderLayer(nn.Module):
    '''GGNN Layer'''
    def __init__(self, d_hidden, d_model, alpha, feature=False, dropout=0.1, attn_dropout=0.1):
        super(GGNNEncoderLayer, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.feature = feature

        self.edge_in_emb = nn.Linear(d_hidden, d_model)
        self.edge_out_emb = nn.Linear(d_hidden, d_model)

        self.output_gate = Propagator(d_model, dropout=dropout)

    def forward(self, nodes, mask, node_type, feat_hidden=None):
        ###=== concatenation ===###
        node_hidden = nodes     # batch_size x node_num x d_model
        ###=== transform using edge matrix ===###
        node_in_hidden = self.edge_in_emb(node_hidden)
        node_out_hidden = self.edge_out_emb(node_hidden)
        ###=== gated recurrent unit ===###
        node_output = self.output_gate(nodes, node_in_hidden, node_out_hidden)

        return node_output


class GATEncoderLayer(nn.Module):
    '''GAT Layer'''
    def __init__(self, d_hidden, d_model, alpha, feature=False, dropout=0.1, attn_dropout=0.1):
        super(GATEncoderLayer, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.feature = feature

        self.transf = nn.Linear(d_hidden, d_model)
        self.graph_attention = GraphAttention(d_hidden, d_model, alpha, dropout=attn_dropout)

    def forward(self, nodes, mask, node_type, feat_hidden=None):
        ###=== concatenation ===###
        node_hidden = nodes     # batch_size x node_num x d_model
        ###=== transform ===###
        node_hidden = self.transf(node_hidden)
        ###=== graph attention ===###
        node_src_hidden = node_hidden.unsqueeze(2).repeat(1, 1, nodes.size(1), 1).view(nodes.size(0), -1, self.d_hidden)
        node_output = self.graph_attention(node_src_hidden, node_hidden.repeat(1, nodes.size(1), 1), mask)

        return node_output


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, 
                 addition_input=0, dropout=0.1, n_enc_layer=0,
                 layer_attn=False, two_step=False):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        
        self.layer_attn, self.two_step = layer_attn, two_step
        if layer_attn and self.two_step:
            self.enc_layer_attn = nn.ModuleList([
                MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout) for _ in range(n_enc_layer)
            ])
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, addition_input=addition_input, dropout=dropout)

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask
        
        if self.layer_attn and self.two_step:
            ### first step: respectively cross attention among each encoder-output with decoder-input ###
            enc_output = [enc_attn(dec_output, enc_output[i], enc_output[i], mask=dec_enc_attn_mask) 
                            for i, enc_attn in enumerate(self.enc_layer_attn)]
            enc_output = torch.stack([e_opt[0] for e_opt in enc_output], dim=2) # batch_size x tgt_dim x layer_num x dim

            ### second step: cross attention on layer-wise representation with decoder-input ###
            batch_size, layer_num, dim = enc_output.size(0), enc_output.size(2), enc_output.size(3)
            enc_output = enc_output.view(-1, layer_num, dim)    # (batch_size x tgt_len) x layer_num x dim
            dec_output = dec_output.view(-1, 1, dim)   # (batch_size x tgt_len) x 1 x dim
            dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output)
            dec_output = dec_output.view(batch_size, -1, dim)   # batch_size x tgt_len x dim
        else:
            dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
