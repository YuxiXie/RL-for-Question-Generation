import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

import onqg.dataset.Constants as Constants

from onqg.models.modules.Attention import GatedSelfAttention
from onqg.models.modules.MaxOut import MaxOut

from onqg.utils.mask import get_attn_key_pad_mask


class RNNEncoder(nn.Module):
    """
    Input: (1) inputs['src_seq']
           (2) inputs['lengths'] 
           (3) inputs['feat_seqs']
    Output: (1) enc_output
            (2) hidden
    """
    def __init__(self, n_vocab, d_word_vec, d_model, n_layer,
                 brnn, rnn, feat_vocab, d_feat_vec, slf_attn, 
                 dropout):
        self.name = 'rnn'

        self.n_layer = n_layer
        self.num_directions = 2 if brnn else 1
        assert d_model % self.num_directions == 0, "d_model = hidden_size x direction_num"
        self.hidden_size = d_model // self.num_directions

        super(RNNEncoder, self).__init__()

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)
        input_size = d_word_vec

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            input_size += len(feat_vocab) * d_feat_vec
        
        self.slf_attn = slf_attn
        if slf_attn:
            self.gated_slf_attn = GatedSelfAttention(d_model)
        
        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size, self.hidden_size, num_layers=n_layer,
                               dropout=dropout, bidirectional=brnn, batch_first=True)
        elif rnn == 'gru':
            self.rnn = nn.GRU(input_size, self.hidden_size, num_layers=n_layer,
                              dropout=dropout, bidirectional=brnn, batch_first=True)
        else:
            raise ValueError("Only support 'LSTM' and 'GRU' for RNN-based Encoder ")
    
    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_vocab'], opt['d_word_vec'], opt['d_model'], opt['n_layer'],
                   opt['brnn'], opt['rnn'], opt['feat_vocab'], opt['d_feat_vec'], 
                   opt['slf_attn'], opt['dropout'])
    
    def forward(self, inputs):
        src_seq, lengths, feat_seqs = inputs['src_seq'], inputs['lengths'], inputs['feat_seqs']
        lengths = torch.LongTensor(lengths.data.view(-1).tolist())
        
        enc_input = self.word_emb(src_seq)
        if self.feature:
            feat_outputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_outputs = torch.cat(feat_outputs, dim=2)
            enc_input = torch.cat((enc_input, feat_outputs), dim=-1)
        
        enc_input = pack(enc_input, lengths, batch_first=True, enforce_sorted=False)
        enc_output, hidden = self.rnn(enc_input, None)
        enc_output = unpack(enc_output, batch_first=True)[0]

        if self.slf_attn:
            # mask = (src_seq == Constants.PAD).byte()
            mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
            enc_output, score = self.gated_slf_attn(enc_output, mask)
        
        return enc_output, hidden
