import torch
import torch.nn as nn

import onqg.dataset.Constants as Constants

from onqg.models.modules.MaxOut import MaxOut
from onqg.models.modules.Layers import DecoderLayer

from onqg.utils.mask import get_non_pad_mask, get_subsequent_mask, get_attn_key_pad_mask
from onqg.utils.sinusoid import get_sinusoid_encoding_table


class TransfDecoder(nn.Module):
    def __init__(self, n_vocab, len_max_seq, d_word_vec, d_model, n_layer,
                 d_inner, n_head, d_k, d_v, layer_attn, n_enc_layer, 
                 feat_vocab, d_feat_vec, maxout_pool_size, dropout, mode='normal'):
        self.name = 'transf'

        super(TransfDecoder, self).__init__()

        n_position = len_max_seq + 5
        self.layer_attn = layer_attn

        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=Constants.PAD),
            freeze=True
        )

        self.feature = False if not feat_vocab else True
        self.d_feat = len(feat_vocab) * d_feat_vec if self.feature else 0
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
        
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, addition_input=self.d_feat,
                         dropout=dropout, layer_attn=layer_attn, n_enc_layer=n_enc_layer)
            for _ in range(n_layer)
        ])

        self.maxout = MaxOut(maxout_pool_size)
        self.maxout_pool_size = maxout_pool_size
    
    @classmethod
    def from_opt(cls, opt):
        if 'mode' not in opt:
            opt['mode'] = 'normal'
        return cls(opt['n_vocab'], opt['len_max_seq'], opt['d_word_vec'], opt['d_model'], opt['n_layer'],
                   opt['d_inner'], opt['n_head'], opt['d_k'], opt['d_v'], opt['layer_attn'], opt['n_enc_layer'],
                   opt['feat_vocab'], opt['d_feat_vec'], opt['maxout_pool_size'], opt['dropout'], opt['mode'],)
    
    def forward(self, inputs, max_length=300, return_attns=False):
        tgt_seq, tgt_pos, feat_seqs = inputs['tgt_seq'], inputs['tgt_pos'], inputs['feat_seqs']
        src_seq, enc_output, _ = inputs['src_seq'], inputs['enc_output'], inputs['hidden']

        non_pad_mask = get_non_pad_mask(tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        if self.layer_attn:
            enc_output = torch.stack(enc_output)    # layer_num x batch_size x src_len x dim
            batch_size, layer_num, dim = enc_output.size(1), enc_output.size(0), enc_output.size(-1)

            layer_src_seq = src_seq.unsqueeze(1).repeat(1, layer_num, 1) # batch_size x layer_num x src_len
            layer_src_seq = layer_src_seq.contiguous().view(batch_size, -1)  # batch_size x (layer_num x src_len)
            dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=layer_src_seq, seq_q=tgt_seq)
            enc_output = enc_output.permute(1, 0, 2, 3).contiguous().view(batch_size, -1, dim)  # batch_size x (layer_num x src_len) x dim
        
        if self.feature:
            feat_inputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_inputs = torch.cat(feat_inputs, dim=2)
            if self.layer_attn:
                feat_inputs = feat_inputs.repeat(1, layer_num, 1)
            enc_output = torch.cat((enc_output, feat_inputs), dim=2)
        
        dec_output = self.word_emb(tgt_seq) + self.pos_emb(tgt_pos)
        
        for dec_layer in self.layer_stack:
            dec_output, *_ = dec_layer(dec_output, enc_output, 
                                       non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask,
                                       dec_enc_attn_mask=dec_enc_attn_mask)
        
        dec_output = self.maxout(dec_output)
        
        rst = {'pred':dec_output}

        return rst