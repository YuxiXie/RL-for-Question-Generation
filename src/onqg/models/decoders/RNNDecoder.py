import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import onqg.dataset.Constants as Constants

from onqg.models.modules.Attention import ConcatAttention
from onqg.models.modules.MaxOut import MaxOut
from onqg.models.modules.DecAssist import StackedRNN, DecInit


class RNNDecoder(nn.Module):
    """
    Input: (1) inputs['tgt_seq']
           (2) inputs['src_seq']
           (3) inputs['src_indexes']
           (4) inputs['enc_output']
           (5) inputs['hidden']
           (6) inputs['feat_seqs']
    Output: (1) rst['pred']
            (2) rst['attn']
            (3) rst['context']
            (4) rst['copy_pred']; rst['copy_gate']
            (5) rst['coverage_pred']

    """
    def __init__(self, n_vocab, ans_n_vocab, d_word_vec, d_model, n_layer,
                 rnn, d_k, feat_vocab, d_feat_vec, d_enc_model, n_enc_layer, 
                 input_feed, copy, answer, separate, coverage, layer_attn,
                 maxout_pool_size, dropout, device=None, encoder_word_emb=None):
        self.name = 'rnn'

        super(RNNDecoder, self).__init__()

        self.n_layer = n_layer
        self.layer_attn = layer_attn
        self.separate = separate
        self.coverage = coverage
        self.copy = copy
        self.maxout_pool_size = maxout_pool_size
        self.n_vocab_size = n_vocab
        input_size = d_word_vec

        self.input_feed = input_feed
        if input_feed:
            input_size += d_enc_model

        self.ans_emb_weight = encoder_word_emb

        self.answer = answer
        tmp_in = d_word_vec if answer else d_enc_model
        self.decInit = DecInit(d_enc=tmp_in, d_dec=d_model, n_enc_layer=n_enc_layer)

        self.feature = False if not feat_vocab else True
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            # input_size += len(feat_vocab) * d_feat_vec  # PS: only for test !!!
        feat_size = len(feat_vocab) * d_feat_vec if self.feature else 0

        self.d_enc_model = d_enc_model

        self.word_emb_type = ans_n_vocab == n_vocab
        self.word_emb = nn.Embedding(n_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.rnn = StackedRNN(n_layer, input_size, d_model, dropout, rnn=rnn)
        self.attn = ConcatAttention(d_enc_model + feat_size, d_model, d_k, coverage)

        self.readout = nn.Linear((d_word_vec + d_model + self.d_enc_model), d_model)
        self.maxout = MaxOut(maxout_pool_size)

        if copy:
            self.copy_switch = nn.Linear(d_enc_model + d_model, 1)
        
        self.hidden_size = d_model
        self.dropout = nn.Dropout(dropout)
        self.device = device

    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_vocab'], opt['ans_n_vocab'], opt['d_word_vec'], opt['d_model'], opt['n_layer'],
                   opt['rnn'], opt['d_k'], opt['feat_vocab'], opt['d_feat_vec'], 
                   opt['d_enc_model'], opt['n_enc_layer'], opt['input_feed'], opt['copy'], opt['answer'], opt['separate'],
                   opt['coverage'], opt['layer_attn'], opt['maxout_pool_size'], opt['dropout'], 
                   opt['device'], opt['encoder_word_emb'])

    def attn_init(self, context):
        if isinstance(context, list):
            context = context[-1]
        if isinstance(context, tuple):
            context = torch.cat(context, dim=-1)
        batch_size = context.size(0)
        hidden_sizes = (batch_size, self.d_enc_model)
        return Variable(context.data.new(*hidden_sizes).zero_(), requires_grad=False)
    
    def forward(self, inputs, max_length=300, rl_type='', generator=None):
        
        tgt_seq, src_seq, src_indexes = inputs['tgt_seq'], inputs['src_seq'], inputs['src_indexes']
        if self.answer:
            ans_seq = inputs['ans_seq']
        enc_output, hidden, feat_seqs = inputs['enc_output'], inputs['hidden'], inputs['feat_seqs']

        src_pad_mask = Variable(src_seq.data.eq(50256).float(), requires_grad=False, volatile=False)    # TODO: fix this magic number later
        if self.layer_attn:
            n_enc_layer = len(enc_output)
            src_pad_mask = src_pad_mask.repeat(1, n_enc_layer)
            enc_output = torch.cat(enc_output, dim=1)
        
        feat_inputs = None
        if self.feature:
            feat_inputs = [feat_emb(feat_seq) for feat_seq, feat_emb in zip(feat_seqs, self.feat_embs)]
            feat_inputs = torch.cat(feat_inputs, dim=2)
            if self.layer_attn:
                feat_inputs = feat_inputs.repeat(1, n_enc_layer, 1)
            # enc_output = torch.cat((enc_output, feat_inputs), dim=2)    # PS: only for test !!!

        cur_context = self.attn_init(enc_output)

        if self.answer:
            ans_words = torch.sum(F.embedding(ans_seq, self.ans_emb_weight), dim=1)
            hidden = self.decInit(ans_words).unsqueeze(0)
        else:
            hidden = self.decInit(hidden).unsqueeze(0)
        
        self.attn.apply_mask(src_pad_mask)

        if rl_type:
            return self.rl_forward(rl_type, generator, tgt_seq, cur_context, hidden, enc_output, 
                                   feat_inputs, src_indexes)
        else:
            return self.nll_forward(tgt_seq, cur_context, hidden, enc_output, feat_inputs, src_indexes)

    def nll_forward(self, tgt_seq, cur_context, hidden, enc_output, feat_inputs, src_indexes):
        tmp_context, tmp_coverage = None, None
        dec_outputs, coverage_output, copy_output, copy_gate_output = [], [], [], []
        dec_input = self.word_emb(tgt_seq)
        dec_input = dec_input.transpose(0, 1)
        for seq_idx, dec_input_emb in enumerate(dec_input.split(1)):
            dec_input_emb = dec_input_emb.squeeze(0)
            raw_dec_input_emb = dec_input_emb
            if self.input_feed:
                dec_input_emb = torch.cat((dec_input_emb, cur_context), dim=1)
            dec_output, hidden = self.rnn(dec_input_emb, hidden)

            if self.coverage:
                if tmp_coverage is None:
                    tmp_coverage = Variable(torch.zeros((enc_output.size(0), enc_output.size(1))))
                    if self.device:
                        tmp_coverage = tmp_coverage.to(self.device)
                cur_context, attn, tmp_context, next_coverage = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                                          coverage=tmp_coverage, feat_inputs=feat_inputs,
                                                                          feature=self.feature)
                avg_tmp_coverage = tmp_coverage / max(1, seq_idx)
                coverage_loss = torch.sum(torch.min(attn, avg_tmp_coverage), dim=1)
                tmp_coverage = next_coverage
                coverage_output.append(coverage_loss)
            else:
                cur_context, attn, tmp_context = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                           feat_inputs=feat_inputs, feature=self.feature)
              
            if self.copy:
                copy_prob = self.copy_switch(torch.cat((dec_output, cur_context), dim=1))
                copy_prob = torch.sigmoid(copy_prob)

                if self.layer_attn:
                    attn = attn.view(attn.size(0), len(enc_output), -1)
                    attn = attn.sum(1)
                
                if self.separate:
                    out = torch.zeros([len(attn), max_length], device=self.device if self.device else None)
                    for i in range(len(attn)):
                        data_length = src_indexes[i]
                        out[i].narrow(0, 1, data_length - 1).copy_(attn[i][1:src_indexes[i]])
                    attn = out
                    norm_term = attn.sum(1, keepdim=True)
                    attn = attn / norm_term
                
                copy_output.append(attn)
                copy_gate_output.append(copy_prob)
            
            readout = self.readout(torch.cat((raw_dec_input_emb, dec_output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)            
            
            dec_outputs.append(output)
        
        dec_output = torch.stack(dec_outputs).transpose(0, 1)

        rst = {}
        rst['pred'], rst['attn'], rst['context'] = dec_output, attn, cur_context
        if self.copy:
            copy_output = torch.stack(copy_output).transpose(0, 1)
            copy_gate_output = torch.stack(copy_gate_output).transpose(0, 1)
            rst['copy_pred'], rst['copy_gate'] = copy_output, copy_gate_output
        if self.coverage:
            coverage_output = torch.stack(coverage_output).transpose(0, 1)
            rst['coverage_pred'] = coverage_output
        return rst

    def rl_forward(self, rl_type, generator, tgt_seq, cur_context, hidden, enc_output, 
                   feat_inputs, src_indexes):        
        tmp_context, tmp_coverage, seq_idx = None, None, 0
        dec_outputs, coverage_output, copy_output, copy_gate_output = [], [], [], []
        max_length, input_seq = tgt_seq.size(-1), tgt_seq.transpose(0, 1)[0]
        rand_input_seq = input_seq.clone().detach()

        decoded_text, rand_decoded_text = [], []
        init_tokens = torch.zeros(input_seq.size(), device=input_seq.device).long()
        rand_tokens = torch.zeros(input_seq.size(), device=input_seq.device).long()
        rand_choice_list = [0, 102] + [idd for idd in range(1001, self.n_vocab_size)]

        for i in range(max_length):
            decoded_text.append(input_seq.long())
            rand_decoded_text.append(rand_input_seq.long())

            dec_input_emb = self.word_emb(input_seq.long())
            raw_dec_input_emb = dec_input_emb
            if self.input_feed:
                dec_input_emb = torch.cat((dec_input_emb, cur_context), dim=1)
            dec_output, hidden = self.rnn(dec_input_emb, hidden)

            if self.coverage:
                if tmp_coverage is None:
                    tmp_coverage = Variable(torch.zeros((enc_output.size(0), enc_output.size(1))))
                    if self.device:
                        tmp_coverage = tmp_coverage.to(self.device)
                cur_context, attn, tmp_context, next_coverage = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                                          coverage=tmp_coverage, feat_inputs=feat_inputs,
                                                                          feature=self.feature)
                avg_tmp_coverage = tmp_coverage / max(1, seq_idx)
                coverage_loss = torch.sum(torch.min(attn, avg_tmp_coverage), dim=1)
                tmp_coverage = next_coverage
                coverage_output.append(coverage_loss)
            else:
                cur_context, attn, tmp_context = self.attn(dec_output, enc_output, precompute=tmp_context, 
                                                           feat_inputs=feat_inputs, feature=self.feature)
              
            if self.copy:
                copy_prob = self.copy_switch(torch.cat((dec_output, cur_context), dim=1))
                copy_prob = torch.sigmoid(copy_prob)

                if self.layer_attn:
                    attn = attn.view(attn.size(0), len(enc_output), -1)
                    attn = attn.sum(1)
                
                if self.separate:
                    out = torch.zeros([len(attn), max_length], device=self.device if self.device else None)
                    for i in range(len(attn)):
                        data_length = src_indexes[i]
                        out[i].narrow(0, 1, data_length - 1).copy_(attn[i][1:src_indexes[i]])
                    attn = out
                    norm_term = attn.sum(1, keepdim=True)
                    attn = attn / norm_term
                
                copy_output.append(attn)
                copy_gate_output.append(copy_prob)
            
            readout = self.readout(torch.cat((raw_dec_input_emb, dec_output, cur_context), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)            
            dec_outputs.append(output)

            paddings = (input_seq.eq(Constants.PAD).float() + input_seq.eq(102).float()).eq(0).float()  # TODO magic number [SEP]
            rand_paddings = (rand_input_seq.eq(Constants.PAD).float() + rand_input_seq.eq(102).float()).eq(0).float()
            
            ##=== next token predict ===##
            token_dict = F.softmax(generator(output), dim=-1)            
            for b in range(input_seq.size(0)):
                ## sampling strategy 1
                selected_idx = token_dict[b].multinomial(1, replacement=False).view(-1).data[0]
                
                ## sampling strategy 2
                # topk = torch.topk(token_dict[b], k=5, dim=-1)  # TODO magic number
                # selected_idx = topk[1][random.choice(range(5))].data

                init_tokens[b] = selected_idx.item()
                rand_tokens[b] = random.choice(rand_choice_list)

            input_seq = torch.where(paddings > 0, init_tokens, paddings.long())
            rand_input_seq = torch.where(rand_paddings > 0, rand_tokens, rand_paddings.long())
            seq_idx += 1
        
        decoded_text.append(input_seq)
        rand_decoded_text.append(rand_input_seq)

        dec_output = torch.stack(dec_outputs).transpose(0, 1)

        rst = {}
        rst['pred'], rst['attn'], rst['context'] = dec_output, attn, cur_context
        rst['decoded_text'] = torch.stack(decoded_text).transpose(0, 1)
        rst['rand_decoded_text'] = torch.stack(rand_decoded_text).transpose(0, 1)
        if self.copy:
            copy_output = torch.stack(copy_output).transpose(0, 1)
            copy_gate_output = torch.stack(copy_gate_output).transpose(0, 1)
            rst['copy_pred'], rst['copy_gate'] = copy_output, copy_gate_output
        if self.coverage:
            coverage_output = torch.stack(coverage_output).transpose(0, 1)
            rst['coverage_pred'] = coverage_output
        
        return rst
