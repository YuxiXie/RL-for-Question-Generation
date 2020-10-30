import torch
import torch.nn as nn
from torch.autograd import Variable


import onqg.dataset.Constants as Constants


class OpenNQG(nn.Module):
    '''
    A seq2seq-based Question Generation model 
    utilize structures of both RNN and Transformer

    Input: (1) src_seq: rnn_enc——src_seq,lengths; transf_enc——src_seq,src_pos 
           (2) tgt_seq: rnn_dec——tgt_seq; transf_dec——tgt_seq,tgt_pos
           (3) src_sep
           (4) feat_seqs: list of feat_seq
           (5) ans_seq: (ans_seq,ans_feat_seqs)/ans_seq,ans_lengths
           (6) max_length: max length of sentences in src (answer not included) ——> for DataParallel Class

    Output: results output from the Decoder (type: dict)
    '''
    def __init__(self, encoder, decoder, answer_encoder=None):
        super(OpenNQG, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        # self.answer = True if answer_encoder else False
        # if self.answer:
        #     self.answer_encoder = answer_encoder
        
        self.encoder_type = self.encoder.name
        self.decoder_type = self.decoder.name

    def forward(self, inputs, max_length=0, rl_type=''):
        #========== forward ==========#
        enc_output, hidden = self.encoder(inputs['encoder'])
        # if self.answer:
        #     _, hidden = self.answer_encoder(inputs['answer-encoder'])
        inputs['decoder']['enc_output'], inputs['decoder']['hidden'] = enc_output, hidden
        dec_output = self.decoder(inputs['decoder'], max_length=max_length, rl_type=rl_type, 
                                  generator=self.generator)
        #========== generate =========#
        dec_output['pred'] = self.generator(dec_output['pred'])

        return dec_output
