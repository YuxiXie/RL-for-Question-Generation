import numpy as np

import torch

import onqg.dataset.Constants as Constants
from onqg.utils.mask import get_slf_attn_mask


def collate_fn(insts, sep=False, sep_id=Constants.SEP):
    """get src_seq, src_pos, src_sep (Tensor)"""
    pad_index = Constants.PAD if sep_id < 50000 else sep_id     # PAD tag in GPT2 is > 50000
    batch_seqs = insts    
    if sep:
        insts_tmp = [[w.item() for w in inst] for inst in insts]
        indexes = [inst.index(sep_id) for inst in insts_tmp]
        insts = [(inst[:indexes[i]+1], inst[indexes[i]+1:]) for i, inst in enumerate(insts)]
        insts = [[inst[0] for inst in insts], [inst[1] for inst in insts]]
    else:
        insts = [insts]
    batch_seq = [np.array([np.array(inst.cpu()) for inst in col]) for col in insts]
    
    if sep_id > 50000:     # PAD tag in GPT2 is > 50000
        batch_pos = [[[pos_i+1 if w_i != pad_index else 0 for pos_i,w_i in enumerate(inst + [10])] for inst in col] for col in batch_seq]
    else:
        batch_pos = [[[pos_i+1 if w_i != pad_index else 0 for pos_i,w_i in enumerate(inst)] for inst in col] for col in batch_seq]
    if sep:
        batch_pos = np.array([src + ans for src, ans in zip(batch_pos[0], batch_pos[1])])
    else:
        batch_pos = np.array(batch_pos[0])
    batch_pos = torch.LongTensor(batch_pos)
    
    if sep:
        batch_sep = [[[i for w in inst] for inst in col] for i, col in enumerate(batch_seq)]
        batch_sep = np.array([src + ans for src, ans in zip(batch_sep[0], batch_sep[1])])
        batch_sep = torch.LongTensor(batch_sep)
    
    del batch_seq
    batch_seq = batch_seqs

    if sep:
        return batch_seq, batch_pos, batch_sep
    else:
        return batch_seq, batch_pos

def preprocess_input(raw, device=None, return_sep=False, sep_id=Constants.SEP):
    """get src_seq, src_pos and (src_sep, max_length)"""
    raw = raw.transpose(0, 1)

    if return_sep:
        seq, pos, sep = collate_fn(raw, sep=return_sep, sep_id=sep_id)
        lengths = [[w.item() for w in sent if w.item() == 0] for sent in sep]     # TODO: fix this magic number
        lengths = [len(sent) for sent in lengths]
        max_length = max(lengths)
        if device:
            sep = sep.to(device)
        sep = (sep, max_length)
    else:
        seq, pos = collate_fn(raw, sep=return_sep)
    seq, pos = seq.to(device), pos.to(device)

    if return_sep:
        return seq, pos, sep
    else:
        return seq, pos

def get_sep_index(corpus):
    """get the index of the first [SEP] in each sentence"""
    if corpus is None:
        return None
    else:
        sep = [[w.item() for w in sent] for sent in corpus]
    indexes = [sent.index(1) for sent in sep]     # TODO: fix this magic number
    return indexes

def preprocess_batch(batch, separate=False, enc_rnn=False, dec_rnn=False,
                     feature=False, dec_feature=0, answer=False, ans_feature=False,
                     sep_id=Constants.SEP, copy=False, attn_mask=False, device=None):
    """Get a batch by indexing to the Dataset object, then preprocess it to get inputs for the model
    Input: batch
        raw-index: idxBatch
        src: (wrap(srcBatch), lengths)
        tgt: wrap(tgtBatch)
        copy: (wrap(copySwitchBatch), wrap(copyTgtBatch))
        feat: (tuple(wrap(x) for x in featBatches), lengths)
        ans: (wrap(ansBatch), ansLengths)
        ans_feat: (tuple(wrap(x) for x in ansFeatBatches), ansLengths)
    Output: 
        (1) inputs dict
            encoder: src_seq, lengths for rnn; 
                    src_seq, src_pos for transf; 
                    feat_seqs for both.
                    src_seq, src_sep for bert
            decoder: tgt_seq, src_indexes for rnn; 
                    tgt_seq, tgt_pos for transf; 
                    src_seq, feat_seqs for both.
            answer-encoder: src_seq, lengths, feat_seqs.
        (2) max_length: max_length of source text 
                        (except for answer part) in a batch
        (3) gold
        (4) (copy_gold, copy_switch)
    """
    
    inputs = {'encoder':{}, 'decoder':{}, 'answer-encoder':{}}

    src_seq, tgt_seq = batch['src'], batch['tgt']
    src_seq, lengths = src_seq[0], src_seq[1]        
    src_sep, max_length = None, 0
    if separate:
        #  for sentences contain [SEP] token
        src_seq, src_pos, src_sep = preprocess_input(src_seq, device=device, return_sep=separate, sep_id=sep_id)
        src_sep, max_length = src_sep[0], src_sep[1]
        inputs['encoder']['src_sep'] = src_sep
    else:
        src_seq, src_pos = preprocess_input(src_seq, device=device)
    tgt_seq, tgt_pos = preprocess_input(tgt_seq, device=device)
        
    if enc_rnn:
        inputs['encoder']['src_seq'], inputs['encoder']['lengths'] = src_seq, lengths
    else:
        inputs['encoder']['src_seq'], inputs['encoder']['src_pos'] = src_seq, src_pos
        if attn_mask:
            inputs['encoder']['slf_attn_mask'] = get_slf_attn_mask(attn_mask=batch['attn_mask'], lengths=lengths[0], 
                                                                   device=device)

    gold = tgt_seq[:, 1:]   # exclude [BOS] token
    if not dec_rnn:
        inputs['decoder']['tgt_seq'], inputs['decoder']['tgt_pos'] = tgt_seq[:, :-1], tgt_pos[:, :-1]   # exclude [EOS] token
    else:
        inputs['decoder']['tgt_seq'], inputs['decoder']['src_indexes'] = tgt_seq[:, :-1], get_sep_index(src_sep)
    inputs['decoder']['src_seq'] = inputs['encoder']['src_seq']
    if answer:
        ans_seq, _ = preprocess_input(batch['ans'][0], device=device)
        inputs['decoder']['ans_seq'] = ans_seq

    src_feats, tgt_feats = None, None
    if feature:
        n_all_feature = len(batch['feat'][0])
        # split all features into src and tgt parts, src_feats are those embedded in the encoder
        src_feats = [feat.transpose(0, 1) for feat in batch['feat'][0][:n_all_feature - dec_feature]]
        if dec_feature:
            # dec_feature: the number of features embedded in the decoder
            tgt_feats = [feat.transpose(0, 1) for feat in batch['feat'][0][n_all_feature - dec_feature:]]
    inputs['encoder']['feat_seqs'], inputs['decoder']['feat_seqs'] = src_feats, tgt_feats

    ans_seq = None
    if answer:
        ans_seq = batch['ans']
        ans_seq, ans_lengths = ans_seq[0], ans_seq[1]
        ans_seq, _ = preprocess_input(ans_seq, device=device)
        ans_feats = None
        if ans_feature:
            ans_feats = [feat.transpose(0, 1) for feat in batch['ans_feat'][0]]
        inputs['answer-encoder']['feat_seqs'] = ans_feats
        inputs['answer-encoder']['src_seq'] = ans_seq
        inputs['answer-encoder']['lengths'] = ans_lengths
        
    copy_gold, copy_switch = None, None
    if copy:
        copy_gold, copy_switch = batch['copy'][1], batch['copy'][0]
        copy_gold, _ = preprocess_input(copy_gold, device=device)
        copy_switch, _ = preprocess_input(copy_switch, device=device)
        copy_gold, copy_switch = copy_gold[:, 1:], copy_switch[:, 1:]
        
    return inputs, max_length, gold, (copy_gold, copy_switch)

def preprocess_rl_batch(inputing, inputing_rnd, rl_type, tokenizer, device):
    """Get a batch by indexing to the Dataset object, then preprocess it to get inputs for the model
    """

    def _get_processed_seqs(src_len, src_seq, tgt_txt, tgt_len):
        yes_or_no = torch.tensor([4208, 1185], device=device)
        sep_token = torch.tensor([102], device=device)

        src_seq, pad_seq = src_seq[:src_len], src_seq[src_len:]

        tgt_seq = torch.tensor([x.item() for x in tgt_txt if x.item() != Constants.PAD], device=device)
        tgt_pad_seq_length = tgt_len - tgt_seq.size(0)

        if tgt_pad_seq_length > 0:
            tgt_pad_seq = torch.tensor([Constants.PAD] * tgt_pad_seq_length, device=device)
            pad_seq = torch.cat([pad_seq, tgt_pad_seq], dim=0)
        
        merge_seq = torch.cat([tgt_seq, yes_or_no, src_seq, sep_token, pad_seq], dim=0)
        merge_type = torch.tensor([0] * tgt_seq.size(0) + [1] * (src_seq.size(0) + 3) + [0] * pad_seq.size(0), device=device)
        merge_mask = torch.tensor([1] * (tgt_seq.size(0) + src_seq.size(0) + 3) + [0] * pad_seq.size(0), device=device)
        _len = [tgt_seq.size(0), tgt_seq.size(0) + src_seq.size(0) + 3]

        return merge_seq, merge_type, merge_mask, _len

    if rl_type == 'fluency':
        '''
        input: [batch-size, seq-length]
        '''
        raw_text = inputing.clone().detach().to(device)
        input_text = raw_text[:, :-1].contiguous().to(device)
        input_mask = input_text.ne(Constants.PAD).type(torch.float)
        output_text = raw_text[:, 1:].contiguous().to(device)

        rnd_raw_text = inputing_rnd.clone().detach().to(device)
        rnd_input_text = rnd_raw_text[:, :-1].contiguous().to(device)
        rnd_input_mask = rnd_input_text.ne(Constants.PAD).type(torch.float)
        rnd_output_text = rnd_raw_text[:, 1:].contiguous().to(device)

        return (input_text, input_mask, output_text, rnd_input_text, rnd_input_mask, rnd_output_text)
    
    elif rl_type =='relevance':
        tgt_text = inputing[1].clone().detach().to(device)
        rand_tgt_text = inputing[3].clone().detach().to(device)
        src_text, src_lengths = inputing[0].clone().detach().to(device), inputing[2][0].long()
        batch_size = tgt_text.size(0)

        tgt_text = tgt_text[:, 1:].contiguous()
        rand_tgt_text = rand_tgt_text[:, 1:].contiguous()
        cls_tokens = (torch.ones((batch_size, 1), device=device) * 101).long()
        src_text = torch.cat([cls_tokens, src_text], dim=-1)

        src_seq_length, tgt_seq_length = src_text.size(-1), tgt_text.size(-1)
        merge_seq_length = src_seq_length + 1 + tgt_seq_length

        merge_seqs, merge_seq_types = [], []
        sep_token, pad_token = torch.tensor([102], device=device), torch.tensor([Constants.PAD], device=device)

        for b in range(batch_size):
            _len, src_seq = src_lengths[b].item() + 1, src_text[b]
            src_seq, pad_seq = src_seq[:_len], src_seq[_len:]

            merge_seq = torch.cat([src_seq, sep_token, tgt_text[b], pad_seq], dim=0)
            merge_type = torch.tensor([0] * (_len + 1) + [1] * (merge_seq_length - _len - 1), device=device)
            
            merge_seqs.append(merge_seq)
            merge_seq_types.append(merge_type)
        
        ids = torch.stack(merge_seqs, dim=0).contiguous()
        type_ids = torch.stack(merge_seq_types, dim=0).contiguous()

        return (ids, type_ids)

    elif rl_type == 'answerability':
        tgt_text = inputing[1].clone().detach().to(device)
        
        src_text, src_lengths = inputing[0].clone().detach().to(device), inputing[2][0].long()

        batch_size = tgt_text.size(0)
        tgt_seq_length = tgt_text.size(-1)

        merge_seqs, merge_seq_types, merge_masks, tgt_len = [], [], [], []
        
        for b in range(batch_size):
            merge_seq, merge_type, merge_mask, _len = _get_processed_seqs(src_lengths[b].item(), src_text[b], 
                                                                          tgt_text[b], tgt_seq_length)

            merge_seqs.append(merge_seq)
            merge_seq_types.append(merge_type)
            merge_masks.append(merge_mask)
            tgt_len.append(_len)
        
        ids = torch.stack(merge_seqs, dim=0).contiguous()
        type_ids = torch.stack(merge_seq_types, dim=0).contiguous()
        merge_masks = torch.stack(merge_masks, dim=0).contiguous()

        return (ids, type_ids, merge_masks, tgt_len), []   
