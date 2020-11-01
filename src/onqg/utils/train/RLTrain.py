import os
import time
import math
import random
import logging
from tqdm import tqdm
import numpy as np
import collections

import torch
from torch import cuda
from torch.nn import KLDivLoss
from onqg.utils.train.Loss import NLLLoss

import onqg.dataset.Constants as Constants
from onqg.dataset.data_processor import preprocess_batch, preprocess_rl_batch
from torch.autograd import Variable


def record_log(logfile, step, bad_cnt=-1, lr=-1,
               loss=-1, ppl=-1, accu=-1, bleu=-1,
               rl_loss=-1, rl_accu=-1):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(str(step) + ':\tnll_loss=' + str(round(loss, 8)) + ',\tppl=' + str(round(ppl, 8)))
        f.write(',\trl_loss=' + str(rl_loss) + ',\trl_accu=' + str(round(rl_accu, 8)))
        f.write(',\tbleu=' + str(bleu) + ',\tnll_accu=' + str(round(accu, 8)))
        f.write(',\tbad_cnt=' + str(bad_cnt) + ',\tlr=' + str(lr) + '\n')

def dev_record_log(logfile, step, bad_cnt=-1, lr=-1,
                   loss=-1, ppl=-1, accu=-1, bleu=-1,
                   rl_loss=-1, rl_accu=-1, flu=-1, rel=-1, ans=-1):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(str(step) + ':\tnll_loss=' + str(round(loss, 8)) + ',\tppl=' + str(round(ppl, 8)))
        f.write(',\tbleu=' + str(bleu) + ',\tnll_accu=' + str(round(accu, 8)))
        f.write(',\tbad_cnt=' + str(bad_cnt) + ',\tlr=' + str(lr) + '\n')
        f.write('rl_loss=' + str(rl_loss) + ',\trl_accu=' + str(round(rl_accu, 8)))
        f.write(',\tflu=' + str(flu) + ',\trel=' + str(rel) + ',\tans=' + str(ans) + '\n====================\n')        


class RLTrainer(object):

    def __init__(self, model, discriminator, loss, optimizer, translator, logger, opt, 
                 training_data, validation_data, src_vocab, tgt_vocab):
        self.model = model
        self.discriminator = discriminator
        self.loss = loss
        self.rl_loss = NLLLoss(opt, do_reduce=False)
        self.optimizer = optimizer
        self.translator = translator
        self.logger = logger
        self.opt = opt
        self.tgt_vocab = tgt_vocab

        self.training_data = training_data
        self.validation_data = validation_data

        self.separate = opt.answer == 'sep'
        self.answer = opt.answer == 'enc'
        self.sep_id = src_vocab.lookup(Constants.SEP_WORD) if self.separate else Constants.SEP

        self.is_attn_mask = True if opt.defined_slf_attn_mask else False
        
        self.cntBatch, self.best_metric, self.best_ppl, self.best_bleu = 0, 0, math.exp(100), 0

    def cal_performance(self, loss_input):

        loss = self.loss.cal_loss(loss_input)

        gold, pred = loss_input['gold'], loss_input['pred']

        pred = pred.contiguous().view(-1, pred.size(2))
        pred = pred.max(1)[1]

        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(Constants.PAD)

        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct, pred
    
    def cal_rl_loss(self, pred, decoded_text, 
                    flu_rl_inputs, rel_rl_inputs, ans_rl_inputs, 
                    flu_discriminator, rel_discriminator, ans_discriminator):
        
        def _get_n_best(logits, n_best_size):
            index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

            best_indexes = []
            for i in range(len(index_and_score)):
                if i >= n_best_size:
                    break
                best_indexes.append(index_and_score[i])
            
            return best_indexes
        
        def _get_score(start_logits, end_logits, indexes):
            start_logits = _get_n_best(start_logits[indexes[0]: indexes[1]], 5)
            end_logits = _get_n_best(end_logits[indexes[0]: indexes[1]], 5)
            b_scores = [[0, 0]]

            for start in start_logits:
                for end in end_logits:
                    if start[0] <= end[0] and end[0] - start[0] < 64:   # TODO: magic number
                        score = torch.tensor([start[1], end[1]])
                        score = math.pow(score[0].item() * score[1].item(), 0.5)
                        b_scores.append([score, end[0] - start[0] + 1])
            b_scores.sort(key=lambda x: x[0], reverse=True)
            return b_scores[0][0]

        batch_size, seq_length, vocab_size = pred.size()
        gold = decoded_text[:, 1:].contiguous()

        ##=== fluency ===##
        flu_reward = 0
        if 'fluency' in self.opt.rl:
            with torch.no_grad():
                output_pred_dicts = flu_discriminator(flu_rl_inputs[0], attention_mask=flu_rl_inputs[1])

                reward_fct = NLLLoss(self.opt, do_reduce=False)

                lm_loss = reward_fct.cal_simple_nll(output_pred_dicts[0], flu_rl_inputs[2])
                lm_loss = lm_loss.view(pred.size(0), -1).mean(-1)
                scores = torch.exp(lm_loss).to(pred.device)
            
            flu_reward = scores.data.sum().item()
            flu_scores_scale = self.opt.flu_alpha - scores.data

        ##=== relevance ===##
        rel_reward = 0
        if 'relevance' in self.opt.rl:
            with torch.no_grad():
                output = rel_discriminator(rel_rl_inputs[0], token_type_ids=rel_rl_inputs[1])
                # get the output logits for [CLS]
                logits = output[0].contiguous().to(pred.device)
            scores = torch.softmax(logits, dim=1).transpose(0, 1)[1].contiguous()

            rel_reward = scores.data.sum().item()
            rel_scores_scale = torch.log(self.opt.rel_alpha / (1 - scores.data + 1e-16))

        ##=== answerability ===##
        ans_reward = 0
        if 'answerability' in self.opt.rl:
            with torch.no_grad():
                batch_start_logits, batch_end_logits = ans_discriminator(ans_rl_inputs[0], ans_rl_inputs[1], ans_rl_inputs[2])
            scores, rand_scores = [], []
            for b in range(batch_start_logits.size(0)):
                start_logits = torch.softmax(batch_start_logits[b], dim=-1)
                end_logits = torch.softmax(batch_end_logits[b], dim=-1)
                score = _get_score(start_logits.detach().cpu().tolist(), 
                                end_logits.detach().cpu().tolist(),
                                ans_rl_inputs[3][b])
                scores.append(score)
            scores = torch.tensor(scores, device=pred.device)

            ans_reward = scores.data.sum().item()
            ans_scores_scale = torch.log(self.opt.ans_alpha / (1 - scores.data + 1e-16))

        ##=== combination ===##
        scores = 0
        if 'fluency' in self.opt.rl:
            scores += flu_scores_scale  * self.opt.flu_gamma
        if 'relevance' in self.opt.rl:
            scores += rel_scores_scale * self.opt.rel_gamma
        if 'answerability' in self.opt.rl:
            scores += ans_scores_scale * self.opt.ans_gamma
                
        n_correct = scores.gt(0).float().sum().item()
        weights = [(batch_size - n_correct) / batch_size, n_correct / batch_size]
        weights = [1/3, 2/3] if weights[0] > 1/3 else weights
        scores_scale_rgt = scores.gt(0).float() * scores * weights[1]
        scores_scale_wrg = scores.lt(0).float() * scores * weights[0]
        scores_scale = scores_scale_rgt + scores_scale_wrg

        log_prb = self.rl_loss.cal_simple_nll(pred.contiguous(), gold.contiguous()).view(batch_size, -1).mean(-1)   
        loss = torch.sum(scores_scale * log_prb)

        return loss, [n_correct, batch_size, [flu_reward, rel_reward, ans_reward]]

    def save_model(self, better, bleu):
        model_state_dict = self.model.module.state_dict() if len(self.opt.gpus) > 1 else self.model.state_dict()
        model_state_dict = collections.OrderedDict([(x,y.cpu()) for x,y in model_state_dict.items()])
        checkpoint = {'model state dict': model_state_dict, 'options': self.opt}

        if self.opt.save_mode == 'all':
            model_name = self.opt.save_model + '_ppl_{ppl:2.5f}.chkpt'.format(ppl=self.best_ppl)
            torch.save(checkpoint, model_name)
        elif self.opt.save_mode == 'best':
            model_name = self.opt.save_model + '.chkpt'
            if better:
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')
        
        if bleu != 'unk' and bleu > self.best_bleu:
            self.best_bleu = bleu
            model_name = self.opt.save_model + '_' + str(round(bleu * 100, 5)) + '_bleu4.chkpt'
            torch.save(checkpoint, model_name)

    def eval_step(self, device, epoch):
        ''' Epoch operation in evaluation phase '''
        self.model.eval()      

        with torch.no_grad():
            max_length = 0
            total_nll_loss, n_word_total, n_word_correct = 0, 0, 0
            total_rl_loss, n_rl_correct = 0, [0, 0]
            flu_reward, rel_reward, ans_reward = 0, 0, 0
            valid_length = len(self.validation_data)
            eval_index_list = range(valid_length)
            
            for idx in tqdm(eval_index_list, mininterval=2, desc='  - (Validation) ', leave=False):
                batch = self.validation_data[idx]
                inputs, max_length, gold, copy = preprocess_batch(batch, separate=self.separate, enc_rnn=self.opt.enc_rnn != '', 
                                                                  dec_rnn=self.opt.dec_rnn != '', feature=self.opt.feature, 
                                                                  dec_feature=self.opt.dec_feature, answer=self.answer, 
                                                                  ans_feature=self.opt.ans_feature, sep_id=self.sep_id, copy=self.opt.copy, 
                                                                  attn_mask=self.is_attn_mask, device=device)
                copy_gold, copy_switch = copy[0], copy[1]

                ### forward ###
                rst = self.model(inputs, max_length=max_length)

                loss_input = {}
                loss_input['pred'], loss_input['gold'] = rst['pred'], gold
                if self.opt.copy:
                    loss_input['copy_pred'], loss_input['copy_gate'] = rst['copy_pred'], rst['copy_gate']
                    loss_input['copy_gold'], loss_input['copy_switch'] = copy_gold, copy_switch
                if self.opt.coverage:
                    loss_input['coverage_pred'] = rst['coverage_pred']
                nll_loss, n_correct, _ = self.cal_performance(loss_input)

                rst = self.model(inputs, max_length=max_length, rl_type=self.opt.rl)

                flu_rl_inputs, flu_discriminator = None, None
                if 'fluency' in self.opt.rl:
                    flu_rl_inputs = preprocess_rl_batch(rst['decoded_text'], rst['rand_decoded_text'], 'fluency', 
                                                        self.tgt_vocab, self.opt.rl_device['fluency'])
                    flu_discriminator = self.discriminator['fluency']
                    flu_discriminator.eval()
                
                rel_rl_inputs, rel_discriminator = None, None
                if 'relevance' in self.opt.rl:
                    inputing = (inputs['encoder']['src_seq'], rst['decoded_text'], inputs['encoder']['lengths'], rst['rand_decoded_text'])
                    rel_rl_inputs = preprocess_rl_batch(inputing, None, 'relevance', self.tgt_vocab, self.opt.rl_device['relevance'])
                    rel_discriminator = self.discriminator['relevance']
                    rel_discriminator.eval()
                
                ans_rl_inputs, ans_discriminator = None, None
                if 'answerability' in self.opt.rl:
                    inputing = (inputs['encoder']['src_seq'], rst['decoded_text'], inputs['encoder']['lengths'], rst['rand_decoded_text'])
                    ans_rl_inputs, rand_rl_inputs = preprocess_rl_batch(inputing, None, 'answerability', self.tgt_vocab, self.opt.rl_device['answerability'])
                    ans_discriminator = self.discriminator['answerability']
                    ans_discriminator.eval()
                
                if self.opt.rl:
                    rl_loss, rl_n_correct = self.cal_rl_loss(rst['pred'], rst['decoded_text'], 
                                                             flu_rl_inputs, rel_rl_inputs, ans_rl_inputs, 
                                                             flu_discriminator, rel_discriminator, ans_discriminator)
                
                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()

                total_nll_loss += nll_loss.item()
                n_word_total += n_word
                n_word_correct += n_correct

                total_rl_loss += rl_loss.item()
                n_rl_correct[0] += rl_n_correct[0]
                n_rl_correct[1] += rl_n_correct[1]

                flu_reward += rl_n_correct[2][0]
                rel_reward += rl_n_correct[2][1]
                ans_reward += rl_n_correct[2][2]
        
            loss_per_word = total_nll_loss / n_word_total
            nll_accuracy = n_word_correct / n_word_total
            loss_per_sample = total_rl_loss / n_rl_correct[1]
            bleu = 'unk' 
            perplexity = math.exp(min(loss_per_word, 16))
            rl_accuracy = n_rl_correct[0] / n_rl_correct[1]
            
            flu_reward /= n_rl_correct[1]
            rel_reward /= n_rl_correct[1]
            ans_reward /= n_rl_correct[1]

            if (perplexity <= self.opt.translate_ppl or perplexity > self.best_ppl):
                if self.cntBatch % self.opt.translate_steps == 0: 
                    bleu = self.translator.eval_all(self.model, self.validation_data)

        return [loss_per_word, loss_per_sample], nll_accuracy, rl_accuracy, bleu, [flu_reward, rel_reward, ans_reward]
   
    def train_epoch(self, device, epoch):
        ''' Epoch operation in training phase'''
        if self.opt.extra_shuffle and epoch > self.opt.curriculum:
            self.logger.info('Shuffling...')
            self.training_data.shuffle()

        self.model.train()

        total_rl_loss, total_nll_loss, n_sample_total, n_sample_correct = 0, 0, 0, 0
        n_word_total, n_word_correct, report_n_word_total, report_n_word_correct = 0, 0, 0, 0
        report_total_rl_loss, report_total_nll_loss, report_n_sample_total, report_n_sample_correct = 0, 0, 0, 0

        batch_order = torch.randperm(len(self.training_data))

        for idx in tqdm(range(len(self.training_data)), mininterval=2, desc='  - (Training)   ', leave=False):

            batch_idx = batch_order[idx] if epoch > self.opt.curriculum else idx
            batch = self.training_data[batch_idx]

            ##### ==================== prepare data ==================== #####
            inputs, max_length, gold, copy = preprocess_batch(batch, separate=self.separate, enc_rnn=self.opt.enc_rnn != '', 
                                                              dec_rnn=self.opt.dec_rnn != '', feature=self.opt.feature, 
                                                              dec_feature=self.opt.dec_feature, answer=self.answer, 
                                                              ans_feature=self.opt.ans_feature, sep_id=self.sep_id, copy=self.opt.copy, 
                                                              attn_mask=self.is_attn_mask, device=device)
            copy_gold, copy_switch = copy[0], copy[1]
                
            ##### ==================== forward ==================== #####
            self.model.zero_grad()
            self.optimizer.zero_grad()
            
            rst = self.model(inputs, max_length=max_length, rl_type=self.opt.rl)

            ##### ==================== backward ==================== #####            
            ##=== rl loss ===##
            flu_rl_inputs, flu_discriminator = None, None
            if 'fluency' in self.opt.rl:
                flu_rl_inputs = preprocess_rl_batch(rst['decoded_text'], rst['rand_decoded_text'], 'fluency', 
                                                    self.tgt_vocab, self.opt.rl_device['fluency'])
                flu_discriminator = self.discriminator['fluency']
                flu_discriminator.eval()
            
            rel_rl_inputs, rel_discriminator = None, None
            if 'relevance' in self.opt.rl:
                inputing = (inputs['encoder']['src_seq'], rst['decoded_text'], inputs['encoder']['lengths'], rst['rand_decoded_text'])
                rel_rl_inputs = preprocess_rl_batch(inputing, None, 'relevance', self.tgt_vocab, self.opt.rl_device['relevance'])
                rel_discriminator = self.discriminator['relevance']
                rel_discriminator.eval()
            
            ans_rl_inputs, ans_discriminator = None, None
            if 'answerability' in self.opt.rl:
                inputing = (inputs['encoder']['src_seq'], rst['decoded_text'], inputs['encoder']['lengths'], rst['rand_decoded_text'])
                ans_rl_inputs, rand_rl_inputs = preprocess_rl_batch(inputing, None, 'answerability', self.tgt_vocab, self.opt.rl_device['answerability'])
                ans_discriminator = self.discriminator['answerability']
                ans_discriminator.eval()

            if self.opt.rl:
                rl_loss, n_correct = self.cal_rl_loss(rst['pred'], rst['decoded_text'], 
                                                      flu_rl_inputs, rel_rl_inputs, ans_rl_inputs, 
                                                      flu_discriminator, rel_discriminator, ans_discriminator)
            
            if len(self.opt.gpus) > 1:
                rl_loss = rl_loss.mean()  # mean() to average on multi-gpu.
            loss = rl_loss

            ###=== NLL loss ===##
            rst = self.model(inputs, max_length=max_length)

            loss_input = {}
            loss_input['pred'], loss_input['gold'] = rst['pred'], gold
            if self.opt.copy:
                loss_input['copy_pred'], loss_input['copy_gate'] = rst['copy_pred'], rst['copy_gate']
                loss_input['copy_gold'], loss_input['copy_switch'] = copy_gold, copy_switch
            if self.opt.coverage:
                loss_input['coverage_pred'] = rst['coverage_pred']

            nll_loss, word_correct, _ = self.cal_performance(loss_input)
            if len(self.opt.gpus) > 1:
                nll_loss = nll_loss.mean()  # mean() to average on multi-gpu.

            self.cntBatch += 1
            if self.cntBatch % 4 == 0 or loss.item() < -10:
                loss = loss + nll_loss

            ##=== backward ===##
            if math.isnan(loss):
                print('loss catch NaN')
                import ipdb; ipdb.set_trace()

            self.optimizer.backward(loss)
            self.optimizer.step()

            if math.isnan(self.model.generator.weight.data.contiguous().view(-1).sum().item()):
                print('parameter catch NaN')
                import ipdb; ipdb.set_trace()

            ##### ==================== note for epoch report & step report ==================== #####
            n_word = gold.ne(Constants.PAD).float().sum().item()
            total_nll_loss += nll_loss.item()
            n_word_total += n_word
            n_word_correct += word_correct

            total_rl_loss += rl_loss.item()
            n_sample_total += n_correct[1]
            n_sample_correct += n_correct[0]

            report_total_nll_loss += nll_loss.item()
            report_n_word_total += n_word
            report_n_word_correct += word_correct

            report_total_rl_loss += rl_loss.item()            
            report_n_sample_total += n_correct[1]
            report_n_sample_correct += n_correct[0]

            ##### ==================== evaluation ==================== #####
            if self.cntBatch % self.opt.valid_steps == 0:                
                ### ========== evaluation on dev ========== ###
                valid_loss, valid_nll_accu, valid_rl_accu, valid_bleu, rewards = self.eval_step(device, epoch)
                valid_ppl = math.exp(min(valid_loss[0], 16))

                report_avg_nll_loss = report_total_nll_loss / report_n_word_total
                report_avg_rl_loss = report_total_rl_loss / report_n_sample_total
                report_avg_ppl = math.exp(min(report_avg_nll_loss, 16))
                report_avg_nll_accu = report_n_word_correct / report_n_word_total
                report_avg_rl_accu = report_n_sample_correct / report_n_sample_total

                better = False
                # metric = (valid_rl_accu if self.opt.rl in ['relevance', 'answerability', ''] else 1 / (valid_loss[1] + 1e-16)) / valid_ppl
                metric = valid_rl_accu / valid_ppl
                if metric >= self.best_metric:
                    self.best_metric = metric
                    better = True

                report_total_nll_loss, report_total_rl_loss = 0, 0
                report_n_word_total, report_n_sample_total = 0, 0
                report_n_word_correct, report_n_sample_correct = 0, 0

                ### ========== update learning rate ========== ###
                self.optimizer.update_learning_rate(better)

                record_log(self.opt.logfile_train, step=self.cntBatch, 
                           rl_loss=report_avg_rl_loss, rl_accu=report_avg_rl_accu, 
                           loss=report_avg_nll_loss, accu=report_avg_nll_accu, ppl=math.exp(min(report_avg_nll_loss, 16)),
                           bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate)
                dev_record_log(self.opt.logfile_dev, step=self.cntBatch, 
                               loss=valid_loss[0], accu=valid_nll_accu, ppl=valid_ppl, bleu=valid_bleu, 
                               rl_loss=valid_loss[1], rl_accu=valid_rl_accu,
                               bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate,
                               flu=rewards[0], rel=rewards[1], ans=rewards[2])

                if self.opt.save_model:
                    self.save_model(better, valid_bleu)

                self.model.train()

        loss_per_word = total_nll_loss / n_word_total
        nll_accuracy = n_word_correct / n_word_total * 100
        rl_accuracy = n_sample_correct / n_sample_total * 100

        return math.exp(min(loss_per_word, 16)), nll_accuracy, rl_accuracy

    def train(self, device):   
        ''' Start training '''
        self.logger.info(self.model)

        for epoch_i in range(self.opt.epoch):
            self.logger.info('')
            self.logger.info(' *  [ Epoch {0} ]:   '.format(epoch_i))
            start = time.time()
            ppl, nll_accu, rl_accu = self.train_epoch(device, epoch_i + 1) 

            self.logger.info(
                ' *  - (Training)   ppl: {ppl: 8.5f}, accuracy: nll - {nll:3.3f} %; rl - {rl:3.3f} %'.format(ppl=ppl, nll=nll_accu, rl=rl_accu)
            )
            print('                ' + str(time.time() - start) + ' seconds for epoch ' + str(epoch_i))
        
