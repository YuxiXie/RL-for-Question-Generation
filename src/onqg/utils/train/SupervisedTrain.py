import os
import time
import math
import logging
from tqdm import tqdm
import numpy as np
import collections

import torch
from torch import cuda

import onqg.dataset.Constants as Constants
from onqg.dataset.data_processor import preprocess_batch


def record_log(logfile, step, loss, ppl, accu, bleu='unk', bad_cnt=0, lr='unk'):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(str(step) + ':\tloss=' + str(round(loss, 8)) + ',\tppl=' + str(round(ppl, 8)))
        f.write(',\tbleu=' + str(bleu) + ',\taccu=' + str(round(accu, 8)))
        f.write(',\tbad_cnt=' + str(bad_cnt) + ',\tlr=' + str(lr) + '\n')


class SupervisedTrainer(object):

    def __init__(self, model, loss, optimizer, translator, logger, opt, 
                 training_data, validation_data, src_vocab):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.translator = translator
        self.logger = logger
        self.opt = opt

        self.training_data = training_data
        self.validation_data = validation_data

        self.separate = opt.answer == 'sep'
        self.answer = opt.answer == 'enc'
        if opt.pretrained.count('gpt2'):
            self.sep_id = src_vocab.lookup('<|endoftext|>')
        else:
            self.sep_id = src_vocab.lookup(Constants.SEP_WORD) if self.separate else Constants.SEP

        self.is_attn_mask = True if opt.defined_slf_attn_mask else False
        
        self.cntBatch, self.best_ppl, self.best_bleu = 0, math.exp(100), 0

    def cal_performance(self, loss_input):

        loss = self.loss.cal_loss(loss_input)

        gold, pred = loss_input['gold'], loss_input['pred']

        pred = pred.contiguous().view(-1, pred.size(2))
        pred = pred.max(1)[1]

        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(Constants.PAD)

        n_correct = pred.eq(gold)
        n_correct = n_correct.masked_select(non_pad_mask).sum().item()

        return loss, n_correct

    def save_model(self, better, bleu):
        model_state_dict = self.model.module.state_dict() if len(self.opt.gpus) > 1 else self.model.state_dict()
        model_state_dict = collections.OrderedDict([(x,y.cpu()) for x,y in model_state_dict.items()])
        checkpoint = model_state_dict
        # {
        #     'model': model_state_dict,
        #     'settings': self.opt,
        #     'step': self.cntBatch}

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
            total_loss, n_word_total, n_word_correct = 0, 0, 0
            for idx in tqdm(range(len(self.validation_data)), mininterval=2, desc='  - (Validation) ', leave=False):
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
                loss, n_correct = self.cal_performance(loss_input)

                non_pad_mask = gold.ne(Constants.PAD)
                n_word = non_pad_mask.sum().item()

                total_loss += loss.item()
                n_word_total += n_word
                n_word_correct += n_correct
        
            loss_per_word = total_loss / n_word_total
            accuracy = n_word_correct / n_word_total
            bleu = 'unk'
            perplexity = math.exp(min(loss_per_word, 16))

            if (perplexity <= self.opt.translate_ppl or perplexity > self.best_ppl):
                if self.cntBatch % self.opt.translate_steps == 0: 
                    bleu = self.translator.eval_all(self.model, self.validation_data)

        return loss_per_word, accuracy, bleu

    def train_epoch(self, device, epoch):
        ''' Epoch operation in training phase'''
        if self.opt.extra_shuffle and epoch > self.opt.curriculum:
            self.logger.info('Shuffling...')
            self.training_data.shuffle()

        self.model.train()

        total_loss, n_word_total, n_word_correct = 0, 0, 0
        report_total_loss, report_n_word_total, report_n_word_correct = 0, 0, 0

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
            
            rst = self.model(inputs, max_length=max_length)

            ##### ==================== backward ==================== #####
            loss_input = {}
            loss_input['pred'], loss_input['gold'] = rst['pred'], gold
            if self.opt.copy:
                loss_input['copy_pred'], loss_input['copy_gate'] = rst['copy_pred'], rst['copy_gate']
                loss_input['copy_gold'], loss_input['copy_switch'] = copy_gold, copy_switch
            if self.opt.coverage:
                loss_input['coverage_pred'] = rst['coverage_pred']

            loss, n_correct = self.cal_performance(loss_input)
            if len(self.opt.gpus) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            if math.isnan(loss.item()) or loss.item() > 1e20:
                print('catch NaN')
                import ipdb; ipdb.set_trace()

            self.optimizer.backward(loss)
            self.optimizer.step()

            ##### ==================== note for epoch report & step report ==================== #####
            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()

            total_loss += loss.item()
            n_word_total += n_word
            n_word_correct += n_correct
            report_total_loss += loss.item()
            report_n_word_total += n_word
            report_n_word_correct += n_correct
            ##### ==================== evaluation ==================== #####
            self.cntBatch += 1
            if self.cntBatch % self.opt.valid_steps == 0:                
                ### ========== evaluation on dev ========== ###
                valid_loss, valid_accu, valid_bleu = self.eval_step(device, epoch)
                valid_ppl = math.exp(min(valid_loss, 16))

                report_avg_loss = report_total_loss / report_n_word_total
                report_avg_ppl = math.exp(min(report_avg_loss, 16))
                report_avg_accu = report_n_word_correct / report_n_word_total
                
                better = False
                if valid_ppl <= self.best_ppl:
                    self.best_ppl = valid_ppl
                    better = True

                report_total_loss, report_n_word_total, report_n_word_correct = 0, 0, 0
                
                ### ========== update learning rate ========== ###
                self.optimizer.update_learning_rate(better)

                record_log(self.opt.logfile_train, step=self.cntBatch, loss=report_avg_loss, ppl=report_avg_ppl, 
                           accu=report_avg_accu, bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate)
                record_log(self.opt.logfile_dev, step=self.cntBatch, loss=valid_loss, ppl=math.exp(min(valid_loss, 16)), 
                           accu=valid_accu, bleu=valid_bleu, bad_cnt=self.optimizer._bad_cnt, lr=self.optimizer._learning_rate)

                if self.opt.save_model:
                    self.save_model(better, valid_bleu)

                self.model.train()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total

        return math.exp(min(loss_per_word, 16)), 100*accuracy

    def train(self, device):
        ''' Start training '''
        self.logger.info(self.model)

        for epoch_i in range(self.opt.epoch):
            self.logger.info('')
            self.logger.info(' *  [ Epoch {0} ]:   '.format(epoch_i))
            start = time.time()
            ppl, accu = self.train_epoch(device, epoch_i + 1) 

            self.logger.info(' *  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %'.format(ppl=ppl, accu=accu))
            print('                ' + str(time.time() - start) + ' seconds for epoch ' + str(epoch_i))
        