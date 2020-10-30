import torch
from torch import cuda
import torch.nn as nn
import argparse
from tqdm import tqdm

from onqg.utils.translate.Translator import Translator
from onqg.dataset.Dataset import Dataset
from onqg.utils.model_builder import build_model


def dump(data, filename, bleu):
    filename = filename.rstrip('txt').rstrip('.') + '_{bleu:3.5f}_.txt'.format(bleu=bleu * 100)
    golds, preds, paras = data[0], data[1], data[2]
    with open(filename, 'w', encoding='utf-8') as f:
        for g, p, pa in zip(golds, preds, paras):
            pa = [w for w in pa if w not in ['[PAD]', '[CLS]']]
            f.write('<para>\t' + ' '.join(pa) + '\n')
            f.write('<gold>\t' + ' '.join(g[0]) + '\n')
            f.write('<pred>\t' + ' '.join(p) + '\n')
            f.write('===========================\n')


def main(opt):
    device = torch.device('cuda' if opt.cuda else 'cpu')

    checkpoint = torch.load(opt.model)
    model_opt = checkpoint['options']   # torch.load('cased_opt.pt')
    model_opt.gpus = opt.gpus
    model_opt.beam_size, model_opt.batch_size = opt.beam_size, opt.batch_size
    # model_opt.checkpoint_mode = 'all'
    #model_opt.slf_attn_type = 'gated'
    #model_opt.max_token_tgt_len = 50
    #model_opt.proj_share_weight = False

    ### Prepare Data ###
    data = torch.load(opt.data)

    src_vocab, tgt_vocab = data['dict']['src'], data['dict']['tgt']
    # validData = Dataset(data['train'], model_opt.batch_size, copy=model_opt.copy, 
    #                     answer=model_opt.answer == 'enc', ans_feature=model_opt.ans_feature, 
    #                     feature=model_opt.feature, opt_cuda=model_opt.gpus)
    validData = Dataset(data['valid'], model_opt.batch_size, copy=model_opt.copy, 
                        answer=model_opt.answer == 'enc', ans_feature=model_opt.ans_feature, 
                        feature=model_opt.feature, opt_cuda=model_opt.gpus)
    
    ### Prepare Model ###
    model, _ = build_model(model_opt, device)
    model.load_state_dict(checkpoint['model state dict'])
    model.eval()

    translator = Translator(model_opt, tgt_vocab, data['valid']['tokens'], src_vocab)

    bleu, outputs = translator.eval_all(model, validData, output_sent=True)

    print('\nbleu-4', bleu, '\n')

    # dump(outputs, opt.output, bleu)

    # import ipdb; ipdb.set_trace()

    golds, preds, paras = outputs[0], outputs[1], outputs[2]
    golds = [[[w.lower() for w in g[0]]] for g in golds]
    preds = [[w.lower() for w in p] for p in preds]
    from nltk.translate import bleu_score
    bleu = bleu_score.corpus_bleu(golds, preds)
    print('\nbleu-4', bleu, '\n')  

    dump(outputs, opt.output, bleu) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True, help='Path to model .pt file')
    parser.add_argument('-data', required=True, help='Path to data file')
    parser.add_argument('-output', required=True, help='Path to output the predictions')
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-gpus', default=[], nargs='+', type=int)

    opt = parser.parse_args()
    opt.cuda = True if opt.gpus else False
    if opt.cuda:
        cuda.set_device(opt.gpus[0])
    
    main(opt)
