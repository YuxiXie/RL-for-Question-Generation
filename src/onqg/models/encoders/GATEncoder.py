import torch
import torch.nn as nn

import onqg.dataset.Constants as Constants

from onqg.models.modules.Layers import GATEncoderLayer


class GATEncoder(nn.Module):
    """Combine GGNN (Gated Graph Neural Network) and GAT (Graph Attention Network)
    Input: (1) nodes - [batch_size, node_num, d_model]
           (2) edges - ([batch_size, node_num * node_num], [batch_size, node_num * node_num]) 1st-inlink, 2nd-outlink
           (3) mask - ([batch_size, node_num, node_num], [batch_size, node_num, node_num]) 1st-inlink, 2nd-outlink
           (4) node_feats - list of [batch_size, node_num]
    """
    def __init__(self, n_edge_type, d_model, n_layer, alpha, d_feat_vec,
                 feat_vocab, layer_attn, dropout, attn_dropout):
        self.name = 'graph'
        super(GATEncoder, self).__init__()
        self.layer_attn = layer_attn

        self.hidden_size = d_model
        self.d_model = d_model
        ###=== node features ===###
        self.feature = True if feat_vocab else False
        if self.feature:
            self.feat_embs = nn.ModuleList([
                nn.Embedding(n_f_vocab, d_feat_vec, padding_idx=Constants.PAD) for n_f_vocab in feat_vocab
            ])
            self.feature_transform = nn.Linear(self.hidden_size + d_feat_vec * len(feat_vocab), self.hidden_size)
        ###=== graph encode layers ===###
        self.layer_stack = nn.ModuleList([
            GATEncoderLayer(self.hidden_size, d_model, alpha, feature=self.feature,
                            dropout=dropout, attn_dropout=attn_dropout) for _ in range(n_layer)
        ])
        ###=== gated output ===###
        self.gate = nn.Linear(2 * d_model, d_model, bias=False)

    @classmethod
    def from_opt(cls, opt):
        return cls(opt['n_edge_type'], opt['d_model'], opt['n_layer'], opt['alpha'], 
                   opt['d_feat_vec'], opt['feat_vocab'], opt['layer_attn'], 
                   opt['dropout'], opt['attn_dropout'])

    def forward(self, inputs):
        nodes, mask = inputs['nodes'], inputs['mask']
        node_feats, node_type = inputs['feat_seqs'], inputs['type']
        nodes = self.activate(nodes)
        node_output = nodes    # batch_size x node_num x d_model
        ###=== get embeddings ===###
        feat_hidden = None
        if self.feature:
            feat_hidden = [feat_emb(node_feat) for node_feat, feat_emb in zip(node_feats, self.feat_embs)]
            feat_hidden = torch.cat(feat_hidden, dim=2)     # batch_size x node_num x (hidden_size - d_model)
            node_output = self.feature_transform(torch.cat((node_output, feat_hidden), dim=-1))
        # batch_size x (node_num * node_num) x hidden_size x d_model
        ##=== forward ===###
        node_outputs = []
        for enc_layer in self.layer_stack:
            node_output = enc_layer(node_output, mask, node_type, feat_hidden=feat_hidden)
            node_outputs.append(node_output)
            
        node_outputs[-1] = node_output

        hidden = [layer_output.transpose(0, 1)[0] for layer_output in node_outputs]
        
        if self.layer_attn:
            node_output = node_outputs
        
        return node_output, hidden

