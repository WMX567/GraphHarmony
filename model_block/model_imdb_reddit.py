from .layers import *
from .utils import *

class GNN_Encoder_noise(nn.Module):
    def __init__(self, in_dim, hs, dp, backbone):
        super(GNN_Encoder_noise, self).__init__()
        self.backbone = backbone
        if backbone == 'gcn':
            self.gnn0 = BatchGraphConvolution(in_dim, hs)
            self.gnn1 = BatchGraphConvolution(hs, hs)
            self.gnn2 = BatchGraphConvolution(hs, hs)
        elif backbone == 'gat':
            self.gnn0 = BatchMultiHeadGraphAttention(1, in_dim, hs, 0.2)
            self.gnn1 = BatchMultiHeadGraphAttention(1, hs, hs, 0.2)
            self.gnn2 = BatchMultiHeadGraphAttention(1, hs, hs*2, 0.2)
        else:
            raise NotImplementedError
        self.dropout = nn.Dropout(dp)
        self.act = nn.ReLU()
    
    def repara(self, mu, lv):
        if self.training:
            eps = torch.randn_like(lv)
            std = torch.exp(lv)
            return mu + eps * std
        else:
            return mu

    def forward(self, x, adj):
        res = dict()
        adj = add_self_loop(adj)
        h = self.dropout(self.act(self.gnn0(x, adj)))
        h = self.dropout(self.act(self.gnn1(h, adj)))
        y = self.gnn2(h, adj)
        res['ymu'], res['ylv'] = y.chunk(chunks=2, dim=-1)
        res['y'] = self.repara(res['ymu'], res['ylv'])
        return res


class GraphDecoderOur(nn.Module):
    def __init__(self, dec_hs, dim_y, droprate):
        super(GraphDecoderOur, self).__init__()
        self.y_lin0 = nn.Linear(dim_y, dim_y)
        self.dym_lin1 = nn.Linear(dim_y, dec_hs)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, y):
        y = self.dropout(self.act(self.y_lin0(y)))
        dym = self.dym_lin1(y)
        adj_recons = torch.bmm(dym, dym.permute(0, 2, 1))
        return adj_recons


class ClassClassifier(nn.Module):
    def __init__(self, hs, n_class, droprate):
        super(ClassClassifier, self).__init__()
        self.lin0 = nn.Linear(hs, hs)
        self.lin1 = nn.Linear(hs, n_class)
        self.dropout = nn.Dropout(droprate)
        self.act = nn.ReLU()

    def forward(self, x, not_skip=False):
        h = self.dropout(self.act(self.lin0(x)))
        logits = self.lin1(h)
        if not_skip:
            return logits
        return logits[:, -1, :]  


class Our_Base_noise(nn.Module):
    def __init__(self, in_dim, hs, m_dim, dp,source_vertex_feats,
                 target_vertex_feats, backbone):
        super(Our_Base_noise, self).__init__()

        self.svf = nn.Embedding(source_vertex_feats.size(0), source_vertex_feats.size(1))
        self.svf.weight = Parameter(source_vertex_feats, requires_grad = False)
        self.tvf = nn.Embedding(target_vertex_feats.size(0), target_vertex_feats.size(1))
        self.tvf.weight = Parameter(target_vertex_feats, requires_grad = False)
        in_dim = int(source_vertex_feats.size(1))

        self.encoder = GNN_Encoder_noise(in_dim, hs, dp, backbone)
        self.classClassifier = ClassClassifier(hs, 2, dp)
        self.graph_decoder = GraphDecoderOur(m_dim, hs, dp)

    def forward(self, x, vts, adj, domain):
        if domain == 0:
            x = self.svf(vts)
        else:
            x = self.tvf(vts)

        x = F.instance_norm(x)
        res = self.encoder(x, adj)
        res['cls_output'] = self.classClassifier(res['y'])
        res['emb'] = res['y'][:, -1, :]
        res['a_recons'] = self.graph_decoder(res['y'])
        return res

