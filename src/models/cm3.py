import os
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss
from utils.utils import build_sim, compute_normalized_laplacian


class CM3(GeneralRecommender):
    def __init__(self, config, dataset):
        super(CM3, self).__init__(config, dataset)

        self.gamma = config['gamma']
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        
        self.max_sim = config['max_sim']
        self.min_sim = config['min_sim']
        self.alpha = config['alpha']

        self.cur_epoch = 0

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.theta = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_users, 3, 1), dtype=torch.float32, requires_grad=True)))

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path,
                                   'mm_adj_cm3_{}_{}.pt'.format(self.knn_k, int(10 * self.mm_image_weight)))

        v_feat_dim, t_feat_dim = 0, 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            v_feat_dim = self.v_feat.shape[1]

            self.v_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
                self.device))
            self.v_MLP = nn.Linear(v_feat_dim, 4 * self.embedding_dim)
            self.v_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            t_feat_dim = self.t_feat.shape[1]
            self.t_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
                self.device))
            self.t_MLP = nn.Linear(t_feat_dim, 4 * self.embedding_dim)
            self.t_MLP_1 = nn.Linear(4 * self.embedding_dim, self.embedding_dim, bias=False)

        self.id_preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(self.n_users, self.embedding_dim), dtype=torch.float32, requires_grad=True), gain=1).to(
            self.device))

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        self.text_nn = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.image_nn = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        nn.init.xavier_normal_(self.text_nn.weight)
        nn.init.xavier_normal_(self.image_nn.weight)

        self.result_embed = nn.Parameter(
            nn.init.xavier_normal_(torch.tensor(np.random.randn(self.n_nodes, self.embedding_dim)))).to(self.device)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)
    
    @staticmethod
    def get_sim_mat(mm_embeddings, min_v=0.0, max_v=1.0):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        sim = torch.clamp(sim, min=min_v, max=max_v)
        return sim
        
    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        self.cur_epoch += 1
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        # mm feature via gcns
        tmp_v_feat = self.v_MLP_1(F.leaky_relu(self.v_MLP(self.v_feat)))
        tmp_t_feat = self.t_MLP_1(F.leaky_relu(self.t_MLP(self.t_feat)))

        # get sim for items
        txt_emb = self.text_nn(self.t_feat)
        img_emb = self.image_nn(self.v_feat)

        # get angle
        v0, v1, eps = txt_emb, img_emb, 1e-8
        v0_norm = F.normalize(v0, p=2, dim=1)
        v1_norm = F.normalize(v1, p=2, dim=1)
        # Compute cosine of angle between vectors
        dot = (v0_norm * v1_norm).sum(dim=1).clamp(-1 + eps, 1 - eps)  # avoid NaN
        theta = torch.acos(dot)
        mix_p = torch.distributions.beta.Beta(self.alpha, self.alpha).sample((v0_norm.size(0),)).to(self.device)
        a = (torch.sin(mix_p * theta)/torch.sin(theta)).unsqueeze(1)
        b = (torch.sin((1-mix_p)*theta)/torch.sin(theta)).unsqueeze(1)
        mix_f = a * v0_norm + b * v1_norm

        # preparing for GCN
        rep_uv = torch.cat((self.v_preference, tmp_v_feat), dim=0)
        rep_ut = torch.cat((self.t_preference, tmp_t_feat), dim=0)
        rep_sh = torch.cat((self.id_preference, mix_f), dim=0)
        v_x = torch.cat((F.normalize(rep_uv), F.normalize(rep_ut), F.normalize(rep_sh)), 1)

        # gcns
        ego_emb = v_x
        all_embeddings = [ego_emb]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_emb)
            ego_emb = side_embeddings
            all_embeddings += [ego_emb]
        representation = torch.stack(all_embeddings, dim=0).sum(dim=0)

        # User/item
        item_rep = representation[self.n_users:]

        vt_rep = representation[:self.n_users]
        v_rep = vt_rep[:, :self.embedding_dim]
        t_rep = vt_rep[:, self.embedding_dim: 2 * self.embedding_dim]
        s_rep = vt_rep[:, 2 * self.embedding_dim:]

        _att2 = F.softmax(self.theta, dim=1)
        # user fusion
        user_rep = torch.cat((_att2[:, 0, :] * v_rep, _att2[:, 1, :] * t_rep, _att2[:, 2, :] * s_rep), dim=1)

        # item gcns
        h = item_rep
        for i in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        item_rep = item_rep + h

        self.result_embed = torch.cat((user_rep, item_rep), dim=0)

        return F.normalize(user_rep, dim=-1), F.normalize(item_rep, dim=-1), mix_f

    ##### LOSS
    @staticmethod
    def alignment(x, y, alpha=2):
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    @staticmethod
    def uniformity(x, t=2):
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    @staticmethod
    def uniformity_i(x, sim, t=2):
        d = (torch.pdist(x, p=2).pow(2)-2+2*sim).mul(-t)
        return d.exp().mean().log()

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, mix_feat = self.forward(self.masked_adj)
        self.build_item_graph = False

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        sim_mt = self.get_sim_mat(mix_feat[neg_items], min_v = self.min_sim, max_v=self.max_sim)
        idx = torch.combinations(torch.arange(neg_items.size(0)))
        sim = sim_mt[idx[:, 0], idx[:, 1]]

        align = self.alignment(u_g_embeddings, pos_i_g_embeddings)
        uniform = self.gamma * (self.uniformity(u_g_embeddings) + self.uniformity_i(neg_i_g_embeddings, sim)) / 2
        mf_loss = align + uniform
        return mf_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

