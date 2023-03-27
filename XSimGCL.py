import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from util.sampler import next_batch_pairwise
from util.load_data import Data
from util.model_utils import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE

# Paper: XSimGCL - Towards Extremely Simple Graph Contrastive Learning for Recommendation


class XSimGCL(nn.Module):
    def __init__(self, args, training_set, test_set):
        super(XSimGCL, self).__init__()
        self.cl_rate = float(args.cl_rate)
        self.eps = float(args.eps)
        self.temp = float(args.temp)
        self.n_layers = int(args.gnn_layer)
        self.layer_cl = int(args.l)
        self.emb_size = int(args.emb_size)
        self.maxEpoch = int(args.epoch)
        self.batch_size = int(args.batch)
        self.reg = float(args.reg_lambda)
        self.lr = float(args.lr)
        self.data = Data(args, training_set, test_set)
        self.model = XSimGCL_Encoder(self.data, self.emb_size, self.eps, self.n_layers, self.layer_cl)

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = model(True)
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([user_idx,pos_idx],rec_user_emb,cl_user_emb,rec_item_emb,cl_item_emb)
                batch_loss = rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0 and n > 0:
                    print('training epoch:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item())
                    break
            # evaluate
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
                self.__evaluate(epoch+1)


    def cal_cl_loss(self, idx, user_view1,user_view2,item_view1,item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()
        user_cl_loss = InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()

    def __metrics(self, uids, predictions, topk):
        user_num = 0
        all_recall = 0
        all_ndcg = 0
        for i in range(len(uids)):
            uid = uids[i]    # uid = test_users_idx
            prediction = list(predictions[i][:topk])
            # label = test_labels[uid]
            # label = np.array(list(self.data.test_set[uid].keys()))   # label = test_items_id
            if(uid>=len(self.data.test_items_idx)):
                print(uid)
            label = self.data.test_items_idx[uid]
            if len(label) > 0:
                hit = 0
                idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
                dcg = 0
                for item in label:
                    if item in prediction:
                        hit += 1
                        loc = prediction.index(item)
                        dcg = dcg + np.reciprocal(np.log2(loc + 2))
                all_recall = all_recall + hit / len(label)
                all_ndcg = all_ndcg + dcg / idcg
                user_num += 1
        return all_recall / user_num, all_ndcg / user_num

    def __evaluate(self,epoch):
        batch_num = int(np.ceil(len(self.data.test_users_idx) /self.batch_size))
        all_recall_20 = 0
        all_ndcg_20 = 0
        for batch_index in tqdm(range(batch_num)):
            start = batch_index * self.batch_size
            end = min((batch_index+1)*self.batch_size, len(self.data.test_users_idx))
            test_uidxs_input = torch.LongTensor(self.data.test_users_idx[start:end])
            preds = self.user_emb[test_uidxs_input] @ self.item_emb.T
            # for n,uid in enumerate(test_uidxs_input):
            #     uid_train_item = self.data.train_item_idx[uid]
            #     preds[n,uid_train_item] = 0
            train_mask = self.data.ui_adj[test_uidxs_input, self.data.user_num:].toarray()
            train_mask = torch.Tensor(train_mask).cuda()
            preds = preds * (1 - train_mask)
            # preds = preds.cpu()
            predictions = preds.argsort(descending=True)   # predictions = preds_items_idx
            # predictions = np.array(predictions.cpu())
            #top@20
            # recall_20, ndcg_20 = self.__metrics(list(self.data.test_set.keys())[start:end],predictions,20)
            recall_20, ndcg_20 = self.__metrics(test_uidxs_input, predictions, 20)
            all_recall_20 += recall_20
            all_ndcg_20 += ndcg_20
        print('-------------------------------------------')
        print('Test of epoch:',epoch,' Recall@20:', all_recall_20 / batch_num, 'Ndcg@20:',all_ndcg_20 / batch_num)


class XSimGCL_Encoder(nn.Module):
    def __init__(self, data, emb_size, eps, n_layers, layer_cl):
        super(XSimGCL_Encoder, self).__init__()
        self.data = data
        self.eps = eps
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k==self.layer_cl-1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.data.user_num, self.data.item_num])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl, [self.data.user_num, self.data.item_num])
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings