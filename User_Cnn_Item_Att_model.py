import time

import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from config import Config

class DeepCoNNDataset(Dataset):
    def __init__(self, data_path, word2vec, config):
        self.word2vec = word2vec
        self.config = config
        self.PAD_WORD_idx = self.word2vec.vocab[self.config.PAD_WORD].index
        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating', 'timestamp'])
        df = df.sort_values('timestamp', ascending=True)  # 按时间排序
        df['review'] = df['review'].apply(self._review2id)  # 分词->数字

        self.null_idx = set()  # 暂存空样本的下标，最后删除他们
        user_idset, user_reviews, user_times = self._get_reviews(df)  # 收集每个user的评论列表
        item_idset, item_reviews, item_times = self._get_reviews(df, 'itemID', 'userID')
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)
        self.user_idset = user_idset[[idx for idx in range(user_reviews.shape[0]) if idx not in self.null_idx]]
        self.item_idset = item_idset[[idx for idx in range(item_reviews.shape[0]) if idx not in self.null_idx]]
        # print('user_idset:', user_idset)
        # print('self.user:', self.user_idset)
        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.null_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.null_idx]]
        self.user_times = user_times[[idx for idx in range(user_times.shape[0]) if idx not in self.null_idx]]
        self.item_times = item_times[[idx for idx in range(item_times.shape[0]) if idx not in self.null_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.null_idx]]

    def __getitem__(self, idx):
        return self.user_idset[idx], self.item_idset[idx], self.user_reviews[idx], self.item_reviews[idx], \
               self.user_times[idx], self.item_times[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review', 'timestamp']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_idset, lead_reviews, lead_times = [], [], []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # 取lead除对当前costar外的评论：列表
            times = df_data['timestamp'][df_data[costar] != costar_id].to_list()
            if len(reviews) == 0:
                self.null_idx.add(idx)
            reviews, times = self._adjust_list(reviews, times, self.config.review_length, self.config.review_count)
            lead_idset.append(lead_id)
            lead_reviews.append(reviews)
            lead_times.append(times)
        return torch.LongTensor(lead_idset), torch.LongTensor(lead_reviews), torch.LongTensor(lead_times)

    def _adjust_list(self, reviews, times, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # 评论数量固定
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # 每条评论定长
        times = times[:r_count] + [time.perf_counter()] * (r_count - len(times))  # 时间数目固定
        for i in range(len(times)):  # 时间差
            times[i] = (times[i + 1] if i < len(times) - 1 else time.perf_counter()) - times[i]
        return reviews, times

    def _review2id(self, review):
        #  将一个评论字符串分词并转为数字
        if not isinstance(review, str):
            return []  # 貌似pandas的一个bug，读取出来的评论如果是空字符串，review类型会变成float
        wids = []
        for word in review.split():
            if word in self.word2vec:
                wids.append(self.word2vec.vocab[word].index)  # 单词映射为数字
            else:
                wids.append(self.PAD_WORD_idx)
        return wids


class DeepCoNNDataset_valid(Dataset):
    def __init__(self, data_path, word2vec, config):
        self.word2vec = word2vec
        self.config = config
        self.PAD_WORD_idx = self.word2vec.vocab[self.config.PAD_WORD].index
        df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating', 'timestamp'])
        df = df.sort_values('timestamp', ascending=True)  # 按时间排序
        df['review'] = df['review'].apply(self._review2id)  # 分词->数字

        self.null_idx = set()  # 暂存空样本的下标，最后删除他们
        user_idset, user_reviews, user_times = self._get_reviews(df)  # 收集每个user的评论列表  # 这个地方修改过
        item_idset, item_reviews, item_times = self._get_reviews(df, 'itemID', 'userID')
        rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

        self.user_idset = user_idset[
            [idx for idx in range(user_reviews.shape[0]) if idx not in self.null_idx]]  # 这两个曾经也是没有的
        self.item_idset = item_idset[
            [idx for idx in range(item_reviews.shape[0]) if idx not in self.null_idx]]  # 这两个曾经也是没有的
        self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.null_idx]]
        self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.null_idx]]
        self.user_times = user_times[[idx for idx in range(user_times.shape[0]) if idx not in self.null_idx]]
        self.item_times = item_times[[idx for idx in range(item_times.shape[0]) if idx not in self.null_idx]]
        self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.null_idx]]

    def __getitem__(self, idx):
        return self.user_idset[idx], self.item_idset[idx], self.user_reviews[idx], self.item_reviews[idx], \
               self.user_times[idx], self.item_times[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review', 'timestamp']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_idset, lead_reviews, lead_times = [], [], []
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            reviews = df_data['review'][df_data[costar] != costar_id].to_list()  # 取lead除对当前costar外的评论：列表
            times = df_data['timestamp'][df_data[costar] != costar_id].to_list()
            if len(reviews) == 0:
                self.null_idx.add(idx)
            reviews, times = self._adjust_list(reviews, times, self.config.review_length, self.config.review_count)
            lead_idset.append(lead_id)
            lead_reviews.append(reviews)
            lead_times.append(times)
        return torch.LongTensor(lead_idset), torch.LongTensor(lead_reviews), torch.LongTensor(lead_times)

    def _adjust_list(self, reviews, times, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))  # 评论数量固定
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]  # 每条评论定长
        times = times[:r_count] + [time.perf_counter()] * (r_count - len(times))  # 时间数目固定
        for i in range(len(times)):  # 时间差
            times[i] = (times[i + 1] if i < len(times) - 1 else time.perf_counter()) - times[i]
        return reviews, times

    def _review2id(self, review):
        #  将一个评论字符串分词并转为数字
        if not isinstance(review, str):
            return []  # 貌似pandas的一个bug，读取出来的评论如果是空字符串，review类型会变成float
        wids = []
        for word in review.split():
            if word in self.word2vec:
                wids.append(self.word2vec.vocab[word].index)  # 单词映射为数字
            else:
                wids.append(self.PAD_WORD_idx)
        return wids


'''
nn.LSTM()参数意义
input_size：x的特征维度
hidden_size：隐藏层的特征维度
num_layers：lstm隐层的层数，默认为1
bias：False则bih=0和bhh=0. 默认为True
batch_first：True则下面输入输出的数据格式为 (batch, seq, feature)
dropout：除最后一层，每一层的输出都进行dropout，默认为: 0
bidirectional：True则为双向lstm默认为False

输入数据格式(batch_first=False的情况下，batch在第二维)：
input(seq_len, batch, input_size) 
h0(num_layers * num_directions, batch, hidden_size) 
c0(num_layers * num_directions, batch, hidden_size)
输出数据格式(batch_first=False)：
output(seq_len, batch, hidden_size * num_directions) 
hn(num_layers * num_directions, batch, hidden_size) 
cn(num_layers * num_directions, batch, hidden_size)
 '''


def to_onehot(id, batchsize, num_id):
    config = Config()
    # uid_onehot = torch.FloatTensor(batchsize, num_id).to(config.device)
    uid_onehot = torch.FloatTensor(batchsize, num_id)
    uid_onehot.zero_()  #
    uid_onehot.scatter_(1, id.unsqueeze(1), 1)  # 根据index中的索引按照dim的方向填进input
    return uid_onehot


class Net(nn.Module):

    def __init__(self, config, word_dim):
        super(Net, self).__init__()
        self.kernel_count = config.kernel_count
        self.review_count = config.review_count

        self.lstm = nn.LSTM(input_size=word_dim, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=41,
                out_channels=config.kernel_count,
                kernel_size=config.kernel_size,
                padding=(config.kernel_size - 1) // 2),  # out shape(new_batch_size, kernel_count, review_length)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, config.review_length)),  # out shape(new_batch_size,kernel_count,1)
            nn.Dropout(p=config.dropout_prob))

        self.linear = nn.Sequential(  # 100 * 10
            nn.Linear(config.kernel_count * config.review_count, config.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))

    def forward(self, vec, time_vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        # vec （1000，80，300）  time_vec(1000,80,1)
        vec, (hn, cn) = self.lstm(vec)
        # vec (1000,80,40)
        # conv in(new_batch_size, word_dim=hidden_size * num_directions, review_length),
        # print('vec', vec)
        # print('time', time_vec)
        vec = torch.cat([vec, time_vec], dim=2)
        # vec = (1000,80,41)

        # print('vec', vec) # latent (1000,100,1)
        latent = self.conv(vec.permute(0, 2, 1))  # out(new_batch_size, kernel_count, 1) kernel count指一条评论潜在向量
        latent = self.linear(latent.reshape(-1, self.kernel_count * self.review_count))  # （100，50）
        return latent  # out shape(batch_size, cnn_out_dim)


class FactorizationMachine(nn.Module):

    def __init__(self, p, k):  # p=cnn_out_dim*2  k=10
        super(FactorizationMachine, self).__init__()

        self.v = nn.Parameter(torch.zeros(p, k))
        self.linear = nn.Linear(p, 1, bias=True)

    def forward(self, x):
        linear_part = self.linear(x)  # input shape(batch_size, cnn_out_dim), out shape(batch_size, 1) (100,1)
        inter_part1 = torch.mm(x, self.v)  # (100,10)
        inter_part2 = torch.mm(x ** 2, self.v ** 2)  # (100,10)
        pair_interactions = torch.sum(inter_part1 ** 2 - inter_part2, dim=1)
        output = linear_part.transpose(1, 0) + 0.5 * pair_interactions
        return output.view(-1, 1)  # out shape(batch_size, 1)

class MLP(nn.Module):  # ying
    def __init__(self, factor_num ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(factor_num, 25)
        self.fc2 = nn.Linear(25, 6)
        self.fc3 = nn.Linear(6, 1)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.view(-1,1)

class DeepCoNN(nn.Module):

    def __init__(self, config, word2vec):
        self.num_uid = config.num_uid
        self.num_iid = config.num_iid
        super(DeepCoNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(word2vec.vectors))

        # self.W_time = nn.Parameter(torch.randn([config.review_length, word2vec.vector_size]))
        # self.bias_time = nn.Parameter(torch.randn([config.review_length, word2vec.vector_size]))
        self.user_mapping = nn.Linear(config.num_uid, 1, bias=False)
        self.item_mapping = nn.Linear(config.num_iid, 1, bias=False)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.review_length = config.review_length
        self.Net_u = Net(config, word_dim=word2vec.vector_size)
        # self.Net_i = Net(config, word_dim=word2vec.vector_size)
        self.Net_i = Net_Att(config, 41, config.kernel_count, config.kernel_size, 1, config.dropout_prob)
        self.fm = FactorizationMachine((config.cnn_out_dim) * 2 , 10)
        # self.mlp = MLP((config.cnn_out_dim+1) * 2)

    # user_review shape(batch_size, review_count, review_length)
    def forward(self, user_id, item_id, user_review, item_review, user_time, item_time):
        count = user_review.shape[0]
        new_batch_size = user_review.shape[0] * user_review.shape[1]
        user_review = user_review.view(new_batch_size, -1)
        item_review = item_review.view(new_batch_size, -1)

        u_vec = self.embedding(user_review)
        i_vec = self.embedding(item_review)
        # 考虑时间
        user_time = user_time.double()
        item_time = item_time.double()

        user_time = (user_time - user_time.min()) / (user_time.max() - user_time.min() + 1.0)  # 归一化
        item_time = (item_time - item_time.min()) / (item_time.max() - item_time.min() + 1.0)

        user_time = user_time.view(new_batch_size, 1, 1).expand(-1, self.review_length, -1)
        item_time = item_time.view(new_batch_size, 1, 1).expand(-1, self.review_length, -1)
        user_time = user_time.float()

        # item_time = torch.zeros(item_time.shape)
        item_time = item_time.float()

        # u_vec = torch.cat([u_vec, user_time], dim=2)
        # i_vec = torch.cat([i_vec, item_time], dim=2)

        uid_onehot = to_onehot(user_id, count, self.num_uid)  # (100,1429)
        iid_onehot = to_onehot(item_id, count, self.num_iid)  # (100,900)

        uid_emb = self.dropout(self.user_mapping(uid_onehot))  # (100,1)
        iid_emb = self.dropout(self.item_mapping(iid_onehot))

        user_latent = self.Net_u(u_vec, user_time)
        # print(user_latent.size())
        # print(uid_emb.size())
        # print("=================")
        user_latent = torch.cat((user_latent, uid_emb), dim=1)  # 不需要uid就删掉这个

        # user_latent = user_latent + user_id  # 我添加的
        # (100,50)
        item_latent = self.Net_i(i_vec, item_time)
        item_latent = torch.cat((item_latent, iid_emb), dim=1) # 不添加Iid 就删掉这个
        # (100,50)
        concat_latent = torch.cat((user_latent, item_latent), dim=1)
        # (100,100)
        prediction = self.fm(concat_latent)
        # prediction = self.mlp(concat_latent)
        return prediction


class Net_Att(nn.Module):

    def __init__(self, config, in_feat, out_feat, kernel_size, num_head, dropout):
        super(Net_Att, self).__init__()
        self.kernel_count = config.kernel_count
        self.review_count = config.review_count
        self.lstm = nn.LSTM(input_size=300, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)
        # self.multihead_att_layer = SelfMultiheadAtt( 80 , config.cnn_out_dim , config.kernel_size , 1 , config.dropout_prob)
        # # 80, 50, 3 , 1, 0.5
        self.multihead_att_layer = SelfMultiheadAtt(config, in_feat, out_feat, kernel_size, num_head, dropout)
        self.linear = nn.Sequential(
            nn.Linear(config.kernel_count * config.review_count, config.cnn_out_dim),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_prob))

    def forward(self, vec, time_vec):  # input shape(new_batch_size, review_length, word2vec_dim)
        # vec （1000，80，300）  time_vec(1000,80,1)
        vec, (hn, cn) = self.lstm(vec)
        # vec (1000,80,40)
        # conv in(new_batch_size, word_dim=hidden_size * num_directions, review_length),
        # print('vec', vec)
        # print('time', time_vec)
        vec = torch.cat([vec, time_vec], dim=2)
        # vec = (1000,80,41)

        # print('vec', vec)
        # 经过 多头 注意力机制的输出需要为 (1000,100,1)
        # latent = self.multihead_att_layer(vec.permute(0,2,1)) # 我们这个地方 应该不需要转换
        latent = self.multihead_att_layer(vec)
        # latent = self.conv(vec.permute(0, 2, 1))  # out(new_batch_size, kernel_count, 1) kernel count指一条评论潜在向量
        # 现在算出来的结果是 1000,80,50
        latent = self.linear(latent.reshape(-1, self.kernel_count * self.review_count))
        return latent  # out shape(batch_size, cnn_out_dim) # (4000,50)


class SelfMultiheadAtt(nn.Module):
    def __init__(self, config, in_feat, out_feat, kernel_size, num_head, dropout):
        super().__init__()
        # 80, 50, 3 , 1, 0.5
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.kernel_size = kernel_size
        self.num_head = num_head
        self.dropout = dropout
        assert out_feat % num_head == 0

        # project q,k,v layer
        padding = (kernel_size - 1) // 2
        self.proj_q = nn.Sequential(nn.Conv1d(in_feat, out_feat, kernel_size, padding=padding),
                                    nn.Tanh())
        self.proj_k = nn.Sequential(nn.Conv1d(in_feat, out_feat, kernel_size, padding=padding),
                                    nn.Tanh())
        self.proj_v = nn.Sequential(nn.Conv1d(in_feat, out_feat, kernel_size, padding=padding),
                                    nn.Tanh())
        # nn.init.xavier_normal_(self.proj_q[0].weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_normal_(self.proj_k[0].weight, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_normal_(self.proj_v[0].weight, gain=nn.init.calculate_gain("relu"))

        # ffn layer
        # self.ffn = nn.Sequential(nn.Linear(out_feat, out_feat),
        #                          nn.Tanh())
        self.ffn = nn.Sequential(nn.MaxPool2d(kernel_size=(config.review_length, 1)))
        # nn.init.xavier_normal_(self.ffn[0].weight, gain=nn.init.calculate_gain('relu'))

        # layernorm
        # self.layer_norm = nn.LayerNorm(out_feat)

        self.dropout = nn.Dropout(dropout) if dropout != 0. else None

    @staticmethod
    def split_heads(q, k, v, num_head):
        def split_last_dimension_then_transpose(tensor, num_head):
            """
            tensor: [bz, seq_len, dim]
            out_tensor: [bz, num_head, seq_len, dim/num_head]
            """
            bz, seq_len, dim = list(tensor.size())
            tensor = tensor.view(bz, seq_len, num_head, dim // num_head).permute(0, 2, 1, 3)
            return tensor

        qs = split_last_dimension_then_transpose(q, num_head)
        ks = split_last_dimension_then_transpose(k, num_head)
        vs = split_last_dimension_then_transpose(v, num_head)

        return qs, ks, vs

    @staticmethod
    def concat_heads(output):
        def transpose_then_concat_last_two_dimension(tensor):
            """
            Args:
                output: [bz, num_head, seq_len, dim//num_head]
            Returns:
                output: [bz, seq_len, dim]
            """
            bz, num_head, seq_len, dim_per_head = list(tensor.size())
            tensor = tensor.permute(0, 2, 1, 3).reshape(bz, seq_len, num_head * dim_per_head)
            return tensor

        output = transpose_then_concat_last_two_dimension(output)

        return output

    @staticmethod
    def scaled_dot_product(qs, ks, vs, key_padding_mask=None):
        """
        Args:
            qs, ks, vs: [bz, num_head, seq_len, dim_per_head]
            key_padding_mask: [bz, seq_len]

        Returns:
            out: [bz, num_head, seq_len, dim_per_head]
        """
        key_dim_per_head = ks.size(-1)
        logits = torch.matmul(qs, ks.permute(0, 1, 3, 2))
        logits = logits / (key_dim_per_head ** 0.5)  # [bz, num_head, seq_len_q, seq_len_k]

        if key_padding_mask != None:
            logits = logits.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                -1e8
            )

        att_weights = F.softmax(logits, dim=-1)  # [bz, num_head, seq_len_q, seq_len_k]

        return torch.matmul(att_weights, vs)

    def forward(self, inputs, key_padding_mask=None):
        """
        Args:
            inputs: [bz, seq_len, in_feat]
            key_padding_mask: [bz, seq_len]

        Returns:
            outpus: [bz, seq_len, out_feat]
        """
        bz, seq_len, _ = inputs.size()  # 现在已经改为了 1000,80,41

        inputs = inputs.permute(0, 2, 1)  # [bz, in_feat, seq_len]  1000,41,80

        # project q, k, v  (1000,80,50)
        q = self.proj_q(inputs).permute(0, 2, 1)
        k = self.proj_k(inputs).permute(0, 2, 1)
        v = self.proj_v(inputs).permute(0, 2, 1)  # [bz, seq_len, out_feat]

        # split  qs ks vs (1000,1,80,50)  最后一个维度取决于 q、k、v的output（卷积的定义部分）
        qs, ks, vs = self.split_heads(q, k, v, self.num_head)  # [bz, num_head, seq_len, out_feat/num_head]
        # concat
        outputs = self.scaled_dot_product(qs, ks, vs)  # [bz, num_head, seq_len, out_feat/num_head]
        outputs = self.concat_heads(outputs)  # [bz, seq_len, out_feat] (1000,80,50) 以上都没问题 同415
        # ffn
        outputs = self.ffn(outputs)  # (1000,80,50)
        # dropout
        if self.dropout != None:
            outputs = self.dropout(outputs)
        # outputs = self.layer_norm(outputs)
        return outputs
