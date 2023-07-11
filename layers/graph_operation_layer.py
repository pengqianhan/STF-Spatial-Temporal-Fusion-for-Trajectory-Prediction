
import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size##=3
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        ##x: (N, C, T, V)=(N, 4, 6, 120)
        ##A: (N, 3(max_hop+1), V, V)=(N, 3(max_hop+1),120,120)
        assert A.size(1) == self.kernel_size
        # print('x shape before conv in graph_operation_layer.py',x.shape)
        x = self.conv(x)
        # print('x shape after conv in graph_operation_layer.py',x.shape)
        n, kc, t, v = x.size()

        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,nkvw->nctw', (x, A))##('kctv,kvw->ctw', (x, A))

        return x.contiguous(), A


##########GAT######################################################


# this efficient implementation comes from https://github.com/xptree/DeepInf/
#this code reference from https://github.com/huang-xx/STGAT/blob/master/STGAT/models.py

#@save
class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))




class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)##assign 0 to the items in self.bias
        else:
            self.register_parameter("bias", None)###????

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h,adj):
        bs, n = h.size()[:2]### h is of size bs x n x f_in
        # print('h.unsqueeze(1) shape', h.unsqueeze(1).shape)  ##
        # print('self.w', self.w.shape)  ##
        h_prime = torch.matmul(h.unsqueeze(1), self.w)### bs x n_head x n x f_out
        attn_src = torch.matmul(h_prime, self.a_src)# bs x n_head x n x 1
        attn_dst = torch.matmul(h_prime, self.a_dst)# bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)

        mask = 1 - adj.unsqueeze(1) # bs x 1 x n x n
        mask=mask>0##new
        # print('mask dtype',mask.dtype,mask.shape)
        attn.data.masked_fill_(mask, float("-inf"))### fill "-inf" to the position in attn where the value is 1

        attn = self.softmax(attn)# bs x n_head x n x n
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)# bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn###output.shape:()

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        ##n_units =[in_channels, 64, 64] (first layer feature,second layer feature, output feature)
        ##n_heads=[3,1] # (first layer number of  heads, second layer number of heads)

        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1##=layers=2
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )

        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x,A):

        #### (N, C, T, V)=(N, C, 6, 120)->(N,720,C)######

        x=x.reshape(x.shape[0],-1,x.shape[1])

        ##x.shape bs x n x f_in
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            ###########MLP##########################


            x, attn = gat_layer(x,A)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x##(bs,n,64)

class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end,A):
        graph_embeded_data = []
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj,A)
            # print('curr_seq_graph_embedding',curr_seq_graph_embedding.shape)
            graph_embeded_data.append(curr_seq_graph_embedding)
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data### shape
