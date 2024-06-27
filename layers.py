import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

class MultiHeadDepthwiseSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0):
        super(MultiHeadDepthwiseSelfAttention, self).__init__()
        ...

    def forward(self, x):
        
        ...

if __name__=="__main__":
    b,n,f = 4, 16, 128
    x = torch.randn(b,n,f)
    # net = MultiHeadSelfAttention(f)
    net = TransformerEncoder(f)
    torchsummary.summary(net, (n,f))
    # out = net(x)
    # print(out.shape)




# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# def const_init(model, fill=0.01):
#     for name, param in model.named_parameters():
#         param.data.fill_(fill)

# class TransformerEncoder(nn.Module):
#     def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
#         super(TransformerEncoder, self).__init__()
#         self.la1 = nn.LayerNorm(feats)
#         # self.msatorch = nn.MultiheadAttention(embed_dim=feats, num_heads=head, dropout=dropout)
#         self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
#         self.la2 = nn.LayerNorm(feats)
#         self.mlp = nn.Sequential(
#             nn.Linear(feats, mlp_hidden),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(mlp_hidden, feats),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
    
#     #tune vit
#     def forward(self, x):
#         out = self.msa(self.la1(x)) + x
#         out = self.mlp(self.la2(out)) + out
#         return out
    
#     #torch vit
#     # def forward(self, x):
#     #     x = x.transpose(0, 1)  # nn.MultiheadAttention expects input as (seq_len, batch, features)
#     #     xl = self.la1(x)
#     #     attn_output, _ = self.msatorch(xl, xl, xl)  # query, key, value are all the same for self-attention
#     #     out = attn_output.transpose(0, 1)  # transpose back to original shape
#     #     out = out + x.transpose(0, 1)  # Residual connection
#     #     out = self.mlp(self.la2(out)) + out  # Another layer and residual connection
#     #     return out

# #head prun ver
# # class MultiHeadSelfAttention(nn.Module):
# #     def __init__(self, feats:int, head:int=8, dropout:float=0.):
# #         super(MultiHeadSelfAttention, self).__init__()
# #         self.head = head
# #         self.feats = feats
# #         self.sqrt_d = self.feats**0.5
# #         prune = 4

# #         self.q = nn.Linear(feats, feats//self.head*prune)
# #         self.k = nn.Linear(feats, feats//self.head*prune)
# #         self.v = nn.Linear(feats, feats//self.head*prune)

# #         self.o = nn.Linear(feats//self.head*prune, feats)
# #         self.dropout = nn.Dropout(dropout)


# #     def forward(self, x):
# #         b, n, f = x.size()
# #         q1 = self.q(x)
# #         k1 = self.k(x)
# #         v1 = self.v(x)

# #         q = q1.view(b, n, -1, self.feats//self.head).transpose(1,2)
# #         k = k1.view(b, n, -1, self.feats//self.head).transpose(1,2)
# #         v = v1.view(b, n, -1, self.feats//self.head).transpose(1,2)

# #         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
# #         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
# #         o = self.dropout(self.o(attn.flatten(2)))
# #         return o
    
# #linear decompose original
# # class MultiHeadSelfAttention(nn.Module):
# #     def __init__(self, feats:int, head:int=8, dropout:float=0.):
# #         super(MultiHeadSelfAttention, self).__init__()
# #         self.head = head
# #         self.feats = feats
# #         self.sqrt_d = self.feats**0.5
# #         decom = 192

# #         self.q1 = nn.Linear(feats, decom)
# #         self.q2 = nn.Linear(decom, feats)
# #         self.k1 = nn.Linear(feats, decom)
# #         self.k2 = nn.Linear(decom, feats)
# #         self.v1 = nn.Linear(feats, decom)
# #         self.v2 = nn.Linear(decom, feats)

# #         self.o1 = nn.Linear(feats, decom)
# #         self.o2 = nn.Linear(decom, feats)
# #         self.dropout = nn.Dropout(dropout)


# #     def forward(self, x):
# #         b, n, f = x.size()
# #         q = self.q1(x)
# #         q = self.q2(q)
# #         k = self.k1(x)
# #         k = self.k2(k)
# #         v = self.v1(x)
# #         v = self.v2(v)

# #         q = q.view(b, n, self.head, self.feats//self.head).transpose(1,2)
# #         k = k.view(b, n, self.head, self.feats//self.head).transpose(1,2)
# #         v = v.view(b, n, self.head, self.feats//self.head).transpose(1,2)

# #         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
# #         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
# #         out = self.o1(attn.flatten(2))
# #         out = self.o2(out)
# #         o = self.dropout(out)
# #         return o

# #many linear ver
# # class MultiHeadSelfAttention(nn.Module):
# #     def __init__(self, feats:int, head:int=8, dropout:float=0.):
# #         super(MultiHeadSelfAttention, self).__init__()
# #         self.head = head
# #         self.feats = feats
# #         self.sqrt_d = self.feats**0.5

# #         self.q = GroupedLinear(feats, feats, num_groups=head)
# #         self.k = GroupedLinear(feats, feats, num_groups=head)
# #         self.v = nn.Linear(feats, feats)

# #         self.o = nn.Linear(feats, feats)
# #         self.dropout = nn.Dropout(dropout)

# #     def forward(self, x):
# #         #batch, seq_len, dim
# #         b, n, f = x.size()
# #         q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
# #         k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
# #         v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

# #         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
# #         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
# #         o = self.dropout(self.o(attn.flatten(2)))
# #         return o

# # original vit
# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feats:int, head:int=8, dropout:float=0.):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.head = head
#         self.feats = feats
#         self.sqrt_d = self.feats**0.5

#         self.q = nn.Linear(feats, feats)
#         self.k = nn.Linear(feats, feats)
#         self.v = nn.Linear(feats, feats)

#         self.o = nn.Linear(feats, feats)
#         self.dropout = nn.Dropout(dropout)


#     def forward(self, x):
#         b, n, f = x.size()
#         q1 = self.q(x)
#         k1 = self.k(x)
#         v1 = self.v(x)

#         q = q1.view(b, n, self.head, self.feats//self.head).transpose(1,2)
#         k = k1.view(b, n, self.head, self.feats//self.head).transpose(1,2)
#         v = v1.view(b, n, self.head, self.feats//self.head).transpose(1,2)

#         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
#         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
#         o = self.dropout(self.o(attn.flatten(2)))
#         return o
    
# # from torch import Tensor
# # class GroupedLinear(nn.Module):
# #     __constants__ = ['in_features', 'out_features', 'num_groups']
# #     in_features: int
# #     out_features: int
# #     num_groups: int
# #     weight: Tensor

# #     def __init__(self, in_features: int, out_features: int, num_groups: int) -> None:
# #         super(GroupedLinear, self).__init__()
# #         self.in_features = in_features
# #         self.out_features = out_features
# #         self.num_groups = num_groups
# #         assert in_features % num_groups == 0, "in_features must be divisible by groups"
# #         assert out_features % num_groups == 0, "out_features must be divisible by groups"
# #         self.gls = nn.ModuleList([nn.Linear(in_features // num_groups, out_features // num_groups) for _ in range(num_groups)])
    
# #     def forward(self, x):
# #         original_shape = x.shape
# #         # Reshape x to 2D
# #         x = x.reshape(-1, self.in_features)
# #         # Split input x into groups
# #         x_groups = x.chunk(self.num_groups, dim=1)
# #         # Apply each linear layer to its corresponding group
# #         outputs = [l(group) for l, group in zip(self.gls, x_groups)]
# #         # Concatenate the outputs
# #         output = torch.cat(outputs, dim=1)
# #         # Reshape back to the original shape with new feature size
# #         return output.reshape(*original_shape[:-1], self.out_features)

# if __name__=="__main__":
#     b,n,f = 4, 16, 128
#     x = torch.randn(b,n,f)
#     # net = MultiHeadSelfAttention(f)
#     net = TransformerEncoder(f)
#     # out = net(x)
#     # print(out.shape)


#----------------------------------------------------------------------------------------------------------------------


#### gyudong group linear
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math


# class TransformerEncoder(nn.Module):
#     def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
#         self.feats = feats
#         self.head = head
#         super(TransformerEncoder, self).__init__()
#         self.la1 = nn.LayerNorm(feats//head)
#         self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
#         self.la2 = nn.LayerNorm(feats//head)
#         self.mlp = nn.Sequential(
#             GroupedLinear(feats, mlp_hidden, num_groups = head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             GroupedLinear(mlp_hidden, feats, num_groups = head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
#     def forward(self, x):
#         b, n, f = x.size()
#         x = x.view(b, n, self.head, self.feats//self.head)
#         out = self.msa(self.la1(x)) + x
#         out = self.mlp(self.la2(out)) + out
#         return out.flatten(2)


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feats:int, head:int=8, dropout:float=0.):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.head = head
#         self.feats = feats
#         self.sqrt_d = self.feats**0.5
#         # self.shuffle_order = list(range(feats)) #random shuffle

#         # # Inverse shuffle order
#         # self.inverse_shuffle_order = [0] * feats #random shuffle
#         # for i, j in enumerate(self.shuffle_order): #random shuffle
#         #     self.inverse_shuffle_order[j] = i #random shuffle
                
#         # random.shuffle(self.shuffle_order) #random shuffle
#         self.q = GroupedLinear(feats, feats, num_groups= head)
#         self.k = GroupedLinear(feats, feats, num_groups= head)
#         self.v = GroupedLinear(feats, feats, num_groups= head)

#         self.o = GroupedLinear(feats, feats, num_groups= head)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         #batch, seq_len, dim
#         b, n, h, f = x.size()
#         # x = x[:, :, self.shuffle_order] #Random shuffling
#         # x = x.transpose(2,3).reshape(b,n,h,f)
#         q = self.q(x).transpose(1,2)
#         k = self.k(x).transpose(1,2)
#         v = self.v(x).transpose(1,2)

#         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
#         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
#         o = self.dropout(self.o(attn))
#         return o

# from torch import Tensor


# class GroupedLinear(nn.Module):
#     __constants__ = ['in_features', 'out_features', 'num_groups']
#     in_features: int
#     out_features: int
#     num_groups: int
#     weight: Tensor
#     def __init__(self, in_features: int, out_features: int, num_groups: int, device=None, dtype=None, bias: bool = True,) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(GroupedLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_groups = num_groups
#         assert in_features % num_groups == 0, "in_features must be divisible by groups"
#         assert out_features % num_groups == 0, "out_features must be divisible by groups"
#         self.weight = nn.Parameter(torch.empty((num_groups, in_features // num_groups, out_features // num_groups), **factory_kwargs))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(num_groups, out_features//num_groups, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)        
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         for g in range(self.num_groups):
#             nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             for g in range(self.num_groups):
#                 nn.init.uniform_(self.bias[g], -bound, bound)

#     def forward(self, x):
#         # x = (.., h, f//h)
#         # Apply each linear layer to its corresponding group
#         out = torch.einsum("...gi, gij->...gj", x, self.weight)
#         if self.bias is not None:
#             out += self.bias
#         return out



#----------------------------------------------------------------------------------------------------------------------
# separable

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # import torchsummary
# import random
# import math
# from torch.cuda import nvtx
# # from thop import profile

# class TransformerEncoder(nn.Module):
#     def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
#         self.feats = feats
#         self.head = head
#         super(TransformerEncoder, self).__init__()
#         self.la1 = nn.LayerNorm(feats//head)
#         self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
#         self.la2 = nn.LayerNorm(feats//head)
#         self.mlp = nn.Sequential(
#             GroupedLinear(feats, mlp_hidden, num_groups = head),
#             FeatureWiseLinear(head,head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             GroupedLinear(mlp_hidden, feats, num_groups = head),
#             FeatureWiseLinear(head,head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
#     def forward(self, x):
#         nvtx.range_push('model forward_split')
#         b, n, f = x.size()
#         x = x.view(b, n, self.head, self.feats//self.head)
#         out = self.msa(self.la1(x)) + x
#         out = self.mlp(self.la2(out)) + out
#         nvtx.range_pop()
#         return out.flatten(2)


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feats:int, head:int=8, dropout:float=0.):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.head = head
#         self.feats = feats
#         self.sqrt_d = self.feats**0.5

#         self.q = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), FeatureWiseLinear(head,head),)
#         self.k = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), FeatureWiseLinear(head,head),)
#         self.v = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), FeatureWiseLinear(head,head),)

#         self.o = nn.Sequential(GroupedLinear(feats, feats, num_groups= head), FeatureWiseLinear(head,head),)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         #batch, seq_len, dim///
#         b, n, h, f = x.size()

#         q = self.q(x).transpose(1,2)
#         k = self.k(x).transpose(1,2)
#         v = self.v(x).transpose(1,2)

#         # nvtx.range_push('Attention + score')
#         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
#         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
#         # nvtx.range_pop()
#         o = self.dropout(self.o(attn))
#         return o

# from torch import Tensor


# class GroupedLinear(nn.Module):
#     __constants__ = ['in_features', 'out_features', 'num_groups']
#     in_features: int
#     out_features: int
#     num_groups: int
#     weight: Tensor
#     def __init__(self, in_features: int, out_features: int, num_groups: int, device=None, dtype=None, bias: bool = True,) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(GroupedLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_groups = num_groups
#         assert in_features % num_groups == 0, "in_features must be divisible by groups"
#         assert out_features % num_groups == 0, "out_features must be divisible by groups"
#         self.weight = nn.Parameter(torch.empty((num_groups, in_features // num_groups, out_features // num_groups), **factory_kwargs))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(num_groups, out_features//num_groups, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)        
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         for g in range(self.num_groups):
#             nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             for g in range(self.num_groups):
#                 nn.init.uniform_(self.bias[g], -bound, bound)

#     def forward(self, x):
#         # x = (.., h, f//h)
#         # Apply each linear layer to its corresponding group
#         out = torch.einsum("...gi, gij->...gj", x, self.weight)
#         if self.bias is not None:
#             out += self.bias
#         return out


# class FeatureWiseLinear(nn.Module):
#     def __init__(self, in_groups: int, out_groups: int):
#         super(FeatureWiseLinear, self).__init__()
#         self.linear = nn.Linear(in_groups, out_groups)
#     def forward(self, x):
#         #b,n,h,f = x.size()
#         x = x.transpose(2,3) # b,n,f,h
#         x = self.linear(x)
#         x = x.transpose(2,3) # b,n,h,f
#         return x
        



#----------------------------------------------------------------------------------------------------------------------
# revised
    
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import random
# import math
# from torch.cuda import nvtx

# class TransformerEncoder(nn.Module):
#     def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
#         self.feats = feats
#         self.head = head
#         super(TransformerEncoder, self).__init__()
#         self.la1 = nn.LayerNorm(feats//head)
#         self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
#         self.la2 = nn.LayerNorm(feats//head)
#         self.mlp = nn.Sequential(
#             GroupedLinear(feats, mlp_hidden, num_groups = head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             GroupedLinear(mlp_hidden, feats, num_groups = head),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )
#     def forward(self, x):
#         nvtx.range_push('model forward_split')
#         b, n, f = x.size()
#         x = x.view(b, n, self.head, self.feats//self.head)
#         out = self.msa(self.la1(x)) + x #This should be changed
#         out = self.mlp(self.la2(out)) + out #This should be changed
#         out = out.transpose(2,3).reshape(b,n,self.head, self.feats//self.head)
#         nvtx.range_pop()
#         return out.flatten(2)


# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, feats:int, head:int=8, dropout:float=0.):
#         super(MultiHeadSelfAttention, self).__init__()
#         self.head = head
#         self.feats = feats
#         self.sqrt_d = self.feats**0.5
#         # self.shuffle_order = list(range(feats)) #random shuffle

#         # # Inverse shuffle order
#         # self.inverse_shuffle_order = [0] * feats #random shuffle
#         # for i, j in enumerate(self.shuffle_order): #random shuffle
#         #     self.inverse_shuffle_order[j] = i #random shuffle
                
#         # random.shuffle(self.shuffle_order) #random shuffle
#         self.q = GroupedLinear(feats, feats, num_groups= head)
#         self.k = GroupedLinear(feats, feats, num_groups= head)
#         self.v = GroupedLinear(feats, feats, num_groups= head)

#         self.o = GroupedLinear(feats, feats, num_groups= head)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         #batch, seq_len, dim///
#         b, n, h, f = x.size()
#         # x = x[:, :, self.shuffle_order] #Random shuffling
#         # x = x.transpose(2,3).reshape(b,n,h,f) #shuffle
#         q = self.q(x).transpose(1,2)
#         k = self.k(x).transpose(1,2)
#         v = self.v(x).transpose(1,2)

#         # nvtx.range_push('Attention + score')
#         score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
#         attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
#         # nvtx.range_pop()
#         o = self.dropout(self.o(attn))
#         return o

# from torch import Tensor


# class GroupedLinear(nn.Module):
#     __constants__ = ['in_features', 'out_features', 'num_groups']
#     in_features: int
#     out_features: int
#     num_groups: int
#     weight: Tensor
#     def __init__(self, in_features: int, out_features: int, num_groups: int, device=None, dtype=None, bias: bool = True,) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(GroupedLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_groups = num_groups
#         assert in_features % num_groups == 0, "in_features must be divisible by groups"
#         assert out_features % num_groups == 0, "out_features must be divisible by groups"
#         self.weight = nn.Parameter(torch.empty((num_groups, in_features // num_groups, out_features // num_groups), **factory_kwargs))
#         if bias:
#             self.bias = nn.Parameter(torch.empty(num_groups, out_features//num_groups, **factory_kwargs))
#         else:
#             self.register_parameter('bias', None)        
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         for g in range(self.num_groups):
#             nn.init.kaiming_uniform_(self.weight[g], a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             for g in range(self.num_groups):
#                 nn.init.uniform_(self.bias[g], -bound, bound)

#     def forward(self, x):
#         # x = (.., h, f//h)
#         # Apply each linear layer to its corresponding group
#         out = torch.einsum("...gi, gij->...gj", x, self.weight)
#         if self.bias is not None:
#             out += self.bias
#         return out

