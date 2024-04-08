import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoderLayer

X=torch.randn(128,64,512)

d_model=512
n_head=4

class multi_head_attention(nn.Module):
    def __init__(self,d_model,n_head):
        super(multi_head_attention,self).__init__()
        self.d_model=d_model
        self.n_head=n_head
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        self.w_combine = nn.Linear(d_model,d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,q,k,v,mask=None):
        batch,time,dimension=q.shape

        n_d=self.d_model//self.n_head
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)

        #维度交换
        q=q.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        k=k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v=v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        score=q@k.transpose(2,3)/math.sqrt(n_d)
        print(score.shape)
        if mask is not None:
        #time大小的下三角矩阵为负无穷
            #mask=torch.tril(torch.ones(time,time ,dtype=bool))
            score=score.masked_fill(mask==0,float("-inf"))


        score=self.softmax(score)@v

        score=score.permute(0,2,1,3).contiguous().view(batch,time,dimension)

        output=self.w_combine(score)
        return output
# attention=multi_head_attention(d_model,n_head)
# output=attention(X,X,X)
#
# total_params = sum(p.numel() for p in attention.parameters())
# print("Total parameters:", total_params)

class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model):
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1)


#位置编码
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,max_len,device):
        super(PositionalEmbedding,self).__init__()
        self.encoding=torch.zeros(max_len,d_model,device=device)
        self.encoding.requires_grad_(False)
        pos=torch.arange(0,max_len,device=device)
        pos=pos.float().unsqueeze(1)
        _2i=torch.arange(0,d_model,2,device=device)

        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))
    def forward(self,x):
        seq_len=x.shape[1]
        return self.encoding[:seq_len,:]

class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-10):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)

        var=x.var(-1,unbiased=False,keepdim=True)
        out=(x-mean)/torch.sqrt(var+self.eps)

        out=self.gamma*out+self.beta

        return out

# layerNorm=LayerNorm(d_model=512)
# layerNorm(X)
#FFN
class PositionwiseFeedForward(nn.Module):
    def __init__ (self,d_model,hidden,dropout):
        super(PositionwiseFeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,hidden)
        self.fc2=nn.Linear(hidden,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        x=self.fc1(x)
        x=F.relu(x)
        x = self.dropout(x)
        x=self.fc2(x)
        return x

#Total Embedding

class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,max_len,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb=TokenEmbedding(vocab_size,d_model)
        self.pos_emb=PositionalEmbedding(d_model,max_len,device)
        self.dropout=nn.Dropout(drop_prob)
    def forward(self,x):
        tok_emb=self.tok_emb(x)
        pos_emb=self.pos_emb(x)
        return self.dropout(tok_emb+pos_emb)


class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob)->None:
        super(EncoderLayer, self).__init__()
        self.attention=multi_head_attention(d_model,num_heads)
        self.norm1=LayerNorm(d_model)
        self.drop1=nn.Dropout(drop_prob)
        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm2=LayerNorm(d_model)
        self.drop2=nn.Dropout(drop_prob)

    def forward(self,x,mask=None):
        _x=x
        x=self.attention(x,x,x,mask)

        x=self.dorp1(x)
        x=self.norm1(x+_x)

        _x=x
        x=self.ffn(x)

        x=self.drop2(x)
        x=self.norm2(x+_x)
        return  x

class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,num_heads,drop_prob)->None:
        super(DecoderLayer,self).__init__()
        self.attention1=multi_head_attention(d_model,num_heads)
        self.norm1=LayerNorm(d_model)
        self.drop1=nn.Dropout(drop_prob)
        self.cross_attention=multi_head_attention(d_model,n_head)
        self.norm2=LayerNorm(d_model)
        self.drop2=nn.Dropout(drop_prob)

        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm3=LayerNorm(d_model)
        self.drop3=nn.Dropout(drop_prob)

    def forward(self,dec,enc,t_mask,s_mask):
        _x=dec
        #下三角掩码
        x=self.attention1(dec,dec,dec,t_mask)
        x=self.drop1(x)
        x=self.norm1(x+_x)

        if enc is not None:
            _x=x
            x=self.cross_attention(x,enc,enc,s_mask)

            x=self.drop2(x)
            x=self.norm2(x+_x)
        _x=x
        x=self.ffn(x)
        x=self.drop3(x)
        x=self.norm3(x+_x)
        return x

class Encoder(nn.Module):
    def __init__(self,env_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Encoder,self).__init__()

        self.embeddings=TransformerEmbedding(d_model,max_len,env_voc_size,drop_prob,device)
        self.layers=nn.ModuleList(
            [EncoderLayer(d_model,ffn_hidden,n_head,drop_prob) for _ in range(n_layer)]
        )

    def forward(self,x,s_mask):
        x=self.embeddings(x)
        for layer in self.layers:
            x=layer(x,s_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layer, drop_prob, device):
        super(Decoder, self).__init__()

        self.embeddings = TransformerEmbedding(d_model, max_len, dec_voc_size, drop_prob, device)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layer)]
        )
        self.fc=nn.Linear(d_model,dec_voc_size)

    def forward(self, dec,enc, s_mask,t_mask):
        dec = self.embeddings(dec)
        for layer in self.layers:
            dec = layer(dec,enc,t_mask,s_mask)

        dec=self.fc(dec)
        return dec

class Transformer(nn.Module):
    def __init__(self,src_pad_idx,trg_pad_idx,enc_voc_size,dec_voc_size,max_len,d_model,n_heads,ffn_hidden,n_layers,drop_prob,device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.encoder=Encoder(enc_voc_size,max_len,d_model,ffn_hidden,n_heads,n_layers,drop_prob,device)
        self.decoder=Decoder(dec_voc_size,max_len,d_model,ffn_hidden,n_heads,n_layers,drop_prob,device)

    def make_casual_mask(self,q,k):
        len_q,len_k=q.size(1),k.size(1)
        mask=torch.tril(torch.ones(len_q,len_k)).type(torch.BoolTensor).to(device)
        return mask
    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q,len_k=q.size(1),k.size(1)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)
        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k=k.repeat(1,1,len_q,1)
        mask=q&k
        return mask

    def forward(self,src,trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx)*self.make_casual_mask(trg,trg)
        src_trg_mask=self.make_pad_mask(trg,src,self.trg_pad_idx,self.src_pad_idx)

        enc=self.encoder(src,src_mask)

        output=self.decoder(trg,enc,trg_mask,src_trg_mask)
        return output


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


if __name__ == "__main__":
    enc_voc_size = 5893
    dec_voc_size = 7853
    src_pad_idx = 1
    trg_pad_idx = 1
    trg_sos_idx = 2
    batch_size = 128
    max_len = 1024
    d_model = 512
    n_layers = 3
    n_heads = 2
    ffn_hidden = 1024
    drop_prob = 0.1
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        d_model=d_model,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        n_heads=n_heads,
        n_layers=n_layers,
        drop_prob=drop_prob,
        device=device,
    ).to(device)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model.apply(initialize_weights)
    src = torch.load('tensor_src.pt').to(device)
    src = torch.cat((src, torch.ones(src.shape[0], 2, dtype=torch.int).to(device)), dim=-1)
    trg = torch.load('tensor_trg.pt').to(device)

    result = model(src, trg)
    print(result, result.shape)








