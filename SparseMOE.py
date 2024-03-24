import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


##############################
torch.manual_seed(42)

###dataset
batch_size=16
block_size=32
###train
max_iters=500
learning_rate=1e-3
device='cuda'
###evel
eval_iters=400
eval_interval=100
###model
head_size=16
n_embed=128
n_head=8
n_layer=8
dropout=0.1
num_experts=8
topk=2
##############################

with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)

stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}

encode=lambda s:[stoi[c] for c in s]
decode=lambda l:''.join(itos[i] for i in l)

#[46, 48, 40, 1, 46, 43, 56, 43]
#hjb here

data=torch.tensor(encode(text),dtype=torch.long)
#torch.Size([1115389])

###################划分数据集
n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]
'''
tensor([258140, 252540, 992416, 387428])
tensor([[39, 52, 58,  1, 58, 46, 43, 47],
        [56, 42,  2,  0, 20, 43,  1, 47],
        [41, 46,  1, 51, 63,  1, 44, 56],
        [39, 63,  1, 63, 53, 59,  1, 52]])
'''
def get_batch(split):
    data=train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x,y

'''
torch.Size([4, 8])
tensor([[ 1, 58, 46, 47, 52, 45,  1, 21],
        [43, 42,  6,  1, 39, 52, 42,  1],
        [51, 43, 52, 58,  0, 32, 46, 39],
        [10,  1, 57, 46, 43,  1, 50, 47]])
'''
#torch.Size([4, 8, 16])

class Head(nn.Module):
    def __init__(self,head_size,n_embed):
        super(Head, self).__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(0.1)

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        score=q@k.transpose(-2,-1)*C**-0.5
        score=score.masked_fill(self.tril[:T,:T]==0,float('-inf'))#[B,T,T]
        score=F.softmax(score,dim=-1)
        score=self.dropout(score)
        output=score@v
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size,n_embed):
        super(MultiHeadAttention, self).__init__()
        self.heads=nn.ModuleList([Head(head_size,n_embed) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embed,n_embed)
        self.dropout=nn.Dropout(0.1)

    def forward(self,x):
        output=torch.cat([h(x) for h in self.heads],dim=-1)
        output=self.dropout(self.proj(output))
        return output

class Expert(nn.Module):
    def __init__(self,n_embed):
        super(Expert, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embed,4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed,n_embed),
            nn.Dropout(0.1),
        )
    def forward(self,x):
        return self.net(x)

class NoisyTopkRouter(nn.Module):
    def __init__(self,n_embed,num_experts,top_k):
        super(NoisyTopkRouter, self).__init__()
        self.topk=top_k
        self.topk_router=nn.Linear(n_embed,num_experts)
        self.noisy_linear=nn.Linear(n_embed,num_experts)

    def forward(self,x):
        #torch.Size([4, 8, 16])
        logits=self.topk_router(x)
        noisy_logits=self.noisy_linear(x)

        noisy=torch.randn_like(logits)*F.softplus(noisy_logits)
        noisy_logits=logits+noisy

        topk_logits,indices=noisy_logits.topk(self.topk,dim=-1)
        zeros=torch.full_like(noisy_logits,float('-inf'))
        sparse_logits=zeros.scatter(-1,indices,topk_logits)
        router_output=F.softmax(sparse_logits,dim=-1)
        return router_output,indices


class SparseMoE(nn.Module):
    def __init__(self,n_embed,num_experts,top_k):
        super(SparseMoE, self).__init__()
        self.router=NoisyTopkRouter(n_embed,num_experts,top_k)
        self.experts=nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k=top_k

    def forward(self,x):
        #torch.Size([4, 8, 16])
        gating_output,indices=self.router(x)
        #torch.Size([4, 8, 8]) 8个专家
        #torch.Size([4, 8, 2])

        final_output=torch.zeros_like(x)
        #torch.Size([4, 8, 16])

        flat_x=x.view(-1,x.size(-1))
        #torch.Size([32, 16])
        flat_gating_output=gating_output.view(-1,gating_output.size(-1))
        #torch.Size([32, 8])

        for i,expert in enumerate(self.experts):
            #当前batch size中，对于第i个专家，对原始输入滤出最大表征为编号i的那部分表征
            expert_mask=(indices==i).any(dim=-1)
            '''
            tensor([[False, False, False, False,  True, False,  True, False],
                    [False, False, False, False, False, False, False, False],
                    [False, False, False, False, False, False, False, False],
                    [ True,  True, False, False,  True, False, False, False]])
            '''
            flat_mask=expert_mask.view(-1)
            #torch.Size([32])

            if flat_mask.any():
                expert_input=flat_x[flat_mask]
                #torch.Size([5, 16])
                expert_output=expert(expert_input)
                #torch.Size([5, 16])

                gating_scores=flat_gating_output[flat_mask,i].unsqueeze(1)#[32,8]->[5,8]->[5]->[5,1]
                #torch.Size([5, 1])

                weighted_output=expert_output*gating_scores
                #torch.Size([5, 16])

                final_output[expert_mask]+=weighted_output.squeeze(1)
                #torch.Size([4, 8, 16])

        return final_output

class Block(nn.Module):
    def __init__(self,n_embed,n_head,head_size,num_experts,top_k):
        super(Block, self).__init__()
        self.sa=MultiHeadAttention(n_head,head_size,n_embed)
        self.smoe=SparseMoE(n_embed,num_experts,top_k)
        self.ln1=nn.LayerNorm(n_embed)
        self.ln2=nn.LayerNorm(n_embed)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.smoe(self.ln2(x))
        return x

class SparseMoELM(nn.Module):
    def __init__(self):
        super(SparseMoELM, self).__init__()

        self.token_embedding_table=nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table=nn.Embedding(block_size,n_embed)

        self.blocks=nn.Sequential(*[Block(n_embed,n_head,head_size,num_experts,topk) for _ in range(n_layer)])

        self.ln_f=nn.LayerNorm(n_embed)

        self.lm_head=nn.Linear(n_embed,vocab_size)

    def forward(self,idx,target=None):
        B,T=idx.shape
        tok_emb=self.token_embedding_table(idx)
        pos_emb=self.position_embedding_table(torch.arange(T,device=device))

        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)

        if target is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            target=target.view(B*T)

            loss=F.cross_entropy(logits,target)

        return logits,loss

    def generate(self,idx,max_new_token):

        for _ in range(max_new_token):
            idx_cond=idx[:,-block_size:]

            logits,loss=self(idx_cond)

            logits=logits[:,-1,:]#[B,C],最后一个位置上
            probs=F.softmax(logits,dim=-1)#[B,C]

            idx_next=torch.multinomial(probs,num_samples=1) #[B,1]

            idx=torch.cat((idx,idx_next),dim=1)

        return idx

def kaiming_init_weights(m):
    if isinstance(m,(nn.Linear)):
        init.kaiming_normal_(m.weight)

model=SparseMoELM().to(device)
model.apply(kaiming_init_weights)

print(sum(p.numel() for p in model.parameters())/1e6,'M parameters')

optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)


for iter in range(max_iters):
    xb,yb =get_batch('train')

    xb=xb.to(device)
    yb=yb.to(device)

    logits,loss=model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 20 ==0:
        print('iter: %d,    loss: %3f' % (iter,loss))

context=torch.zeros((1,1),dtype=torch.long,device=device)

print(decode(model.generate(context,2000)[0].tolist()))











