import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Categorical
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


def moving_avarage(a,window_size):
    cumulative_sum=np.cumsum(np.insert(a,0,0))
    middle=(cumulative_sum[window_size:]-cumulative_sum[:-window_size])/window_size

    r=np.arange(1,window_size-1,2)
    begin=np.cumsum(a[:window_size-1])[::2]/r
    end=(np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((begin,middle,end))

#在每一个状态下，梯度的修改是让策略更多地采样到带来较高Q值的动作，更少地采样到带来较低Q值的动作。
class PolicyNet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x),dim=1)


class REINFORCE:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,device):
        self.policy_net=PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimzer=torch.optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        self.gamma=gamma
        self.device=device

    def take_action(self,state):
        #[0.00117113 0.39857933 0.12473702 0.01926376]
        state=torch.tensor([state],dtype=torch.float).to(self.device)
        probs=self.policy_net(state)
        #tensor([[0.6315, 0.3685]], grad_fn=<SoftmaxBackward0>)
        action_dist=torch.distributions.Categorical(probs)
        #Categorical(probs: torch.Size([1, 2]))
        action=action_dist.sample()#返回的是分布的下标
        #tensor([1])
        return action.item()

    def update(self,trandiction_dict):
        reward_list=trandiction_dict['rewards']
        state_list=trandiction_dict['states']
        action_list=trandiction_dict['actions']

        G=0
        self.optimzer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward=reward_list[i]
            #1.0
            state=torch.tensor([state_list[i]],dtype=torch.float).to(self.device)
            #[1,4]
            action=torch.tensor([action_list[i]]).view(-1,1).to(self.device)
            #[1,1]

            log_prob=torch.log(self.policy_net(state).gather(1,action))

            G=reward+self.gamma*G

            loss=-log_prob*G
            loss.backward()

        self.optimzer.step()

learning_rate=1e-3
num_epsiodes=1000
hidden_dim=128
gamma=0.98
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env_name="CartPole-v1"
env=gym.make(env_name)

torch.manual_seed(0)

state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n

agent=REINFORCE(state_dim,hidden_dim,action_dim,learning_rate,gamma,device)

return_list=[]
for i in range(10):
    with tqdm(total=int(num_epsiodes/10),desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_epsiodes/10)):
            episode_return=0
            transition_dict={
                'states':[],
                'actions':[],
                'next_states':[],
                'rewards':[],
                'dones':[],
            }
            state=env.reset()[0]
            #[0.32213205 0.15685384 0.15635999 0.8576582 ]
            done=False
            while not done:
                action=agent.take_action(state)
                #1
                next_state,reward,_,done,_=env.step(action)
                #[0.32213205 0.15685384 0.15635999 0.8576582 ] 1.0 False

                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)

                state=next_state
                episode_return+=reward

            return_list.append(episode_return)
            agent.update(transition_dict)

            if (i_episode+1)%10==0:
                pbar.set_postfix({
                    'episode':'%d'%(num_epsiodes/10+i_episode+1),
                    'return' :'%.3f'%np.mean(return_list[-10:])
                                  })
            pbar.update(1)

epsiodes_list=list(range(len(return_list)))
plt.plot(epsiodes_list,return_list)
plt.xlabel('Epsiodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()

mv_return=moving_avarage(return_list,9)
plt.plot(epsiodes_list,mv_return)
plt.xlabel('Epsidoes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()



