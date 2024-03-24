import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self,ncol,nrow):
        self.ncol=ncol
        self.nrow=nrow
        #起始坐标
        self.x=0
        self.y=self.nrow-1

    def step(self,action):
        change=[[0,-1],[0,1],[-1,0],[1,0]]

        self.x=min(self.ncol-1,max(0,self.x+change[action][0]))
        self.y=min(self.nrow-1,max(0,self.y+change[action][1]))

        new_state=self.y*self.ncol+self.x

        reward=-1

        done=False

        if self.y==self.nrow-1 and self.x>0:
            done=True
            if self.x!=self.ncol-1:
                reward=-100

        return new_state,reward,done

    def reset(self):
        self.x=0
        self.y=self.nrow-1
        return self.y*self.ncol+self.x



class nstep_sarsa:
    def __init__(self,n,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        self.Q_table=np.zeros([ncol*nrow,n_action])
        self.n_action=n_action

        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon

        self.n=n
        #n步的sarsa算法

        self.state_list=[]
        self.action_list=[]
        self.reward_list=[]

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action

    def best_action(self,state):#用于打印策略
        Q_max=np.max(self.Q_table[state])
        a=[0 for _ in range(self.n_action)]

        for i in range(self.n_action):
            if self.Q_table[state,i]==Q_max:
                a[i]=1
        return a

    def update(self,s0,a0,r,s1,a1,done):
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)

        if len(self.state_list)==self.n:
            G=self.Q_table[s1,a1]
            for i in reversed(range(self.n)):
                G=self.gamma*G+self.reward_list[i]

                #终止状态到达，更新前面没在当前轮中被更新的Q
                if done and i>0:
                    s=self.state_list[i]
                    a=self.action_list[i]
                    self.Q_table[s,a]+=self.alpha*(G-self.Q_table[s,a])

            s=self.state_list.pop(0)
            a=self.action_list.pop(0)
            self.reward_list.pop(0)

            #更新头部
            self.Q_table[s,a]+=self.alpha*(G-self.Q_table[s,a])

        if done:
            self.state_list=[]
            self.action_list=[]
            self.reward_list=[]

np.random.seed(0)
n_step=5

alpha=0.1
gamma=0.9
epsilon=0.1

ncol=12
nrow=4

env=CliffWalkingEnv(ncol,nrow)
agent=nstep_sarsa(n_step,ncol,nrow,epsilon,alpha,gamma)
num_episodes=500 #智能体在环境中运行的序列数量

return_list=[]#记录每一条序列的回报

for i in range(10):#10轮
    with tqdm(total=int(num_episodes/10),desc='Iteration %d'%i) as pbar:
        for i_episode in range(int(num_episodes/10)):

            episode_return=0
            state=env.reset()
            action=agent.take_action(state)
            done=False
            #先做一个动作，然后后续只要不是走到悬崖或者终点就一直跑
            while not done:
                next_state,reward,done=env.step(action)
                next_action=agent.take_action(next_state)

                episode_return+=reward

                agent.update(state,action,reward,next_state,next_action,done)

                state=next_state
                action=next_action

            return_list.append(episode_return)

            if (i_episode+1)%10==0:
                pbar.set_postfix({
                    'episode':'%d' % (num_episodes/10*i+i_episode+1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                                  })
            pbar.update(1)


episodes_list=list(range(len(return_list)))

plt.plot(episodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('5-steps Sarsa on {}'.format('Cliff Waling'))
plt.show()