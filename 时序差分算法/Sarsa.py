import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class CliffWalkingEnv:
    def __init__(self,nrow,ncol):
        self.nrow=nrow
        self.ncol=ncol

        #记录起点
        self.x=0
        self.y=self.nrow-1

    def step(self,action):
        change=[[0,-1],[0,1],[-1,0],[1,0]]

        self.x=min(self.ncol-1,max(0,self.x+change[action][0]))
        self.y=min(self.nrow-1,max(0,self.y+change[action][1]))

        new_state=self.y*self.ncol+self.x
        reward=-1
        done=False

        if self.y==self.nrow-1 and self.x>0:#下一个位置在悬崖或者是终点
            done=True
            if self.x!=self.ncol-1:
                reward=-100
        return  new_state,reward,done

    def reset(self):
        self.x=0
        self.y=self.nrow-1
        return self.y*self.ncol+self.x


class Sarasa:
    def __init__(self,ncol,nrow,
                 epsilon,gamma,alpha,
                 n_action=4):
        self.Q_table=np.zeros([nrow*ncol,n_action])
        self.n_action=n_action

        self.epsilon=epsilon
        self.gamma=gamma
        self.alpha=alpha

    #贪心采样最大动作价值的动作，为1-epsilon 贪心
    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.n_action)
        else:
            action=np.argmax(self.Q_table[state])
        return action

    def update(self,s0,a0,r,s1,a1):
        td_error=r+self.gamma*self.Q_table[s1,a1]-self.Q_table[s0,a0]
        self.Q_table[s0,a0]+=self.alpha*td_error

    #打印
    def best_action(self,state):
        Q_max=np.max(self.Q_table[state])
        a=[0 for _ in range(self.n_action)]

        for i in range(self.n_action):
            if self.Q_table[state][i]==Q_max:
                a[i]=1
        return a

ncol=12
nrow=4

env=CliffWalkingEnv(ncol,nrow)

np.random.seed(0)


epsilon=0.1
alpha=0.1
gamma=0.9

agent=Sarasa(ncol,nrow,epsilon,gamma,alpha)

num_episodes=500

return_list=[]

for i in range(10):#10轮
    with tqdm(total=int(num_episodes/10),desc='Iteration %d' % i) as pbar:#每一轮中的序列数50条

        for i_episode in range(int(num_episodes/10)):#一条序列
            epsilon_return=0#累积奖励

            state=env.reset()
            action=agent.take_action(state)
            done=False

            while not done:
                next_state,reward,done=env.step(action)
                next_action=agent.take_action(next_state)

                epsilon_return+=reward

                agent.update(state,action,reward,next_state,next_action)

                state=next_state
                action=next_action
            return_list.append(epsilon_return)

            #每十条序列打印一下这十条序列的平均回报
            if (i_episode+1)%10==0:
                pbar.set_postfix({
                    'episode':'%d' % (num_episodes/10*i+i_episode+1),
                    'return':'%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episode_list=list(range(len(return_list)))
plt.plot(episode_list,return_list)

plt.xlabel('Episodes')
plt.ylabel('Returns')

plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.show()
