#4*4的冰湖环境
#起点在左上角，终点在右下角
#到达冰洞或者目标终点时行走就会提前结束
#每一步行走的奖励是0，到达目标的奖励是1
import copy

import gym
env=gym.make("FrozenLake-v1")#创建环境
env=env.unwrapped#解封装才能访问状态转移矩阵P
env.render()#环境渲染，通常是弹窗显示或打印出可视化环境

holes=set()
ends=set()

for s in env.P:
    #0-15
    for a in env.P[s]:
        #0-3
        for s_ in env.P[s][a]:
            #p next_state r done
            if s_[2]==1.0:
                ends.add(s_[1])#代表是目标
            #不是目标
            if s_[3]==True:
                holes.add(s_[1])#代表是冰湖

holes=holes-ends

print("冰洞的索引：",holes)
print("目标的索引：",ends)
#冰洞的索引： {11, 12, 5, 7}
#目标的索引： {15}

#查看目标左边一个单位的状态转移信息,可以看到每个动作都可以等概率滑行到3种可能的结果
for a in env.P[14]:
    print(env.P[14][a])


#下面尝试使用策略迭代

class PolicyIteration:
    """
    策略迭代算法
    """
    def __init__(self,env,theta,gamma):
        self.env=env
        self.theta=theta
        self.gamma=gamma

        self.v=[0]*self.env.ncol*self.env.nrow
        self.pi=[[0.25,0.25,0.25,0.25] for i in range(self.env.ncol*self.env.nrow)]

    def policy_evaluation(self):#策略评估
        cnt=1
        while 1:
            max_diff=0
            new_v=[0]*self.env.nrow*self.env.ncol

            for s in range(self.env.nrow*self.env.ncol):
                qsa_list=[] #开始计算状态s下的所有Q(s,a)
                for a in range(4):
                    qsa=0

                    for res in self.env.P[s][a]:
                        p,next_state,r,done=res
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))

                    qsa_list.append(self.pi[s][a]*qsa)

                new_v[s]=sum(qsa_list)
                max_diff=max(max_diff,abs(new_v[s]-self.v[s]))
            self.v=new_v
            if max_diff<self.theta:break
            cnt+=1
        print("策略评估进行%d轮后完成"%cnt)

    def policy_improvement(self):#策略提升
        for s in range(self.env.nrow*self.env.ncol):
            qsa_list=[]
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p,next_state,r,done=res
                    qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                qsa_list.append(qsa)
            maxq=max(qsa_list)
            cntq=qsa_list.count(maxq)
            self.pi[s]=[1/cntq if q==maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):
        while 1:
            self.policy_evaluation()
            old_pi=copy.deepcopy(self.pi)
            new_pi=self.policy_improvement()
            if old_pi==new_pi:
                break


def print_agent(agent,action_meaning,disaster,end):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print("%6.6s" % ('%.3f' % agent.v[i*agent.env.ncol+j]),end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i*agent.env.ncol+j) in disaster:
                print("****",end=' ')
            elif (i*agent.env.ncol+j) in end:
                print("EEEE",end=' ')
            else:
                action=agent.pi[i*agent.env.ncol+j]
                pi_str=''
                for k in range(len(action_meaning)):
                    pi_str+=action_meaning[k] if action[k]>0 else 'o'
                print(pi_str,end=' ')
        print()

                
action_meaning=['<','v','>','^']
theta=1e-5
gamma=0.9
agent=PolicyIteration(env,theta,gamma)
agent.policy_iteration()

print_agent(agent,action_meaning,[5,7,11,12],[15])

'''
冰洞的索引： {11, 12, 5, 7}
目标的索引： {15}
[(0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False)]
[(0.3333333333333333, 13, 0.0, False), (0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True)]
[(0.3333333333333333, 14, 0.0, False), (0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False)]
[(0.3333333333333333, 15, 1.0, True), (0.3333333333333333, 10, 0.0, False), (0.3333333333333333, 13, 0.0, False)]
策略评估进行25轮后完成
策略提升完成
策略评估进行58轮后完成
策略提升完成
状态价值：
 0.069  0.061  0.074  0.056 
 0.092  0.000  0.112  0.000 
 0.145  0.247  0.300  0.000 
 0.000  0.380  0.639  0.000 
策略：
<ooo ooo^ <ooo ooo^ 
<ooo **** <o>o **** 
ooo^ ovoo <ooo **** 
**** oo>o ovoo EEEE 
'''


#下面尝试使用价值迭代

class Valueiteration:
    def __init__(self,env,theta,gamma):
        self.env=env
        self.theta=theta
        self.gamma=gamma

        self.v=[0]*self.env.ncol*self.env.nrow
        self.pi=[None for i in range(self.env.nrow*self.env.ncol)]


    def value_iteration(self):
        cnt=0
        while 1:
            max_diff=0
            new_v=[0]*self.env.nrow*self.env.ncol
            for s in range(self.env.nrow*self.env.ncol):
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    for res in self.env.P[s][a]:
                        p,next_state,r,done=res
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                    qsa_list.append(qsa)
                new_v[s]=max(qsa_list)
                max_diff=max(max_diff,abs(new_v[s]-self.v[s]))
            self.v=new_v
            if max_diff<self.theta:break
            cnt+=1
        print("价值迭代一共进行%d轮"%cnt)
        self.get_policy()

    def get_policy(self):
        #根据价值函数导出一个贪心策略
        for s in range(self.env.nrow*self.env.ncol):
            qsa_list=[]
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p,next_state,r,done=res
                    qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                qsa_list.append(qsa)
            maxq=max(qsa_list)
            cntq=qsa_list.count(maxq)
            self.pi[s]=[1/cntq if q==maxq else 0 for q in qsa_list]


action_meaning=['<','v','>','^']
theta=1e-5
gamma=0.9
agent=Valueiteration(env,theta,gamma)
agent.value_iteration()

print_agent(agent,action_meaning,[5,7,11,12],[15])

'''
价值迭代一共进行60轮
状态价值：
 0.069  0.061  0.074  0.056 
 0.092  0.000  0.112  0.000 
 0.145  0.247  0.300  0.000 
 0.000  0.380  0.639  0.000 
策略：
<ooo ooo^ <ooo ooo^ 
<ooo **** <o>o **** 
ooo^ ovoo <ooo **** 
**** oo>o ovoo EEEE 
'''

#可以看到价值迭代算法的结果和策略迭代算法的结果完全一致
