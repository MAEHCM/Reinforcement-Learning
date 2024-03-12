import copy

class CliffWalkingEnv:
    """
    悬崖漫步环境
    """
    def __init__(self,ncol=12,nrow=4):
        self.ncol=ncol
        self.nrow=nrow

        self.P=self.createP()

    def createP(self):

        P=[[[] for j in range(4)] for i in range(self.ncol*self.nrow)]
        change=[[0,-1],[0,1],[-1,0],[1,0]]

        #初始化状态
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    #位置在悬崖或者是终点，因为无法交互，因此任何奖励动作都是0
                    if i==self.nrow-1 and j>0:
                        P[i*self.ncol+j][a]=[(1,i*self.ncol+j,0,True)]

                    #其他位置
                    nextx=min(self.ncol-1,max(0,j+change[a][0]))
                    nexty=min(self.nrow-1,max(0,i+change[a][1]))
                    next_state=nexty*self.ncol+nextx
                    reward=-1
                    done=False

                    #下一个位置在悬崖或者终点
                    if nexty==self.nrow-1 and nextx>0:
                        done=True
                        #如果不是终点的话
                        if nextx!=self.ncol-1:
                            reward=-100
                    P[i*self.ncol+j][a]=[(1,next_state,reward,done)]
        return P


class ValueIteration:
    """
    价值迭代算法
    """
    def __init__(self,env,theta,gamma):
        self.env=env
        self.theta=theta
        self.gamma=gamma

        self.v=[0] * self.env.ncol*self.env.nrow
        #价值迭代后得到的策略
        self.pi=[None for i in range(self.env.ncol*self.env.nrow)]


    def value_iteration(self):
        cnt=0
        while 1:
            max_diff=0
            new_v=[0]*self.env.ncol*self.env.nrow

            for s in range(self.env.ncol*self.env.nrow):
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    for res in self.env.P[s][a]:
                        p,next_state,r,done=res
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                    qsa_list.append(qsa)

                #最优状态价值==max(最优动作价值)==max( Q(s,a) )
                new_v[s]=max(qsa_list)
                max_diff=max(max_diff,abs(new_v[s]-self.v[s]))
            self.v=new_v
            if max_diff <self.theta:
                break
            cnt+=1

        #然后返回一个确定性策略
        print("价值迭代一共进行%d轮"%cnt)
        self.get_policy()


    def get_policy(self):
        #根据价值函数导出一个贪婪策略
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
            #动作均分概率
            self.pi[s]=[1/cntq if q==maxq else 0 for q in qsa_list]


def print_agent(agent,action_meaning,disaster,end):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6s' % ('%.3f' % agent.v[i*agent.env.ncol+j]),end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if(i*agent.env.ncol+j) in disaster:
                print('****',end=' ')
            elif(i*agent.env.ncol+j) in end:
                print('EEEE',end=' ')
            else:
                action=agent.pi[i*agent.env.ncol+j]
                pi_str=''
                for k in range(len(action_meaning)):
                    pi_str+=action_meaning[k] if action[k]>0 else 'o'
                print(pi_str,end=' ')
        print()

env=CliffWalkingEnv()
action_meaning=['^','v','<','>']
theta=0.001
gamma=0.9

agent=ValueIteration(env,theta,gamma)
agent.value_iteration()

print_agent(agent,action_meaning,list([range(37,47)]),[47])