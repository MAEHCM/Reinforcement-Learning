import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self,k):
        self.probs=np.random.uniform(size=k)
        self.best_idx=np.argmax(self.probs)
        self.best_prob=self.probs[self.best_idx]

        self.k=k

    def step(self,k):
        #当玩家选择了k号老虎机以后，根据拉动该老虎机的k号拉杆获得奖励的概率返回1/0
        if np.random.rand()<self.probs[k]:
                return 1
        else:
                return 0

#####
np.random.seed(1)
k=10
bandit_10_arm=BernoulliBandit(k)
print("随机生成一个%d臂伯努利老虎机"%k)
print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f"%(bandit_10_arm.best_idx,bandit_10_arm.best_prob))
#####

#下面实现策略选择动作：
# 1.根据策略选择动作
# 2.根据动作获取奖励
# 3.更新期望奖励估值
# 4.更新累计懊悔和技术

class Solver:
    def __init__(self,bandit):
        self.bandit=bandit
        self.counts=np.zeros(self.bandit.k)#每根拉杆的尝试次数
        self.reget=0    #当前步的累计懊悔
        self.actions=[] #记录每一步的动作选择哪个杆
        self.regets=[]  #记录每一个步的累计懊悔

    def update_reget(self,k):
        #计算累计懊悔并保存，k为当前动作选择的拉杆编号
        self.reget+=self.bandit.best_prob-self.bandit.probs[k]
        self.regets.append(self.reget)

    def run_one_step(self):
        #返回当前动作选择哪一根拉杆，由具体策略决定
        raise NotImplementedError

    def run(self,num_steps):
        #运行一定的次数，num_steps为总运行次数
        for _ in range(num_steps):
            k=self.run_one_step()

            self.counts[k]+=1
            self.actions.append(k)
            self.update_reget(k)

#在多臂老虎机问题中，设计策略时就需要平衡探索和利用的次数，使得每次累积的奖励最大化。
#一个比较常用的思路就是开始时做比较多的探索，在对每根拉杆都有比较准确的估计后，再进行利用
#目前已有的一些比较经典的算法来解决这个问题。
#如e-贪婪算法，上置信界算法和汤普森采样算法

#e-贪婪算法
#另e=0.01，T=5000

class EpsilonGreedy(Solver):
    def __init__(self,bandit,epslion=0.01,init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon=epslion
        #初始化拉动所有拉杆的 期望 奖励估值
        self.estimates=np.array([init_prob]*self.bandit.k)

    def run_one_step(self):
        if np.random.random()<self.epsilon:
            #随机选择一根拉杆
            k=np.random.randint(0,self.bandit.k)
        else:
            k=np.argmax(self.estimates) #选择期望奖励估值最大的拉杆

        #拉动k杆子获得奖励 1/0
        r=self.bandit.step(k)

        #估计期望奖励更新公式
        self.estimates[k]+= 1. / (self.counts[k]+1)*(r-self.estimates[k])

        return k

#为了更加直观地展示，可以把每一时间步的累积函数绘制出来
def plot_results(solvers,solver_names):
    #生成累积懊悔随时间变化的图像。输入solvers是一个列表，列表中的每个元素是一种特定的策略，solver_names为策略名称
    for idx,solver in enumerate(solvers):
        time_list=range(len(solver.regets))
        plt.plot(time_list,solver.regets,label=solver_names[idx])

    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.k)
    plt.legend()
    plt.show()

'''np.random.seed(1)
epsilon_greedy_solver=EpsilonGreedy(bandit_10_arm,epslion=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：',epsilon_greedy_solver.reget)
plot_results([epsilon_greedy_solver],["EpsilonGreedy"])'''

#在经历了开始的一小段时间后，e-贪婪算法的累积懊悔几乎是线性增长的。
np.random.seed(0)
epsilons=[1e-4,0.01,0.1,0.25,0.5]
epsilon_greedy_solver=[EpsilonGreedy(bandit_10_arm,epslion=e) for e in epsilons]
epsilon_greedy_names=["epsilon={}".format(e) for e in epsilons]

for solver in epsilon_greedy_solver:
    solver.run(5000)

plot_results(epsilon_greedy_solver,epsilon_greedy_names)
#实验结果表明，基本上无论e取值为多少，累积懊悔都是线性增长的，在这个例子中，随着e的增大，累积懊悔增长的遂率也会增大

#接下来我们尝试e值随时间衰减的e-贪婪算法，采用的具体衰减形式为反比例衰减，e=1/t
class DecayEpsilonGreedy(Solver):
    def __init__(self,bandit,init_prob=1.0):
        super(DecayEpsilonGreedy, self).__init__(bandit)
        self.estimates=np.array([init_prob]*self.bandit.k)
        self.total_count=0

    def run_one_step(self):
        self.total_count+=1

        if np.random.random()<1/self.total_count:
            k=np.random.randint(0,self.bandit.k)
        else:
            k=np.argmax(self.estimates)

        r=self.bandit.step(k)#得到reward
        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])#计算期望值

        return k

np.random.seed(1)
decaying_epsilon_greedy_solver=DecayEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(500)

print('epsilon值衰减的贪婪算法的累积懊悔为:',decaying_epsilon_greedy_solver.reget)
plot_results([decaying_epsilon_greedy_solver],["DecayingEpsilonGreedy"])

#从实验结果图中可以发现，随着时间做反比例衰减的e-贪婪算法能够使累积懊悔与时间步的关系变成次线性的，这明显优于固定e的e-贪婪算法

#上置信界算法
#这种思路主要是基于不确定性，因为此时第一根拉杆只被拉动过一次，它的不确定性很高，一根拉杆的不确定性越大，他就越具有探索的价值，因此探索之后我们可能发现他的期望奖励很大
#U(a)随着一个动作被尝试次数的增大而减小

class UCB(Solver):
    def __init__(self,bandit,coef,init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count=0
        self.estimates=np.array([init_prob]*self.bandit.k)
        self.coef=coef

    def run_one_step(self):
        self.total_count+=1
        ucb=self.estimates+self.coef*np.sqrt(np.log(self.total_count)/(2*(self.counts+1)))#计算上置信界的最大拉杆

        k=np.argmax(ucb)#选出上置信界最大的拉杆

        r=self.bandit.step(k)#reward

        self.estimates[k]+=1./(self.counts[k]+1)*(r-self.estimates[k])

        return k

np.random.seed(1)
coef=1
UCB_solver=UCB(bandit_10_arm,coef)
UCB_solver.run(5000)
print('上置信界的算法累积懊悔为:',UCB_solver.reget)
plot_results([UCB_solver],['UCB'])

#MAB中还有一种经典的算法，“汤普森采样”（计算所有拉杆的最高奖励概率的蒙特卡洛采样方法）

class ThompsonSampling(Solver):
    def __init__(self,bandit):
        super(ThompsonSampling, self).__init__(bandit)

        self._a=np.ones(self.bandit.k)  #列表，表示每根拉杆奖励为1的次数
        self._b=np.ones(self.bandit.k)  #列表，表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples=np.random.beta(self._a,self._b)#按照Beta分布采样的一组奖励样本
        k=np.argmax(samples)
        r=self.bandit.step(k)

        self._a[k]+=r#更新Beta分布的第一个参数
        self._b[k]+=(1-r)#更新Beta分布的第二个参数

        return k

np.random.seed(1)
thompson_sampling_solver=ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为:',thompson_sampling_solver.reget)
plot_results([thompson_sampling_solver],['thompsonsamping'])
#
#通过实验我们可以得到以下结论，e-贪婪算法的累积懊悔是随着时间线性增长的
#而另外三种算法(e-衰减贪婪算法，上置信界算法，汤普森采样算法)的累积懊悔都是随着时间次线性增长的(具体为对数形式增长)

#多臂老虎机问题与强化学习的一大区别在于其与环境的交互并不会改变环境，即多臂老虎机的每次交互结果和以往的动作无关，所以看做无状态的强化学习








