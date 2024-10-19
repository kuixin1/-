from matplotlib import pyplot as plt 
import numpy as np
from geomdl import NURBS 
from geomdl.visualization import VisMPL
from mpl_toolkits.mplot3d import Axes3D  

class NURBSCurve:
    def __init__(self, T, P, W=None):
        self.reset(T, P, W)
 
    # 初始化NURBS相关参数
    def reset(self, T, P, W=None): 
        # T节点向量 P控制点坐标 W 权重
        self.T = T
        self.P = P
        # 如果 W 是 None 那就是 BSpline 曲线
        if W is None:
            # 跟 P 形状一样的数组，元素都是1
            self.W = np.ones_like(P)
        else:
            self.W = W
        self.m = len(T) # 节点的数量
        self.n = len(P) # 控制点的数量
        self.k = self.m - self.n - 1 # NURBS 阶数的计算 BSpline 的阶数通常为3
        self.dpN = np.zeros(self.m) # 动态规划求N的 用于计算基函数
        self.P2 = None
 
    # 动态规划 计算基函数的值
    def caculate_dpN(self, t):
        for k in range(self.k+1): # 动态规划
            for i in range(self.n): 
                if k == 0:
                    self.dpN[i] = 1 * (self.T[i] <= t < self.T[i+1])
                else:
                    w1 = w2 = 0
                    if self.T[i+k] != self.T[i]:
                        w1 = (t-self.T[i])/(self.T[i+k]-self.T[i])
                    if self.T[i+k+1] != self.T[i+1]:
                        w2 = (self.T[i+k+1] - t)/(self.T[i+k+1]-self.T[i+1])  
                    self.dpN[i] = w1*self.dpN[i] + w2*self.dpN[i+1]

    # 计算曲线在参数 t 处的值    
    def __call__(self, t:int):
        c1 = 1e-30 # 分子 float 范围 1.7e38
        c2 = 1e-30 # 分母 float 范围 1.7e38
        self.caculate_dpN(t)
        for i in range(self.n):  # 0 - n
            c1 += self.P[i] * self.dpN[i] * self.W[i]
            c2 += self.dpN[i] * self.W[i]
        return c1/c2
    
    # 对曲线重采样，增强数据的可用性
    def resample(self, sample:int): 
        if self.P2 != None:
            if self.P2.shape[0] == sample:
                return self.P2 
        first = self.T[self.k]
        gama = 1e-9
        last = self.T[-self.k-1] * (1-gama) + gama * self.T[-self.k-2] # 
        ts = np.linspace(first, last, sample)
        self.P2 = np.array([self(t) for t in ts])
        return self.P2
 
    def plot(self, sample):
        P2 = self.resample(sample)

        # 创建一个三维图像
        fig = plt.figure(figsize=[6, 8], dpi=96)
        ax = fig.add_subplot(111, projection='3d')  # 设置为3D坐标系

        # 绘制控制点和曲线
        ax.plot(self.P.T[0], self.P.T[1], self.P.T[2], 'b-.', label='control points')  # 控制点
        ax.plot(P2.T[0], P2.T[1], P2.T[2], color='red', marker='.', markersize=1, label='curve')  # 曲线
        ax.scatter(self.P.T[0], self.P.T[1], self.P.T[2], color='b', label='control points')  # 控制点散点

        ax.set_aspect('auto')
        ax.legend()
        
        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z') 
        plt.tight_layout()
        plt.show()
 
def main():
    P = np.array([[0, 1, 2],
                [1, 0, 3],
                [0, 2, 2],
                [3, 0, 3],
                [3, 1, 1],
                [2, 4, 6],
                ],) 
    W = [1] * (len(P) - 1) + [1,] 
    p = 3  # 3阶
    n = P.shape[0] - 1  # 从 0 - n, P的长度是n+1
    m = n + p + 1



    # clame
    # 节点向量
    T1 = [0]*p + np.linspace(0, 1, m + 1 - 2 * p).tolist() +  [1] * p  # 从0-m, 个数是 m+1个 
    curve = NURBS.Curve() 
    curve.degree = p
    curve.ctrlpts = P 
    curve.weights = W
    curve.knotvector = T1
    curve.delta = 0.01
    curve.vis = VisMPL.VisCurve3D() 

    c = NURBSCurve(T1, P, curve.weights)
    c.plot(int(1/curve.delta)) 
    #plt.savefig("nurbs-clame(our).png")  # 保存为图片
    plt.close()




    # close
    P = np.concatenate((P, P[:p+1]), axis=0)
    W = np.concatenate((W, W[:p+1]), axis=0)
    n = P.shape[0] - 1  # 从 0 - n, P的长度是n+1 
    m = n + p + 1
    T2 = np.linspace(0, 1, m + 1).tolist() # 从0-m, 个数是 m+1个 

    curve = NURBS.Curve()
    curve.degree = p
    curve.ctrlpts = P
    curve.weights = W
    curve.knotvector = T2
    curve.delta = 0.01
    curve.vis = VisMPL.VisCurve3D()  

    c = NURBSCurve(T2, P, curve.weights)
    c.plot(int(1/curve.delta)) 

    #plt.savefig("nurbs-close(our).png")  # 保存为图片
    plt.close()
 
 
if __name__ == '__main__': 
    main()