
import numpy as np 
import matplotlib.pyplot as plt
class MLP:
    def __init__(self, hiddenSize, learningRate, max_lopp):
        self.hiddenNum=len(hiddenSize)
        self.hiddenSize=hiddenSize
        self.learningRate=learningRate
        self.max_lopp=max_lopp
        self.X=None
        self.Y=None
        self.w=None
        self.v=None
        self.hidw=None
        self.h=[]
        self.y_pred=None
    
    def sigmoid(self, z):
        # z = np.clip(z, -100, 100)  # 将 z 的值限制在 -100 到 100 之间
        return 1/(1+np.exp(-z))
    
    def preprocess(self,X,Y):
        # np.random.seed(0)
        X=np.insert(X,0,1,axis=1)
        self.X=X
        self.Y=Y
        self.v=np.random.rand(self.hiddenSize[0],self.X.shape[1])
        self.w=np.random.rand(self.Y.shape[1],self.hiddenSize[-1]+1)
        self.hidw=[]
        for i in range (1,self.hiddenNum):
            self.hidw.append(np.random.rand(self.hiddenSize[i],self.hiddenSize[i-1]+1))
            
    def forward(self):
        for i in range (self.hiddenNum):
            if i==0:
                self.h.append(self.sigmoid(np.dot(self.X,self.v.T)))
            else:
                self.h.append(self.sigmoid(np.dot(np.insert(self.h[i-1],0,1,axis=1),self.hidw[i-1].T)))
        self.y_pred=self.sigmoid(np.dot(np.insert(self.h[-1],0,1,axis=1),self.w.T))
        
    def loss(self):
        err=self.y_pred-self.Y
        total_err = np.sum(np.square(err))/2  # 计算总的MSE
        return total_err
    
    def gradient(self):
        err=self.y_pred-self.Y
        grad_out=err
        grad_z=self.y_pred*(1-self.y_pred)
        self.grad_w=np.dot(np.insert(self.h[-1],0,1, axis=1).T,grad_out*grad_z).T
        self.grad_hidw=[]
        if self.hiddenNum==1:
            grad_h=np.dot(grad_out*grad_z,self.w[:,1:])
            grad_z=self.h[0]*(1-self.h[0])
            self.grad_v=np.dot(self.X.T,grad_h*grad_z).T
        else:
            for i in range(self.hiddenNum-1,0,-1):
                if i == self.hiddenNum-1:
                    grad_h=np.dot(grad_out*grad_z,self.w[:,1:])
                else:
                    grad_h=np.dot(grad_h_0,self.hidw[i][:,1:])
                grad_z=self.h[i]*(1-self.h[i])
                self.grad_hidw.append(np.dot(np.insert(self.h[i-1],0,1,axis=1).T,grad_h*grad_z).T)
                grad_h_0=grad_h
            grad_h=np.dot(grad_h_0,self.hidw[0][:,1:])
            grad_z=self.h[0]*(1-self.h[0])
            self.grad_v=np.dot(self.X.T,grad_h*grad_z).T
            
            
    def train(self):
        mse_values = []  # List to store MSE values for each iteration
        for _ in range (self.max_lopp+1):
            self.forward()
            self.gradient()
            self.w=self.w-self.learningRate*self.grad_w
            for i in range (self.hiddenNum-1):
                self.hidw[i]=self.hidw[i]-self.learningRate*self.grad_hidw[-1-i]
            self.v=self.v-self.learningRate*self.grad_v
            if _%10==0:
                print("Loop: ",_," MSE: ",self.loss())
                mse = self.loss()
                mse_values.append(mse)
        plt.plot(range(0, self.max_lopp + 1, 10), mse_values)
        plt.xlabel('Loop')
        plt.ylabel('Loss')
        plt.title('Loop-MSE Graph, learning rate = 0.01')
        plt.grid(True)
        plt.show()
        
        
    def predict(self, X):
        X=np.insert(X,0,1,axis=1)
        h=[]
        for i in range (self.hiddenNum):
            if i==0:
                h.append(self.sigmoid(np.dot(X,self.v.T)))
            else:
                h.append(self.sigmoid(np.dot(np.insert(h[i-1],0,1,axis=1),self.hidw[i-1].T)))
        y_pred=self.sigmoid(np.dot(np.insert(h[-1],0,1,axis=1),self.w.T))
        return y_pred
    
    def plot_boundary(self):
        x_min, x_max = self.X[:, 1].min()-0.5 , self.X[:, 1].max()+0.5 
        y_min, y_max = self.X[:, 2].min()-0.5 , self.X[:, 2].max()+0.5
        h = 0.01
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z=self.predict(np.c_[xx.ravel(), yy.ravel()])
        for i in range(len(Z)):
            if Z[i]>0.5:
                Z[i]=1
            else:
                Z[i]=0
        Z=Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
        plt.scatter(self.X[:,1],self.X[:,2],c=self.Y[:,0],cmap=plt.cm.Spectral)
        plt.show()
    
red_points=np.array([[0.5,0],[1,0],[3,4],[4,5],[1,2],[2,3],[2,-3],[1,-1],[0.5,-3],[-4,4.5],[-4.5,4],])
blue_points=np.array([[-1,-1],[-2,-3],[-2,3],[-1,2],[-4,-4],[-4.5,-4],[-3,-4],[-4,-3],[-3,-3],[-3,-2],[-2,-2],[1,1]])
input_data = np.vstack((red_points, blue_points))
# print(input_data)
output_data = np.vstack((np.ones((red_points.shape[0], 1)), np.zeros((blue_points.shape[0], 1)))).reshape(23,1)
# print(output_data)

model=MLP([20],0.01,100000)
model.preprocess(input_data,output_data)
print('initial w is: ',model.w)
print('initial v is',model.v)
model.train()
model.plot_boundary()
print(len(model.grad_hidw))
print(len(model.hidw))


