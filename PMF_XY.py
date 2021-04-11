import pandas as pd
from sklearn import model_selection as ms
import numpy as np
import math
import time
import matplotlib.pyplot as plt
# header=['user_id','item_id','rating','timestamp']
# u_da=pd.read_csv('./ml-100k/u.data',sep='\t',names=header)
# I=np.zeros((u_da.shape[0],u_da.shape[0]))

# for row in u_da.itertuples(): # 这样读出来的每一行都是一个pandas数据格式，即Pandas(Index=0, user_id=196, item_id=242, rating=3, timestamp=881250949)
#     if row[1] not in user_set:
#         user_set[row[1]]=u_indx
#         row_ind=u_indx.copy()
#         u_indx+=1
#     if row[2] not in item_set:
#         item_set[row[2]]=i_indx
#         col_ind=i_indx
#         i_indx+=1
#     I[row_ind,col_ind]=1
#     M=u_indx
#     N=i_indx
# I=I[:M,:N]
"load data and creat indicator matrix"
file = open('./ml-100k/u.data')
data_read = file.read().splitlines()
data_read=[l.split('\t') for l in data_read]

def count_user_item(data):#计数用户数和item数，同时将数据化为int、float类型
    num_sample = len(data)
    k = 0
    user_set = {}
    item_set = {}
    u_count = 0
    i_count = 0
    for l in data:
        u, i, r, _ = l
        if u not in user_set:
            user_set[u] = u_count
            u_count += 1
        if i not in item_set:
            item_set[i] = i_count
            i_count += 1
        # row_ind = float(u).copy()
        # col_ind = float(i).copy()
        data[k] = [user_set[u], item_set[i], float(r)]  # 这时的userid和itemid相当于都是从0编号起走，且中间不会有遗漏
        k+=1
    M,N=u_count,i_count
    result=[M,N,num_sample,data]
    return result

'create indicator matrix and rating matrix'
[num_user_all,num_item_all,num_sample,data_all]=count_user_item(data_read)
train_data,test_data=ms.train_test_split(data_all,train_size=0.8)

maxi=max(num_item_all,num_user_all)
I = np.asmatrix(np.zeros((maxi,maxi)))
R=np.asmatrix(np.zeros((num_user_all,num_item_all)))
for row in train_data:          # 相当于只用了训练集数据的Rij，并用于之后的损失计算
    row_ind=row[0]
    col_ind=row[1]
    I[row_ind,col_ind]=1
    R[row_ind,col_ind]=row[2]
I=I[:num_user_all,:num_item_all]

class PMF:
    def __init__(self,M,N,K,sigma_square,sigma_u,sigma_v,learning_rate,iteration,train,test):
        self.num_user = M
        self.num_item=N
        self.latent_factor=K
        self.condi_variance=sigma_square
        self.sig_u=sigma_u
        self.sig_v=sigma_v
        self.lambda_u=sigma_square/sigma_u
        self.lambda_v=sigma_square/sigma_v
        self.stepsize=learning_rate
        self.iter_max=iteration
        self.test=test
        self.train=train

        # U=np.random.normal(0, sigma_u, (num_user, K))
        # V=np.random.normal(0, sigma_v, (num_item, K))
    def rmse_compute(self,data,U,V,t):
        bias_sum=0
        num_data=len(data)
        for rows in data:
            r_ind=rows[0]
            c_ind=rows[1]
            bias=rows[2]-np.dot(U[r_ind],V[c_ind])
            bias2=bias**2
            bias_sum+=bias2
        self.rmse=math.sqrt(bias_sum/num_data)
        print('at iter = {},rmse={}'.format(t,self.rmse))

    def GD_update(self,I,R):
        #生成参数先验,这里U、V都看做按行的多维正态
        U = np.zeros((self.num_user, self.latent_factor))
        V = np.zeros((self.num_item, self.latent_factor))
        for t in range(self.num_user):
            U[t] = np.random.multivariate_normal(np.zeros(self.latent_factor), self.sig_u * np.eye(self.latent_factor))
        for t in range(self.num_item):
            V[t] = np.random.multivariate_normal(np.zeros(self.latent_factor), self.sig_v * np.eye(self.latent_factor))
        #初始化目标函数obj和rmse
        err_mat=R-np.dot(U,V.T)
        err_mat2=np.multiply(err_mat,err_mat)
        obj=0.5*np.sum(np.multiply(err_mat2,I))+\
            0.5*self.lambda_u*np.sum(np.square(U))+0.5*self.lambda_v*np.sum(np.square(V))
        obj_vec=[obj]*(self.iter_max+1)
        self.rmse_compute(self.test,U,V,0)
        rmse_initial = self.rmse
        rmse_vec=[rmse_initial]*self.iter_max
        # 开始用全批量GD迭代
        for t in range(self.iter_max):
            obj_last=obj.copy()
            for i in range(self.num_user):
                err_u_vec=R[i]-np.dot(U[i],V.T)#误差行向量,是一个matrix,维数1682
                coef_u_mat=np.zeros((self.num_item,self.latent_factor))
                for j in range(self.num_item):
                    coef_u_mat[j]=I[i,j]*V[j] #乘出来是一个array向量（k,）
                err_derivative_ui=-err_u_vec @ coef_u_mat #乘出来是一行matrix格式，需要转换
                err_derivative_ui=np.asarray(err_derivative_ui).reshape((self.latent_factor,))
                gradient_ui=err_derivative_ui+self.lambda_u*U[i]#check一下乘法是否合理
                U[i]=U[i]-self.stepsize*gradient_ui
            for j in range(self.num_item):
                err_v_vec=R[:,j].T-np.dot(V[j],U.T)#同理，减出来应是一个943维的一行matrix
                coef_v_mat=np.zeros((self.num_user,self.latent_factor))
                for i in range(self.num_user):
                    coef_v_mat[i]=I[i,j]*U[i]
                err_derivative_vj=-err_v_vec @ coef_v_mat
                err_derivative_vj=np.asarray(err_derivative_vj).reshape((self.latent_factor,))
                gradient_vj=err_derivative_vj+self.lambda_v*V[j]
                V[j]=V[j]-self.stepsize*gradient_vj
            err_mat = R - np.dot(U, V.T)
            err_mat2 = np.multiply(err_mat, err_mat)
            obj = 0.5*np.sum(np.multiply(err_mat2, I))+\
                  0.5*self.lambda_u*np.sum(np.square(U))+0.5*self.lambda_v*np.sum(np.square(V))
            obj_vec[t+1]=obj
            rmse_t=self.rmse_compute(self.test,U,V,t+1)
            rmse_vec[t+1]=rmse_t
            if abs(obj_last-obj)<=1e-10:
                self.real_iter = t + 1
                break
            else:
                self.real_iter=self.iter_max

        self.U_learned=U
        self.V_learned=V
        self.loss_vec=obj_vec[:self.real_iter+1]
        self.loss_final=self.loss_vec[-1]
        self.rmse_vec=rmse_vec[:self.real_iter+1]
        self.rmse_final=self.rmse_vec[-1]

    def SGD_update(self,train):
        U = np.zeros((self.num_user, self.latent_factor))
        V = np.zeros((self.num_item, self.latent_factor))
        for t in range(self.num_user):
            U[t] = np.random.multivariate_normal(np.zeros(self.latent_factor), self.sig_u * np.eye(self.latent_factor))
        for t in range(self.num_item):
            V[t] = np.random.multivariate_normal(np.zeros(self.latent_factor), self.sig_v * np.eye(self.latent_factor))
        acc=1e-8
        #计算初始化obj以及rmse
        obj=0
        for tr in train:
            u_ind = tr[0]
            i_ind = tr[1]
            r_ij = tr[2]
            err_ij = r_ij - np.dot(U[u_ind], V[i_ind].T)
            obj_ij = 0.5 * err_ij ** 2 + 0.5 * self.lambda_u * np.linalg.norm(
                U[u_ind]) ** 2 + 0.5 * self.lambda_v * np.linalg.norm(V[i_ind]) ** 2
            obj = obj + obj_ij
        obj_vec_sgd=[obj]*self.iter_max
        self.rmse_compute(self.test, U, V, 0)
        rmse=self.rmse
        rmse_vec_sgd=[rmse]*self.iter_max
        uptime=0
        tol=0.01
        flag=0
        velo_ui=np.zeros(self.latent_factor)
        velo_vj=velo_ui
        # SGD update
        for t in range(self.iter_max):
            # rmse_last=rmse.copy()
            obj_last = obj.copy()
            obj=0
            for tr in train:
                u_ind = tr[0]
                i_ind = tr[1]
                r_ij = tr[2]

                err_ij = r_ij - np.dot(U[u_ind], V[i_ind].T)
                grad_ui=-err_ij*V[i_ind]+self.lambda_u*U[u_ind]
                grad_vj=-err_ij*U[u_ind]+self.lambda_v*V[i_ind]
                # plus momentum: stepsize=0.006; tol=0.01; decay rate=0.4; rmse=1.194
                # velo_ui=0.9*velo_ui+grad_ui
                # velo_vj=0.9*velo_vj+grad_vj
                # U[u_ind] = U[u_ind] - self.stepsize * velo_ui
                # V[i_ind] = U[u_ind] - self.stepsize * velo_vj

                U[u_ind] =U[u_ind]-self.stepsize * grad_ui
                V[i_ind] =U[u_ind]-self.stepsize * grad_vj

                obj_ij = 0.5 * err_ij ** 2 + 0.5 * self.lambda_u * np.linalg.norm(
                    U[u_ind]) ** 2 + 0.5 * self.lambda_v * np.linalg.norm(V[i_ind]) ** 2
                obj = obj + obj_ij
            obj_vec_sgd[t+1]=obj
            self.rmse_compute(self.test,U,V,t+1)
            rmse=self.rmse
            rmse_vec_sgd[t+1]=rmse #注意rmse和obj的的对象时不同的，rmse是对测试集而言，而obj是对训练集而言

            # if rmse_vec_sgd[t]-rmse_vec_sgd[t+1]<1e-5:
            #     flag=1
            # else:
            #     flag=0
            if abs(rmse_vec_sgd[t+1]-rmse_vec_sgd[t])<0.007:
                self.stepsize=0.8*self.stepsize+0.2*self.stepsize*flag #under sgd:decay=0.8;tol=0.007;stepsize=0.04
            print("stepsize:",self.stepsize)
            # if abs(obj-obj_last)<acc:  #因此最好不用这种终止方式，因为train的obj可以尽量小，但test的rmse可能因为过拟合而增大，所以需要下面的early stop
            #     self.real_iter = t + 1
            #     break
            if rmse_vec_sgd[t+1]>rmse_vec_sgd[t]:
                uptime+=1
            if uptime>4:
                self.real_iter = t + 1
                break

            # if abs(rmse_vec_sgd[t+1]-rmse_vec_sgd[t])<acc:#test的RMSE几乎不再下降时退出
            #     self.real_iter = t + 1
            #     break
        self.U_learned = U
        self.V_learned = V
        self.loss_vec = obj_vec_sgd[:self.real_iter + 1]
        self.loss_final = self.loss_vec[-1]
        self.rmse_vec = rmse_vec_sgd[:self.real_iter + 1]
        self.rmse_final = self.rmse_vec[-1]


if __name__ == '__main__':
    latent_fac_num,stepsize,sigma2,sigmau,sigmav,itermax\
        =5 , 0.04 , 0.001 , 0.01 , 0.01 , 10000
    # time_s=time.time()
    # pmf=PMF(num_user_all,num_item_all,latent_fac_num,0.001,0.1,0.1,0.005,10000,train_data,test_data)
    # pmf.GD_update(I,R)
    # time_e=time.time()
    # time_gd=round(time_e-time_s,2)

    time_s = time.time()
    pmf = PMF(num_user_all, num_item_all, latent_fac_num, sigma2, sigmav, sigmau, stepsize, itermax, train_data, test_data)
    pmf.SGD_update(train_data)
    time_e=time.time()
    time_sgd=round(time_e-time_s,2)

    plt.figure(dpi=120)
    iter_vec=list(range(pmf.real_iter+1))
    plt.plot(iter_vec, pmf.rmse_vec, label='RMSE({} s)'.format(time_sgd), linewidth=1.2)
    plt.xlabel('iterations')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.show()
