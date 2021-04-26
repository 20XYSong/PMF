import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

# # 评分信息
# rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
# ratings = pd.read_csv('./ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
# # 电影信息
# mnames = ['movie_id', 'title', 'genres']
# movies = pd.read_csv('./ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python')
# attribute_name = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
# 				  'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
# 				  'Mystery', 'Romance','Sci-Fi', 'Thriller','War', 'Western']
# # 用户信息
# u_col_name = ['user_id', 'Gender', 'Age', 'Occupation', 'Zip-code']
# user_info = pd.read_csv('./ml-1m/users.dat',sep='::', names=u_col_name, engine='python')
#
#
# # 下面开始整合数据
# def attr2vec(attribute):	# attribute 是字符串，现将其转为向量
# 	initial_attr_vec = list(range(0, 35, 2))
# 	attr_vec = initial_attr_vec.copy()
# 	str_vec = attribute.split(sep='|')
# 	for s in str_vec:
# 		ind = attribute_name.index(s)	# 获取字符串s在属性名称中的索引位置
# 		attr_vec[ind] = initial_attr_vec[ind]+1	# 属性按“否是否是”排列
# 		pass
# 	return attr_vec
#
# movies['attr_vec_str'] = 'none'	# 为movies新建一列以储存属性向量（字符串类型）
# for index,item in movies.iterrows():
# 	attribute = item['genres']
# 	attr_vec = attr2vec(attribute)
# 	movies.iat[index,3] = str(attr_vec)	# pandas中修改值
# 	# movies['attr_vec_str'][index] = str(attr_vec)
# 	# movies.replace(a, b, inplace=True)
# 	pass
#
# integ_data = pd.merge(ratings,movies,how='inner', left_on='movie_id', right_on='movie_id')
# integ_data['like or not'] = 0
# integ_data.loc[integ_data.rating > 3, 'like or not'] = 1	# 将评分大于等于4的认为是喜欢；否则作为负样例
# final_data = integ_data.drop(columns=['timestamp', 'title', 'genres'])
# # eval(integ_data['attr_vec_str'][0])
# # final_data.to_csv('./data_all.csv',index=False)
#
#
# # TO get user_embeding matrix(18_dim vec):
# total_user_num = len(user_info)
# user_emb_mat = np.zeros([total_user_num,18])
# for t in range(len(integ_data)):
# 	u_i = integ_data['user_id'][t] - 1	# 因为dataframe中user_id的是从1开始的
# 	attr_arry = np.array(eval(integ_data['attr_vec_str'][t])) # an ndarray
# 	ind_of_attr = np.where(attr_arry % 2 != 0)[0]
# 	# print(ind_of_attr)
# 	for j in ind_of_attr:
# 		user_emb_mat[u_i,j]=user_emb_mat[u_i,j]+1	# count the nums of watching such attr movie
# 	pass
# pass
# # 将矩阵归一化：
# for k in range(total_user_num):
# 	sum_num = sum(user_emb_mat[k])
# 	user_emb_mat[k] = user_emb_mat[k]/sum_num
# 	pass
# user_emb_frame = pd.DataFrame(user_emb_mat)
# # user_emb_frame.to_csv('./user_present.csv', index=False, header=None)

# # 划分测试训练集，并导出为对应csv
# u_emb_df = pd.read_csv('./user_present.csv',header=None)
# data_total_df = pd.read_csv('./data_all.csv')
# data_total_df['like or not'] = data_total_df['like or not'].astype(float)
# # 划分测试训练集时应注意分层划分, but 用这个函数会出来warning：
# def train_test_sp(data, target, test_size = 0.2):
#     """
#     简单分层抽样
#     基于 sklearn.model_selection.StratifiedShuffleSplit
#     param data: 数据集
#     param target: 分层拆分比例依据
#     param test_size: 测试集比例 float
#     """
#     split_ned = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
#     for train_index, test_index in split_ned.split(data, data[target]):
#             strat_train_set = data.loc[train_index]
#             strat_test_set = data.loc[test_index]
#     return strat_train_set, strat_test_set
# train_data, test_data = train_test_sp(data_total_df, 'like or not')
# test_data.to_csv('./test_data.csv',index=False)
# train_data.to_csv('./train_data.csv',index=False)

"""Note: 是否需要将类中的train方法拿出来？因为放在类中貌似只能进行单一样本的更新，不能进行mini-batch的更新"""


class Discriminator(nn.Module):	# 注意输入判别器是user-item对输入，因此既需要user_embedding也需要相应的item_attribute
	def __init__(self, attribute_embed_num, hidden_dim, step_size, decay_rate, bat_size):
		super().__init__()
		# 在判别器中同样也需要对item_attr做embedding
		self.attribute_num = 18
		self.attribute_embed_num = attribute_embed_num
		self.hidden_dim = hidden_dim
		self.initial_attr_mat = nn.init.xavier_normal_(
			torch.Tensor(2 * self.attribute_num, self.attribute_embed_num))
		self.use_emb_dim = self.attribute_num
		self.bat_size = bat_size

		self.model = nn.Sequential(
			nn.Linear(self.attribute_num*self.attribute_embed_num+self.use_emb_dim,
					  self.hidden_dim),
			nn.Tanh(),
			nn.Linear(self.hidden_dim,self.hidden_dim),
			nn.Tanh(),
			nn.Linear(self.hidden_dim,1),
			# nn.Linear(self.hidden_dim,self.use_emb_dim),	# 让输出层也是18维
			nn.Sigmoid()
		)
		self.loss_function = nn.BCELoss()
		# 创建优化器，torch将可学习的参数传给了self.paramets()，于是我们可以通过self.parames()来访问学习参数
		self.optimiser = torch.optim.Adam(self.parameters(), lr=step_size, weight_decay=decay_rate)
		self.counter = 0  # 用于记录训练过程
		self.loss_progress = []  # 用于记录损失，不过是考虑的每10个样本增加一个损失值
		self.current_loss = 0 #记录当前训练时期的损失
		pass

	def attri_embed_query(self, attr_vec_str):  # 用于计算属性embed后的矩阵，并拉直为18*attr_embed_num大小的tensor, 这才能作为网络输入
		if type(attr_vec_str) == str:
			attr_vec = eval(attr_vec_str)  # to a list
			attr_tensor = torch.IntTensor(attr_vec)  # to a tensor
			mat_query = torch.nn.functional.embedding(attr_tensor, self.initial_attr_mat)  # torch中的embedding，或者说矩阵查询；构成最终的生成器输入
			longvec = torch.reshape(mat_query, (-1, self.attribute_embed_num * self.attribute_num))  # 拉直
			attr_input = longvec.reshape((self.attribute_embed_num * self.attribute_num,))
		else:	 # 当输入网络的是batch数据不是一条一条的数据时
			attr_input = torch.zeros((self.bat_size, self.attribute_embed_num * self.attribute_num))
			for inde, st in enumerate(attr_vec_str):
				st_vec = eval(st)
				st_tensor = torch.IntTensor(st_vec)
				mat_query = torch.nn.functional.embedding(st_tensor, self.initial_attr_mat)
				longvec_st = torch.reshape(mat_query, (-1, self.attribute_embed_num * self.attribute_num))
				st_input = longvec_st.reshape((self.attribute_embed_num * self.attribute_num,))
				attr_input[inde] = st_input
				pass
		return attr_input

	def connect(self, user_emb, attr_emb):	 # 连接user_emb和item_attr_emb,这才能作为输入对
		input_vec = torch.cat([user_emb, attr_emb], -1)
		return input_vec

	def forward(self, use_emb, item_attr):	 # 同上述item_attr只需要是原始str类型，不用转换
		attr_emb = self.attri_embed_query(item_attr)
		input_vec = self.connect(use_emb, attr_emb)
		return self.model(input_vec)

	def train(self, user_emb, item_attr, targets):
		# 由网络计算输出
		outputs = self.forward(user_emb, item_attr)

		loss = self.loss_function(outputs, targets)
		self.current_loss = loss.item()
		# 每进来一批(batch)样本，counter+1，
		self.counter += 1
		if self.counter % 10 == 0:
			self.loss_progress.append(self.current_loss)
			pass
		if self.counter % 1000 == 0:
			print("counter = ", self.counter, 'discriminator loss:', self.current_loss)
			pass
		# 优化过程：
		self.optimiser.zero_grad()  # 相当于每一轮迭代更新时，需要将以前计算的梯度归零
		loss.backward()  # 从loss函数中计算网络的梯度，并反向传播
		self.optimiser.step()  # 用上述由loss计算的梯度来更新网络的学习参数
		pass

	def plot_progress(self):
		df_loss = pd.DataFrame(self.loss_progress, columns=['loss'])
		df_loss.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.',
					 grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
		pass
	pass


class Generator(nn.Module):
	def __init__(self, attribute_embed_num, hidden_dim, step_size, decay_rate_g, bat_size):
		super().__init__()

		self.attribute_num = 18
		self.attribute_embed_num = attribute_embed_num # 属性向量嵌入时的维数
		self.hidden_dim = hidden_dim	  # 隐层的节点数
		self.initial_attr_mat = nn.init.xavier_normal_(
			torch.Tensor(2 * self.attribute_num, self.attribute_embed_num))	# 初始化属性的embed矩阵
		self.use_emb_dim = self.attribute_num
		# self.attri_embed_query()	# 自动计算属性的embed，以方便作为网络输入
		self.bat_size = bat_size
		self.current_loss = 0

		self.model = nn.Sequential(
			nn.Linear(self.attribute_num*self.attribute_embed_num, self.hidden_dim),
			nn.Tanh(),
			nn.Linear(self.hidden_dim, self.hidden_dim),
			nn.Tanh(),
			nn.Linear(self.hidden_dim, self.use_emb_dim),
			nn.Tanh()
		)

		self.optimiser = torch.optim.Adam(self.parameters(), lr=step_size, weight_decay=decay_rate_g)
		self.counter = 0  # 用于记录训练过程
		self.loss_progress = []  # 用于记录损失，不过是考虑的每10个样本增加一个损失值
		pass

	def attri_embed_query(self, attr_vec_str):	 # 用于计算属性embed后的矩阵，并拉直为18*attr_embed_num大小的tensor, 这才能作为网络输入
		if type(attr_vec_str) == str:
			attr_vec = eval(attr_vec_str)	         # to a list
			attr_tensor = torch.IntTensor(attr_vec)  # to a tensor
			mat_query = torch.nn.functional.embedding(attr_tensor, self.initial_attr_mat)  # torch中的embedding，或者说矩阵查询；构成最终的生成器输入
			longvec = torch.reshape(mat_query, (-1, self.attribute_embed_num*self.attribute_num))	# 拉直
			attr_input = longvec.reshape((self.attribute_embed_num*self.attribute_num,))
		else:
			attr_input = torch.zeros((self.bat_size, self.attribute_embed_num*self.attribute_num))
			for inde, st in enumerate(attr_vec_str):
				st_vec = eval(st)
				st_tensor = torch.IntTensor(st_vec)
				mat_query = torch.nn.functional.embedding(st_tensor, self.initial_attr_mat)
				longvec_st = torch.reshape(mat_query, (-1, self.attribute_embed_num * self.attribute_num))
				st_input = longvec_st.reshape((self.attribute_embed_num * self.attribute_num,))
				attr_input[inde] = st_input
				pass
		return attr_input

	def forward(self, inputs):		# 同理，inputs只需是str类型，还不用转换
		real_input = self.attri_embed_query(inputs)
		return self.model(real_input)

	def train(self, discriminator, inputs, targets):
		g_output = self.forward(inputs)   					# 由inputs这个item属性生成的用户表达
		d_output = discriminator.forward(g_output, inputs)   # 注意输入给判别器的是用户表达+item属性
		loss = discriminator.loss_function(d_output, targets)
		self.current_loss = loss.item()
		# 每训练一批(batch), counter+1
		self.counter += 1
		if self.counter % 10 == 0:
			self.loss_progress.append(loss.item())
			pass
		if self.counter % 1000 == 0:
			print("generator loss:", loss.item())
			pass

		self.optimiser.zero_grad()
		loss.backward()
		self.optimiser.step()
		pass

	def plot_progress(self):
		loss_df = pd.DataFrame(self.loss_progress, columns=['gen_loss'])
		loss_df.plot(ylim=(0), figsize=(16, 8), alpha=0.1, marker='.',
					 grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
		pass


# 建立数据集对象：
class Dataset_ml_1m(Dataset):
	def __init__(self, data_with_attr):    # 就是将数据赋给Dataset_ml_1m中的属性
		self.data = data_with_attr
		# self.u_emb_data = user_emb_data
		pass

	# 获取样本数量的方法
	def __len__(self):
		num_interact = len(self.data)
		# num_user = len(self.u_emb_data)
		return len(self.data)

	# 获取某一个样本的商品属性(字符串即可，不用转化为tensor)、对应user的表达(需要转化为一个tensor), label(是否喜欢)
	def __getitem__(self, index):
		u_id = self.data.iloc[index, 0]
		# 获取index对应样本的label: like or not; 以此判定正负样例
		like = self.data.iloc[index, 4]
		label = torch.FloatTensor([like])  # label也需要转化为一个tensor数据类型才能计算损失
		# label = torch.FloatTensor(label)
		# target = torch.zeros((10))
		# target[label.astype(int)] = 1.0

		attr_str = self.data.iloc[index, 3]
		# ui_emb = u_emb_df.iloc[u_id, ]
		# user_emb_tensor = torch.FloatTensor(ui_emb.values)	 # 需要将user表达转化为一个tensor，才能输入网络中
		return u_id, attr_str, label
	pass

u_emb_df = pd.read_csv('./user_present.csv',header=None)
train_data = pd.read_csv('./train_data.csv')
test_data = pd.read_csv('./test_data.csv')
train_dataset = Dataset_ml_1m(train_data)
test_dataset = Dataset_ml_1m(test_data)


# set hyper-parameters:
bat_size = 64
attr_emb_nums = 5
hidden_dims = 100
step_sizes = 1e-4
decay_rate = 1.0   # weight_decay in adam optimizer
decay_rate_g = 0   # for generator
epochs = 3 		   # 整个样本（full-batch）用几次
mini_batch_sup = 2000

train_loader = DataLoader(train_dataset, batch_size=bat_size, shuffle=True)

# training GAN with single sample and batch-size sample:
dis = Discriminator(attr_emb_nums, hidden_dims, step_sizes, decay_rate, 1)
gen = Generator(attr_emb_nums, hidden_dims, step_sizes, decay_rate_g, 1)

dis_bat = Discriminator(attr_emb_nums, hidden_dims, step_sizes, decay_rate, bat_size)
gen_bat = Generator(attr_emb_nums, hidden_dims, step_sizes, decay_rate_g, bat_size)
# check network:
# uid5, attr_str5, label5 = train_dataset[5]
# ui_emb5 = u_emb_df.iloc[uid5,]
# user_emb_tensor5 = torch.FloatTensor(ui_emb5.values)
# attr_emb_longvec = dis.attri_embed_query(attr_str5)
# torch.cat([user_emb_tensor5, attr_emb_longvec], -1)
# an_output = dis.forward(user_emb_tensor5,attr_str5)
# print(label5, '\n', an_output)

"""Start training"""

# training in single sample
# for k in range(epochs):
# 	print("training epoch:", k + 1, 'of', epochs)
# 	for u_id, attr_str, label in train_dataset:
# 		ui_emb = u_emb_df.iloc[u_id-1, :]
# 		user_emb_tensor = torch.FloatTensor(ui_emb.values)	 # 需要将user表达转化为一个tensor，才能输入网络中
# 		# break
# 		dis.train(user_emb_tensor, attr_str, label)  						 # 送入真实样例，有正例有负例
# 		dis.train(gen.forward(attr_str).detach(), attr_str, torch.FloatTensor([0.0]))  # 送入生成样例，其标签对于判别器来说是0
# 		gen.train(dis, attr_str, torch.FloatTensor([1.0]))					 # 训练生成器，对于生成器来说，它期望的结果是判别器识别为1.0
# 		pass
# 	pass
# dis.plot_progress()
# gen.plot_progress()
# torch.save(dis, './Discriminator_1batch.pkl')
# torch.save(gen, './Generator_1batch.pkl')
# dis_v1 = torch.load('./Discriminator_1batch.pkl')
# gen_v1 = torch.load('./Generator_1batch.pkl')


# training in batch samples
epochs_for_bat = 2
fake_u_dis_label_bat = torch.FloatTensor([0.0]*bat_size).reshape((bat_size, 1))
fake_u_gen_label_bat = torch.FloatTensor([1.0]*bat_size).reshape((bat_size, 1))
flag = 0
for t in range(epochs_for_bat):
	print("training epoch:", t + 1, 'of', epochs_for_bat)
	for u_id_bat, attr_str_bat, label_bat in train_loader:
		if len(u_id_bat) != bat_size:
			res_sample_num = len(u_id_bat)
			print(res_sample_num)
			fake_u_dis_label_bat = torch.FloatTensor([0.0] * res_sample_num).reshape((res_sample_num, 1))
			fake_u_gen_label_bat = torch.FloatTensor([1.0] * res_sample_num).reshape((res_sample_num, 1))
			dis_bat.bat_size = res_sample_num
			gen_bat.bat_size = res_sample_num
			flag = 1
			pass

		ui_emb_bat = u_emb_df.iloc[u_id_bat-1, :]
		ui_emb_tensor_bat = torch.FloatTensor(ui_emb_bat.values)

		dis_bat.train(ui_emb_tensor_bat, attr_str_bat, label_bat)
		fake_user_emb_bat = gen_bat.forward(attr_str_bat).detach()
		dis_bat.train(fake_user_emb_bat, attr_str_bat, fake_u_dis_label_bat)
		gen_bat.train(dis_bat, attr_str_bat, fake_u_gen_label_bat)

		if flag == 1:
			dis_bat.bat_size = bat_size
			gen_bat.bat_size = bat_size
			fake_u_dis_label_bat = torch.FloatTensor([0.0] * bat_size).reshape((bat_size, 1))
			fake_u_gen_label_bat = torch.FloatTensor([1.0] * bat_size).reshape((bat_size, 1))
			flag = 0
			pass

dis_bat.plot_progress()
gen_bat.plot_progress()
torch.save(dis_bat, './Discriminator_64batch.pkl')
torch.save(gen_bat, './Generator_64batch.pkl')
dis_v2 = torch.load('./Discriminator_64batch.pkl')
gen_v2 = torch.load('./Generator_64batch.pkl')

