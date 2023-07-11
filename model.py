import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.graph_operation_layer import GATEncoder, GAT,PositionWiseFFN##new

from layers.seq2seq import Seq2Seq, EncoderRNN,Transformer

import numpy as np 

torch.cuda.empty_cache()##new

class Model(nn.Module):
	def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
		super().__init__()

		# load graph
		self.graph = Graph(**graph_args)###graph_args={'max_hop':2, 'num_node':120}
		A = np.ones((graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node']))
		# A_big=

		# build networks
		spatial_kernel_size = np.shape(A)[0]
		temporal_kernel_size = 5 #9 #5 # 3
		kernel_size = (temporal_kernel_size, spatial_kernel_size)##(5,3)

		# best
		self.st_gcn_networks = nn.ModuleList((
			nn.BatchNorm2d(in_channels),
			Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		))

		self.mlp1=PositionWiseFFN(16,32)##new
		self.gat=GAT(n_units=[32,16,32],n_heads=[4,1])##new
		self.mlp2 = PositionWiseFFN(32, 64)##new
		self.linear_for_TF=nn.Linear(2,64)
		# self.transformer_model = nn.Transformer(d_model=64, nhead=16, num_encoder_layers=12, batch_first=True)  ##new
		# self.linear_for_output = nn.LazyLinear(2)  ### need to be improved in the future
		self.transformer_model=Transformer(num_tokens=2, dim_model=64, num_heads=16, num_encoder_layers=12, num_decoder_layers=12, dropout_p=0.1)#new new


		# initialize parameters for edge importance weighting
		if edge_importance_weighting:
			self.edge_importance = nn.ParameterList(
				[nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
				)
			self.edge_importance_biggraph=nn.Parameter(torch.Tensor(720,720))###biggraph can also be dynamical
		else:
			self.edge_importance = [1] * len(self.st_gcn_networks)

		self.num_node = num_node = self.graph.num_node
		self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate
		self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)


	def reshape_for_lstm(self, feature):
		##(N, C, T, V)->((N*V, T, C)
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() 
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C) 
		return now_feat###

	def reshape_from_lstm(self, predicted):
		# predicted (N*V, T, C)
		NV, T, C = predicted.size()
		now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def forward(self, pra_x, pra_A,pra_A_big, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
		x = pra_x##(N, C, T, V)=(N, 4, 6, 120)
		##A: (N, 3(max_hop+1), V, V)=(N, 3(max_hop+1),120,120)

		# forwad
		for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
			if type(gcn) is nn.BatchNorm2d:
				x = gcn(x)
			else:
				x, _ = gcn(x, pra_A + importance)
				# print('importance shape in model.py:',importance.shape)

		##big graph attention computation
		# print('in model.py')
		# print('x shape before put into mlp1:',x.shape)
		pra_x_process = pra_x.permute(0,3,2,1)# (N, C, T, V)->(N,V,T,C)
		mlp1_output =self.mlp1(pra_x_process)
		# print('mlp1 output shape:',mlp1_output.shape)
		gat_input=mlp1_output.permute(0, 3, 2, 1)##(N,V,T,C)->(N,C,T,V)
		# print('gat_input shape',gat_input.shape)##
		# print('pra_A_big shape in model.py:', pra_A_big.shape)##(N,T*V,T*C)=(N,720,720)

		# print('self.edge_importance_biggraph',self.edge_importance_biggraph)
		# gat_output = self.gat(gat_input,pra_A_big+self.edge_importance_biggraph)##(N,T*V,C)
		gat_output = self.gat(gat_input, pra_A_big)  ##(N,T*V,C)
		# print('gat output shape:',gat_output.shape)

		mlp2_output = self.mlp2(gat_output)
		# print('mlp2_output shape:',mlp2_output.shape)
		x_big=mlp2_output.reshape(mlp2_output.shape[0],mlp2_output.shape[2],pra_x.shape[2],-1)##(N,T*V,C)->(N,C,T,V)
		# print('x_big shape:',x_big.shape)##(N,C, T, V)=(N,64,6,120)

		#####add method: add x_big to x
		x=x+x_big## next to designed a gate mechanism

		## gate method
		# x=weighted*x+(1-weighted)*x_big
		### cat method
		# x_cat=torch.cat((x,x_big),1)
		# x_cat=x_cat.permute(0, 3, 2, 1)##(N,C,T,V)->(N,V, T, C)
		# mlp3_output=self.mlp3(x_cat)
		# x=mlp3_output.permute(0, 3, 2, 1)##(N,V, T, C)->(N,C,T,V)

		# prepare for seq2seq lstm model
		graph_conv_feature = self.reshape_for_lstm(x)##(N*V, T, C)

		last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> [(N*V, T, C)]


		#### seq2seq method##########
		if pra_teacher_forcing_ratio>0 and type(pra_teacher_location) is not type(None):
			pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)
		#### now_predict.shape = (N, T, V*C)
		# print('graph_conv_feature.shape in model.py',graph_conv_feature.shape)
		# print('last_position.shape in model.py',last_position.shape)
		now_predict_car = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict_car = self.reshape_from_lstm(now_predict_car) # (N, C, T, V)
		
		now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict_human = self.reshape_from_lstm(now_predict_human) # (N, C, T, V)
		
		now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict_bike = self.reshape_from_lstm(now_predict_bike) # (N, C, T, V)
		
		now_predict = (now_predict_car + now_predict_human + now_predict_bike)/3.


		# ##############transformer method #########################
		# pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)
		# # future_trajectory = self.reshape_for_lstm(pra_teacher_location)  ##(N*V, T, 2)
		# pra_teacher_location = self.linear_for_TF(pra_teacher_location)  ###(N*V, T, 2)->(N*V,T,64)
		# # print('in model.py line 154')
		# # print('graph_conv_feature shape: {}, pra_teacher_location shape: {}'.format(graph_conv_feature.shape,pra_teacher_location.shape))
		# sequence_length = pra_teacher_location.size(1)
		# tgt_mask = self.transformer_model.get_tgt_mask(sequence_length)## device
		# # print('tgt_mask device before to function in model.py',tgt_mask.device)
		# tgt_mask = tgt_mask.to('cuda:0')##dev = 'cuda:0' in main.py
		# # print('tgt_mask device after to function in model.py', tgt_mask.device)
		# now_predict_car_TF=self.transformer_model(src=graph_conv_feature, tgt=pra_teacher_location,tgt_mask=tgt_mask)
		# now_predict_car_TF=self.reshape_from_lstm(now_predict_car_TF)

		# now_predict_human_TF=self.transformer_model(src=graph_conv_feature, tgt=pra_teacher_location,tgt_mask=tgt_mask)
		# now_predict_human_TF=self.reshape_from_lstm(now_predict_human_TF)

		# now_predict_bike_TF=self.transformer_model(src=graph_conv_feature, tgt=pra_teacher_location,tgt_mask=tgt_mask)
		# now_predict_bike_TF=self.reshape_from_lstm(now_predict_bike_TF)

		# now_predict=(now_predict_car_TF + now_predict_human_TF + now_predict_bike_TF)/3.##(N, 2, T, V)





		return now_predict 

if __name__ == '__main__':

	model = Model(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
	print(model)
