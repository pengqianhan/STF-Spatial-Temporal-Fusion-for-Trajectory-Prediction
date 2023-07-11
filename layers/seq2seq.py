import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

####################################################
# Seq2Seq LSTM AutoEncoder Model
# 	- predict locations
####################################################
class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, isCuda=True):
		super(EncoderRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.isCuda = isCuda
		# self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.lstm = nn.GRU(input_size, hidden_size*30, num_layers, batch_first=True)
		
	def forward(self, input):
		output, hidden = self.lstm(input)
		return output, hidden

class DecoderRNN(nn.Module):
	def __init__(self, hidden_size, output_size, num_layers, dropout=0.5, isCuda=True):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
		self.isCuda = isCuda
		# self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
		self.lstm = nn.GRU(hidden_size, output_size*30, num_layers, batch_first=True)

		#self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=dropout)
		self.linear = nn.Linear(output_size*30, output_size)
		self.tanh = nn.Tanh()
	
	def forward(self, encoded_input, hidden):
		decoded_output, hidden = self.lstm(encoded_input, hidden)
		# decoded_output = self.tanh(decoded_output)
		# decoded_output = self.sigmoid(decoded_output)
		decoded_output = self.dropout(decoded_output)
		# decoded_output = self.tanh(self.linear(decoded_output))
		decoded_output = self.linear(decoded_output)
		# decoded_output = self.sigmoid(self.linear(decoded_output))
		return decoded_output, hidden

class Seq2Seq(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, dropout=0.5, isCuda=True):
		super(Seq2Seq, self).__init__()
		self.isCuda = isCuda
		# self.pred_length = pred_length
		self.encoder = EncoderRNN(input_size, hidden_size, num_layers, isCuda)
		self.decoder = DecoderRNN(hidden_size, hidden_size, num_layers, dropout, isCuda)
	
	def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
		batch_size = in_data.shape[0]
		out_dim = self.decoder.output_size
		self.pred_length = pred_length

		outputs = torch.zeros(batch_size, self.pred_length, out_dim)
		if self.isCuda:
			outputs = outputs.cuda()

		encoded_output, hidden = self.encoder(in_data)
		
		decoder_input = last_location
		for t in range(self.pred_length):
			# encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
			now_out, hidden = self.decoder(decoder_input, hidden)
			now_out += decoder_input
			outputs[:,t:t+1] = now_out 
			teacher_force = np.random.random() < teacher_forcing_ratio
			decoder_input = (teacher_location[:,t:t+1] if (type(teacher_location) is not type(None)) and teacher_force else now_out)
			# decoder_input = now_out
		return outputs

####################################################
####################################################
##the code is from https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
class PositionalEncoding(nn.Module):
	def __init__(self, dim_model, dropout_p, max_len):
		super().__init__()
		# Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
		# max_len determines how far the position can have an effect on a token (window)

		# Info
		self.dropout = nn.Dropout(dropout_p)

		# Encoding - From formula
		pos_encoding = torch.zeros(max_len, dim_model)
		positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
		division_term = torch.exp(
			torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

		# PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
		pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

		# PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
		pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

		# Saving buffer (same as parameter without gradients needed)
		pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pos_encoding", pos_encoding)

	def forward(self, token_embedding: torch.tensor) -> torch.tensor:
		# Residual connection + pos encoding
		return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):
	"""
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

	# Constructor
	def __init__(
			self,
			num_tokens,
			dim_model,
			num_heads,
			num_encoder_layers,
			num_decoder_layers,
			dropout_p,
	):
		super().__init__()

		# INFO
		self.model_type = "Transformer"
		self.dim_model = dim_model

		# LAYERS
		self.positional_encoder = PositionalEncoding(
			dim_model=dim_model, dropout_p=dropout_p, max_len=5000
		)
		# self.embedding = nn.Embedding(num_tokens, dim_model)
		self.transformer = nn.Transformer(
			d_model=dim_model,
			nhead=num_heads,
			num_encoder_layers=num_encoder_layers,
			num_decoder_layers=num_decoder_layers,
			dropout=dropout_p,
			batch_first=True
		)
		self.out = nn.Linear(dim_model, num_tokens)#####num_tokens=out_dim

	def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
		# Src size must be (batch_size, src sequence length, feature_dim=dim_model)#new
		# Tgt size must be (batch_size, tgt sequence length,feature_dim=dim_model)#new

		# Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
		# src = self.embedding(src) * math.sqrt(self.dim_model)#delete
		# tgt = self.embedding(tgt) * math.sqrt(self.dim_model)#delete
		src = src * math.sqrt(self.dim_model)
		tgt = tgt * math.sqrt(self.dim_model)
		src = self.positional_encoder(src)
		tgt = self.positional_encoder(tgt)

		# We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
		# to obtain size (sequence length, batch_size, dim_model),#delete
		# src = src.permute(1, 0, 2)#delete
		# tgt = tgt.permute(1, 0, 2)#delete

		# Transformer blocks - Out size = (batch_size,sequence length ,num_tokens)#new
		transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask,
										   tgt_key_padding_mask=tgt_pad_mask)
		out = self.out(transformer_out)

		return out

	def get_tgt_mask(self, size) -> torch.tensor:
		# Generates a squeare matrix where the each row allows one word more to be seen
		mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix,item is boolean
		mask = mask.float()## boolean -> float
		mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
		mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

		# EX for size=5:
		# [[0., -inf, -inf, -inf, -inf],
		#  [0.,   0., -inf, -inf, -inf],
		#  [0.,   0.,   0., -inf, -inf],
		#  [0.,   0.,   0.,   0., -inf],
		#  [0.,   0.,   0.,   0.,   0.]]

		return mask

	def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
		# If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
		# [False, False, False, True, True, True]
		return (matrix == pad_token)
