from Transformer import Attention
from Transformer import MultiHeadedAttention

import torch

def unit_test_attention(): 

	att = Attention()


	q = 2
	k = 3
	v = 5
	kv = 4
	N = 5
	batch_size = 2

	x_list = [torch.randn(batch_size, N, q, k), torch.randn(batch_size, N, kv, k), torch.randn(batch_size, N, kv, v)]

	y = att(*x_list)

	# print("y", att(keys=x_list[1], queries=x_list[0],values=x_list[2], mask=True))
	print("y", y.shape , "== size:",   N,"*",batch_size, ",", q, ",", v)


unit_test_attention()


def unit_mult_attention(): 


unit_mult_attention()
