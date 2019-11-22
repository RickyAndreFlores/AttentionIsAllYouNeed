from Transformer import Attention
from Transformer import MultiHeadedAttention
import torch

	
seq = 3
batch_size = 2
embedding_size:int = 512
num_heads:int = 6
depth_q: int = 64
depth_k: int = 64
depth_v: int = 64

def unit_test_attention(): 

	att = Attention()


	x_list = [torch.randn(batch_size, num_heads, seq, depth_k), torch.randn(batch_size, num_heads, seq, depth_k), torch.randn(batch_size, num_heads, seq, depth_v)]

	y = att(*x_list)

	# print("y", att(keys=x_list[1], queries=x_list[0],values=x_list[2], mask=True))
	print("y", y.shape , "== size:",   num_heads,"*",batch_size, ",", seq, ",", depth_v)

	return y

# unit_test_attention()


def unit_mult_attention(): 

	multi = MultiHeadedAttention()

	previous = torch.randn(batch_size, seq, embedding_size)

	y = multi(previous)

	# print("result", y)

unit_mult_attention()
