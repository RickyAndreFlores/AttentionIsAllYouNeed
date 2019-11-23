# from Transformer import Attention
# from Transformer import MultiHeadedAttention
# from Transformer import FeedForward
# from Transformer import Encoder
# from Transformer import positional_encodings
# from Transformer import Masked_MultiHeadedAttention

import torch



seq = 3
batch_size = 2
embedding_size:int = 512
num_heads:int = 6
depth_q: int = 64
depth_k: int = 64
depth_v: int = 64




import torch

def unit_test_attention(): 
	from Transformer import Attention

	att = Attention()


	x_list = [torch.randn(batch_size, num_heads, seq, depth_k), torch.randn(batch_size, num_heads, seq, depth_k), torch.randn(batch_size, num_heads, seq, depth_v)]

	y = att(*x_list)

	# print("y", att(keys=x_list[1], queries=x_list[0],values=x_list[2], mask=True))
	print("y", y.shape , "== size:",   num_heads,"*",batch_size, ",", seq, ",", depth_v)

	return y

# unit_test_attention()


def unit_mult_attention(): 
	from Transformer import MultiHeadedAttention

	multi = MultiHeadedAttention()

	previous = torch.randn(batch_size, seq, embedding_size)

	y = multi(previous)

	# print("result", y)

# unit_mult_attentions()


def unit_feed_forward(): 
	from Transformer import FeedForward

	ff = FeedForward()
 
	y = ff(previous)

# unit_feed_forward()

def unit_encoder(): 
	from Transformer import Encoder

	en = Encoder()

	emeddings = torch.randn(batch_size, seq, embedding_size)
	
	out = en(emeddings)

# unit_encoder()


def unit_pos_encoder():

	from Transformer import positional_encodings

	em = positional_encodings(5, 11)


	out = em( torch.LongTensor([1,2,3,4]) )
	print(out)
# unit_pos_encoder()


def unit_masked(): 
	from Transformer import Masked_MultiHeadedAttention

	test = Masked_MultiHeadedAttention()
	print("is nn.Module", isinstance(test, torch.nn.Module) )

	previous = torch.randn(batch_size, seq, embedding_size)

	y = test(previous)
	

unit_masked()
