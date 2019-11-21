import torch
from torch import nn
from math import sqrt


# TODO 1conv linear for q,vk

class TensorPrep():

	@staticmethod
	def attention_get_dims(attention):

			
		def wrapper(*args, **kwargs): 
			"""
					In
								queries: [batch, heads, length_q, depth_k]
								keys:    [batch, heads, length_kv, depth_k]
								values:  [batch, heads, length_kv, depth_v]

					out

								queries: [batch * heads, length_q, depth_k]
								keys:    [batch * heads, length_kv, depth_k]
								values:  [batch * heads, length_kv, depth_v]
			"""

			# get last 2 dimensions 
			get_shape = lambda tensor : { "length": tensor.shape[-2], "depth":  tensor.shape[-1] } 

			#create dictionary of vectors and their last 2 dimension 
			names = ["queries", "keys", "values"]
			tensors = {names.pop(0): get_shape(x) for x in args if (type(x) == torch.Tensor)}

			# get keywords or replace
			for k,x in kwargs.items(): 
				if type(x) == torch.Tensor:

					tensors[k] = get_shape(x) 


			return attention(*args, **kwargs, dims = tensors)
		
		return wrapper


class Attention(nn.Module): 

	def __init__(self):
		""" 
		multiheaded, scaled, dot-product attention

		Args:
			queries: linear transformation of antecedent or previous output
			keys: linear transformation of antecedent or previous output
			values: linear transformation of antecedent or previous output
			mask:

		Input Shape: 
			queries: [batch, heads, length_q, depth_k] 
			keys:    [batch, heads, length_kv, depth_k] 
			values:  [batch, heads, length_kv, depth_v] 
			mask:
			dims: dictionary of  last 2 dimensions {tensor: {length: , depth: } , ... }
				{'queries': {'length': , 'depth': }, 'keys': {'length': , 'depth': }, 'values': {'length': , 'depth': }}


		"""

		super().__init__()
		
		self.softmax  = torch.nn.Softmax(dim=2)

	@TensorPrep.attention_get_dims
	def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: bool = False, dims = None): 

		 

		Q = queries.reshape(-1,  dims['queries']['length'],  dims['queries']['depth'])  # [batch*heads, length_q,  depth_k] 
		K = keys.reshape(   -1,  dims['keys']['length'],     dims['keys']['depth'])     # [batch*heads, length_kv,  depth_k] 
		V = values.reshape( -1,  dims['values']['length'],   dims['values']['depth'])   # [batch*heads, length_kv, depth_v] 


		# # math.sqrt if not tensor
		shrinking_weight = sqrt(dims['queries']['depth'])

		# [batch*heads, length_q,  depth_k] * [batch*heads, depth_k, length_kv]
		scaled_result = torch.bmm(Q, K.transpose(2,1)) / shrinking_weight  # out : [batch*heads, length_q,  length_kv] 
		v_filter = self.softmax(scaled_result)							  

		# [batch*heads, length_q,  length_kv] * [batch*heads, length_kv, depth_v]
		attention_results = torch.bmm(v_filter, V) 				 # out:   [batch*heads, length_q,  depth_v]

		# if mask:
		# 	# TODO mask 
		# 	pass
		
		return attention_results



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
	print("y", y.shape )

unit_test_attention()



# class MultiHeadedAttention(Attention): 

#     def __init__(self, input_size, num_heads, d_k, d_v):
#         # TODO set init for attention correctly 
#         Attention.__init__()

#         # set size of keys/queries transformation so you get size d_k for each head
#         output_size_qk = num_heads * d_k
#         # set size of values transformation so you get size d_k for each head
#         output_size_v  = num_heads * d_v

#         self.linear_q = torch.nn.Linear(input_size, output_size_qk)
#         self.linear_k = torch.nn.Linear(input_size, output_size_qk)
#         self.linear_v = torch.nn.Linear(input_size, output_size_v)


#     def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: bool): 



#         Q, K, V =  self.linear_q(queries), self.linear_k(keys), self.linear_v(values)


#         # Attention()  in parralell - inlcudes lineawr  already
#         # make it go through single atttention
#         Attention.forward(self, Q, K, V, mask)
		
		
#         # Attention()  in parralell - inlcudes lineawr  already
#         # TODO concat
#         # TODO other linear 


#         #TODO  add residual and norm
		
#         pass

# class FeedForward(nn.Module):

#     def __init__(self, in_channels, out_channels):


#         self.position_wise_linear = nn.Sequential(        
#             nn.Conv1d( in_channels, out_channels, kernel_size=1),
#             nn.Conv1d( in_channels, out_channels, kernel_size=1), 
#             nn.LayerNorm()
#         )
		

#     def forward(self, atten_results: torch.Tensor):

#         output = self.position_wise_linear(atten_results)


#         # TODO add attention residual and norm 


#         return

# N = 6 # num layers in encoder 

# class Encoder(FeedForward, MultiHeadedAttention): 

#     def __init__(self): 
#         FeedForward.__init__(self)
#         MultiHeadedAttention.__init__(self)

		# self.word_embeddings = nn.Embeddings(num_words, d_model = 512)  
		# self.positional_encoding = nn.Embedding(num_embeddings, d_model = 512) do from pretrained (lambda function?)  

#     def forward(self): 
		# sum embeddings + positional encoding
#         outputs = MultiHeadedAttention.forward( )
#         queries, keys = FeedForward()

#         return queries, keys

# class Decoder(Encoder): 

#     def __init__(self):
#         super().__init__()

#     def forward(self): 

#         masked_values = super().MultiHeadedAttention(Q, K, V, mask =true)
#         encoded_q, encoded_K = Encoder.forward()

#         out = super().MultiheadedAttention(encoded_K, encoded_q, masked_values)
#         results = super().FeedForward(out)



# class Transformer(Decoder):

#     def __init__(self):
#         super().__init__()
#         self.get_probabilites = nn.Sequential( [ 
#             nn.Linear(), 
#             nn.Softmax()
#         ])

#     def forward(self): 

#         results = super().forward()
#         output_prob = self.get_probabilites(results)
		
#         return output_prob
		
