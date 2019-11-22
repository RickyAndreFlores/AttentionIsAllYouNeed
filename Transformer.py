import torch
from torch import nn
from math import sqrt
from helper import TensorPrep


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


		output: 
			[batch*heads, length_q,  depth_v]

		"""

		super().__init__()
		self.softmax  = nn.Softmax(dim=2)


	@TensorPrep.show__tensor_sizes
	@TensorPrep.attention_get_dims
	def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: bool = False, dims: dict = None): 

		 

		Q = queries.reshape(-1,  dims['queries']['length'],  dims['queries']['depth'])  # [batch*heads, length_q,  depth_k] 
		K = keys.reshape(   -1,  dims['keys']['length'],     dims['keys']['depth'])     # [batch*heads, length_kv,  depth_k] 
		V = values.reshape( -1,  dims['values']['length'],   dims['values']['depth'])   # [batch*heads, length_kv, depth_v] 

		# TODO make sure this is safe-
		# print("see if unsafe", Q.grad_fn)

		# # math.sqrt if not tensor
		shrinking_weight = sqrt(dims['queries']['depth'])

		# [batch*heads, length_q,  depth_k] * [batch*heads, depth_k, length_kv]
		scaled_result = torch.bmm(Q, K.transpose(2,1)) / shrinking_weight 			 # out : [batch*heads, length_q,  length_kv] 

		# if mask:
		# 	# TODO mask 
		# 	pass
		
		v_filter = self.softmax(scaled_result)							  

		# [batch*heads, length_q,  length_kv] * [batch*heads, length_kv, depth_v]
		attention_results = torch.bmm(v_filter, V) 									 # out:   [batch*heads, length_q,  depth_v]

		# essentaily return the result of multiheaded attention with each head output (including across batches) concated in first dimension
		return attention_results

	


class MultiHeadedAttention(nn.Module): 

	def __init__(self, embedding_size:int = 512, num_heads:int = 6, depth_q: int = 64, depth_k: int = 64, depth_v: int = 64):
		"""
		Init Args: 
			embedding_size 
			num_heads
			depth_q
			depth_k
			depth_v

		Forward Arg/Input: 
			previous_output: Output of previous sub_layer or batch of word embeddings initially
	
		Input shape
			previous_output :  ( batch_size x sequence_size x embed_size )
		Output  shape
			previous_output :  ( batch_size x sequence_size x embed_size )
		"""

		super().__init__()

		self.d_model =			 embedding_size 		# Size of embeddings
		self.num_heads = 		num_heads
		self.depth_q = 			depth_q
		self.depth_k = 			depth_k
		self.depth_v = 		    depth_v

		# Size of embeddings
		self.d_model = embedding_size
		
		# size of output of linear projection
		self.proj_depth = (depth_q + depth_k + depth_v)*num_heads

		# linearly projection , self attention 
		self.projection = nn.Conv1d( in_channels=self.d_model, out_channels= self.proj_depth , kernel_size=1)

		# scaled dot product attention
		self.scaled_dot_attention = Attention()

		# ending linear transform 
		self.final_linear = torch.nn.Linear(self.depth_v*self.num_heads, self.d_model )

		self.norm = nn.LayerNorm(self.d_model)


	@TensorPrep.show__tensor_sizes
	def forward(self, previous_output: torch.Tensor , mask: bool = False): 
		"""		
		Arg/Input: 
			previous_output: Output of previous sub_layer or batch of word embeddings initially
		
		Input shape
			previous_output :  ( batch_size x sequence_size x embed_size )

		What is going on here: 
			1. in: ( batch_size x sequence_size x embed_size )
					embed_size == depth_model == d_model
			2. projection through conv 
				shape-> ( batch_size x sequence_size x  (depth_q *num_heads + depth_k*num_heads + depth_v*num_heads)   )
					==  ( batch_size x sequence_size x  (depth_q  + depth_k+ depth_v)*num_heads)   )
			3. split into Q, V, C 
				shape -> (Batch x seq x depth_q * num_heads) (Batch x seq x depth_k * num_heads) (Batch x seq x depth_v * num_heads)
				where depth_k == depth_q is mandatory. As notation I might write this as d_qk 
				although in implementation its also true that depth_qk == d_v  
			4. Split into num_head sequences of depth depth_q and depth_kv respectively 
				shape -> (Batch x num_heads x seq x depth_q) (Batch x num_heads x seq x depth_kv) (Batch x num_heads x seq x depth_kv)
				This now matched the input needed for our scaled dot product attention : [batch, heads, length_q, depth_k]
			5. Input and do attention 
				-> output shape = [batch*heads, length_q,  depth_v] == (batch*heads x  sequence  x  depth_v)
			6. Concat heads
				shape -> (batch x  sequence  x  depth_v*heads )
			7. Linear transform at end  concat(headi..headn)*W_o 
				densely connected. Where W_o shape = (h*depth_v x d_model)
				operation = (batchx  sequence  x  depth_v*heads ) * (h*depth_v x d_model)
				output shape -> (batchx  sequence  x   d_model)
				== ( batch_size x sequence_size x embed_size )
				== input size
		""" 

		residual = previous_output

		self.batch_size = previous_output.shape[0]
		self.seq = previous_output.shape[1]

		# ( batch_size x sequence_size x embed_size ) ->( batch_size x embed_size x sequence_size )
		# Align channels for 1D convolution
		prev_transpose = previous_output.transpose(2, 1)

		# !D coblution proj_depth times to extract proj_depth featrures, then transpose to original dim order
		projected_results = self.projection(prev_transpose).transpose(1,2) # out : ( batch_size x sequence_size x proj_depth )

		# split in QKV along 3rd dimension == slice up filter results into 3 chunks
		# out: (Batch x seq x depth_q*num_heads) ... (Batch x seq x depth_v*num_heads) 
		qkv = torch.split(projected_results, [self.depth_q*self.num_heads, self.depth_k*self.num_heads, self.depth_v*self.num_heads], 2)

 		# in (Batch x seq x depth_q*num_heads) -> (Batch x seq x num_heads x depth_q) ->
		# out : -> (Batch x num_heads x seq x depth_q)
		split_heads = lambda unsplit: unsplit.contiguous().view(self.batch_size, self.seq, self.num_heads, -1).permute(0,2,1,3)
		Q, K, V = split_heads(qkv[0]),  split_heads(qkv[1]),  split_heads(qkv[2])

		att_results = self.scaled_dot_attention(Q, K, V)  # out: [batch*heads, length_q,  depth_v]

		# seperate heads from batches
		sep_batches = att_results.view(self.batch_size, self.num_heads, self.seq, self.depth_v)

		# switch collumns to keep order for merge
		# concat heads along last dimension (i.e keep sequences in tact) -> out: (batch x  sequence  x  depth_v*heads )
		multi_att_concat = sep_batches.permute(0,2,1,3).contiguous().view(self.batch_size, self.seq, self.depth_v*self.num_heads)

		# linear transform of contact that brings values back to d_model dimensions ==  embed_size
		output = self.final_linear(multi_att_concat)  #out: ( batch_size x sequence_size x d_model )
		
		add_norm = self.norm(residual + output)
		
		return add_norm
		
		# TODO add dropout in appropriate locations
		
class FeedForward(nn.Module):

	def __init__(self, d_model=512, hidden=2048):

		super().__init__()


		self.position_wise_linear = nn.Sequential(        
			nn.Conv1d( d_model, hidden, kernel_size=1),
			nn.ReLU(),
			nn.Conv1d( hidden, d_model, kernel_size=1), 
		)

		self.norm = nn.LayerNorm(d_model)

	@TensorPrep.show__tensor_sizes
	def forward(self, multi_atten_results: torch.Tensor):
		"""
		input :   ( batch_size x sequence_size x d_model )
		output:   ( batch_size x sequence_size x d_model )
		"""
		
		residual = multi_atten_results

		# output -> ( batch_size x d_model x sequence_size)
		output = self.position_wise_linear(multi_atten_results.permute(0,2,1))

		add_norm= self.norm(residual + output.permute(0,2,1))

		return add_norm


class Encoder(nn.Module): 

	def __init__(self, num_words: int=2, N_layers: int=6, d_model:int = 512): 
		super().__init__()


		self.word_embeddings = nn.Embedding(num_words, d_model)  
		# TODO positional encodings
		self.positional_encoding = nn.Embedding(num_words, d_model)  #map values to sin function in paper


		layers = []
		for _ in range(N_layers): 
			layers.append(MultiHeadedAttention())
			layers.append(FeedForward())

		# layers = [ MultiHeadedAttention(), FeedForward() for _ in range(N_layer) ]
		print(layers)
		self.encoder_layers = nn.Sequential(*layers) 

	@TensorPrep.show__tensor_sizes
	def forward(self, input): 

		output = self.encoder_layers(input)
		
		return output 


# TODO positional encodings
def positional_encodings():
	pass


# TODO decoder
# TODO mask
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
		
