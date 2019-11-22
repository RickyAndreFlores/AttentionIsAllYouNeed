import torch
from torch import nn
from math import sqrt
from helper import TensorPrep


# TODO 1conv linear for q,vk

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
		
		self.softmax  = torch.nn.Softmax(dim=2)

	@TensorPrep.attention_get_dims
	def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: bool = False, dims: dict = None): 

		 

		Q = queries.reshape(-1,  dims['queries']['length'],  dims['queries']['depth'])  # [batch*heads, length_q,  depth_k] 
		K = keys.reshape(   -1,  dims['keys']['length'],     dims['keys']['depth'])     # [batch*heads, length_kv,  depth_k] 
		V = values.reshape( -1,  dims['values']['length'],   dims['values']['depth'])   # [batch*heads, length_kv, depth_v] 


		# # math.sqrt if not tensor
		shrinking_weight = sqrt(dims['queries']['depth'])

		# [batch*heads, length_q,  depth_k] * [batch*heads, depth_k, length_kv]
		scaled_result = torch.bmm(Q, K.transpose(2,1)) / shrinking_weight  # out : [batch*heads, length_q,  length_kv] 

		# if mask:
		# 	# TODO mask 
		# 	pass
		
		return attention_results
		v_filter = self.softmax(scaled_result)							  

		# [batch*heads, length_q,  length_kv] * [batch*heads, length_kv, depth_v]
		attention_results = torch.bmm(v_filter, V) 				 # out:   [batch*heads, length_q,  depth_v]

		# essentaily return the result of multiheaded attention with each head output (including across batches) concated in first dimension
		return attention_results

"""
given d_model dimensional keys, linearlly project to d_k, d_k, d_v dimension (Q, K, V)
 get Q K V from linearrly projected values of embeddings

output d_v dimensional 

concanted outputs + linear again

d_k = d_model / h = 64
T_q = length q

depth_v = d_model /h  = 64
h = 8
d_model = 512
"""

class MultiHeadedAttention(Attention): 
	"""

	Arg/Input: 
		previous_output: Output of previous sub_layer or batch of word embeddings initially
	

	Input shape
		previous_output :  ( batch_size x sequence_size x embed_size )


		d_model = input_size 

	"""
    def __init__(self, input_size, num_heads, d_k, d_v):
        # TODO set init for attention correctly 
        Attention.__init__()


		# Size of embeddings
		d_model = input_size

        # set size of keys/queries transformation so you get size d_k for each head
        output_size_qk = num_heads * d_k
        # set size of values transformation so you get size d_k for each head
        output_size_v  = num_heads * d_v


		# Weights: d_model x depth_k 
        self.linear_q = nn.Linear(d_model, output_size_qk)
		# Weights: d_model x depth_k 
        self.linear_k = nn.Linear(d_model, output_size_qk)
		# Weights: d_model x depth_v 
        self.linear_v = nn.Linear(d_model, output_size_v)

		
		# enough depth in order to split it up int Q,K,V
		out_channels = output_size_qk + output_size_qk + output_size_v
		self.conv1d = nn.Conv1d( in_channels= ? ,out_channels , kernel_size=(1,1))

    def forward(self, previous_output: torch.Tensor , mask: bool = False): 
		"""			
		What is going on here: 
			1. in: ( batch_size x sequence_size x embed_size )
					embed_size == depth_model == d_model
			2. projection through conv 
				shape-> ( batch_size x sequence_size x  (depth_q *num_heads + depth_k*num_heads + depth_v*num_heads)   )
					==  ( batch_size x sequence_size x  (depth_q  + depth_k+ depth_v)*num_heads)   )

			3. split into Q, V, C 
				shape -> (Batch x seq x depth_q * num_heads) (Batch x seq x depth_k * num_heads) (Batch x seq x depth_v * num_heads)
				where depth_k == depth_v is mandatory. As notation I might write this as d_kv 
				In implementation its also true that depth_k == d_kv  
			4. Split into num_head sequences of depth depth_q and depth_kv respectively 
				shape -> (Batch x num_heads x seq x depth_q) (Batch x num_heads x seq x depth_kv) (Batch x num_heads x seq x depth_kv)
				This now matched the input needed for our scaled dot product attention : [batch, heads, length_q, depth_k]
			5. Input and do attention 
				-> output shape = [batch*heads, length_q,  depth_v] == (batch*heads x  sequence  x  depth_v)
			6. Concat heads
				shape -> (batchx  sequence  x  depth_v*heads )
			7. Linear transform at end  concat(headi..headn)*W_o 
				densely connected. Where W_o shape = (h*depth_v x d_model)
				operation = (batchx  sequence  x  depth_v*heads ) * (h*depth_v x d_model)
				output shape -> (batchx  sequence  x   d_model)
				== ( batch_size x sequence_size x embed_size )
				== input size
		""" 
        Q, K, V =  self.linear_q(queries), self.linear_k(keys), self.linear_v(values)

		
	
        Attention.forward(self, Q, K, V, mask)  # out: [batch*heads, length_q,  depth_v]
		
		
        # Attention()  in parralell - inlcudes lineawr  already
        # TODO concat
        # TODO other linear 


        #TODO  add residual and norm
		
        pass

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
		
