import torch
from torch import nn
from math import sqrt
from helper import TensorInfo


# TODO limit dependencies, clearly state them

class Attention(nn.Module): 

	def __init__(self):
		""" 
		scaled dot-product attention

		Args:
			queries: linear transformation of antecedent or previous output
			keys: linear transformation of antecedent or previous output
			values: linear transformation of antecedent or previous output
			mask:

		Input Shape: 
			queries: [batch, heads, length_q, depth_k] 
			keys:    [batch, heads, length_kv, depth_k] 
			values:  [batch, heads, length_kv, depth_v] 
			mask: Torch.BoolTensor
			dims: dictionary of  last 2 dimensions {tensor: {length: , depth: } , ... }
				{'queries': {'length': , 'depth': }, 'keys': {'length': , 'depth': }, 'values': {'length': , 'depth': }}


		output: 
			[batch*heads, length_q,  depth_v]

		"""

		super().__init__()
		self.softmax  = nn.Softmax(dim=2)


	@TensorInfo.show__tensor_sizes
	def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor = None, dims: dict = None): 
		"""
		Given queries, keys,values 
			queries: [batch, heads, length_q, depth_k] 
			keys:    [batch, heads, length_kv, depth_k] 
			values:  [batch, heads, length_kv, depth_v] 
		Returns 
			[batch*heads, length_q==seq,  depth_v]

			essentaily return the result of multiheaded attention with each head output (including across batches) concated in first dimension


		Theoretically:
			- Q, K, V = shape : [all_outputs_across_batches_and_heads, sequence_length,  num_dims_in_learned_embedding] 
			
			lets focus on the 2D matrix of the last 2 dimensions ( sequence_length,  num_dims_in_learned_embedding )

			- Each dimension in an embedding space is a measure a certain feautre or "meaning"
			- since each row corresponds to a word, all rows concated correspond to entire sentence
			- a collumn in matrix Q == a sentence's vector along a certain "meaning" dimension


			- Q, K ,V = list of transformed word embeddedings 
			- transformed embeddings = projection of original meaning onto new "meaning" space
			- Thus transformed embeddings = projection of original meaning onto new "meaning" space
				= list of learned meanings for each word 

			- constant * Q*K_transpose =  
					for each collumn in output (== for each word):
						linear combination of the "meaning" vector_i of sentence(all words) in Query matrix
						, using as weights each value of the "meaning" vector of word_i in Key matrix

						== cross correlation of Key vector (multiple meanings :one word) of each word
							and each "meaning" vector in Q for whole sentence ( one meaning: all words)

					So essentially, each Key for a word (each row in K) learns the importance of each "meaning" or embedding dimension  
					Meanwhile a collumn in Q learns a representation of entire sentence along a single "meaning" dimension

					each collumnn in Q*K_transpose = 
						a feature vector for entire sentence that represent a weighted sum of each "interpretation" of a sentence
							where weights = a row in K = the "importance" of each meaning dimension
							and "interpretation" of a sentence is the values of a sentence along an embedding/"meaning dimension
						= cross correlation of a single learned "meanings" importance and each word in sentence 
						= while looking how a sentence ranks across differnt properties,
							meaure how high they rank on the certain combination of those properties that make up a different feature 


					for each row i in Q*K_transpose
						for each column j in QK_transpose
							QK_t[i][j]= cross correlation of (learned "meanings" importance)_i= K[:]][j] and leanred_word_embedding_i = Q[:][i]  
			
			softmax(scaled_attention_weights) * V  =  
				adds a non-linearlity 
				that creates "filter" for the each sentence "interpretation" of V  

			Mask and offset:
				'This masking, combined with fact that the output embeddings are offset by one position, 
				ensures that the predictions for position_i can depend only on the known outputs at positions less than i'

				Helps predict next word based off input words and predicted words so far only
		"""

		length_q = queries.shape[2]
		depth_k = queries.shape[3]

		length_kv = keys.shape[2]
		depth_v = values.shape[3]

		Q = queries.reshape(-1,  length_q,  depth_k)    # [batch*heads, length_q,  depth_k] 
		K = keys.reshape(   -1,  length_kv, depth_k)    # [batch*heads, length_kv,  depth_k] 
		V = values.reshape( -1,  length_kv, depth_v)    # [batch*heads, length_kv, depth_v] 

		# # math.sqrt if not tensor
		shrinking_weight = sqrt(depth_k)

		# [batch*heads, length_q,  depth_k] * [batch*heads, depth_k, length_kv] = [batch*heads, length_q,  length_kv] 
		scaled_attention_weights = torch.bmm(Q, K.transpose(2,1)) / shrinking_weight 			 # out : [batch*heads, seq,  seq] 

		if mask != None:
            # Fill positions highlighted by mask with zero
			scaled_attention_weights = scaled_attention_weights.masked_fill(mask, 0)

		# create a matix that is theortically (batch  x feature/extracted meaning x importance of each word by position)
		v_filter = self.softmax(scaled_attention_weights)							  

		# [batch*heads, length_q,  length_kv] * [batch*heads, length_kv, depth_v]
		# ==  [batch*heads, seq,  seq] * [batch*heads, seq, depth_v] =  [batch*heads, length_q==seq,  depth_v]
		attention_results = torch.bmm(v_filter, V) 	

		return attention_results

	
class MultiHeadedAttention(nn.Module): 

	def __init__(self, embedding_size:int = 512, num_heads:int = 6, depth_qk: int = 64, depth_v: int = 64):
		"""
		Init Args: 
			embedding_size 
			num_heads
			depth_qk = depth_q =depth_k
			depth_v

		Forward Arg/Input: 
			previous_output: Output of previous sub_layer or batch of word embeddings initially
	
		Input shape
			previous_output :  ( batch_size x sequence_size x embed_size )
		Output  shape
			previous_output :  ( batch_size x sequence_size x embed_size )
		"""

		super().__init__()

		self.d_model =			embedding_size 		# Size of embeddings
		self.num_heads = 		num_heads
		self.depth_qk = 		depth_qk
		self.depth_v = 		    depth_v
	
		# size of output of linear projection
		self.proj_depth = (depth_qk + depth_qk + depth_v)*num_heads

		# linearly projection , self attention 
		self.projection = nn.Conv1d( in_channels=self.d_model, out_channels= self.proj_depth , kernel_size=1)


		# Same as 1x1 convolution (when input is transpose) OPTIONAL replacement for self.projection
		self.linear_q = nn.Linear(self.d_model, depth_qk*num_heads)
		self.linear_k = nn.Linear(self.d_model, depth_qk*num_heads)
		self.linear_v = nn.Linear(self.d_model, depth_v*num_heads)

		# scaled dot product attention
		self.scaled_dot_attention = Attention()

		# ending linear transform 
		self.final_linear = torch.nn.Linear(self.depth_v*self.num_heads, self.d_model )

		self.norm = nn.LayerNorm(self.d_model)


	@TensorInfo.show__tensor_sizes
	def forward(self, previous_output: torch.Tensor , mask: torch.Tensor = None): 
		"""		
		Arg/Input: 
			previous_output: Output of previous sub_layer or batch of word embeddings initially
		
		Input shape
			previous_output :  ( batch_size x sequence_size x embed_size )

		What is going on here: 
			1. in: ( batch_size x sequence_size x embed_size )
					embed_size == depth_model == d_model
			2. projection throughs number of filters * 1x1xsequence convs
				transpose for pytorch> convoltion -> transpose -> 
					essentially a fully connected layer across the entire embedding, for each word in sequence
				out shape-> ( batch_size x sequence_size x  (depth_q *num_heads + depth_k*num_heads + depth_v*num_heads)   )
					==  ( batch_size x sequence_size x  (depth_q  + depth_k+ depth_v)*num_heads)   )
			3. split into (projected/transformed) Q, V, C 
				shape -> (Batch x seq x depth_q * num_heads) (Batch x seq x depth_k * num_heads) (Batch x seq x depth_v * num_heads)
				where depth_k == depth_q is mandatory. As notation I might write this as d_qk 
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

		# ( batch_size x sequence_size x embed_size )
		residual = previous_output

		Q, K, V = self.get_qkv(previous_output)

		att_results = self.scaled_dot_attention(Q, K, V, mask)  # out: [batch*heads, length_q,  depth_v]

		multi_att_concat = self.concat_heads(att_results)

		# linear transform of contact that brings values back to d_model dimensions ==  embed_size
		output = self.final_linear(multi_att_concat)  #out: ( batch_size x sequence_size x d_model )
		
		add_norm = self.norm(residual + output)
		
		return add_norm
		
		# TODO add dropout in appropriate locations
	
	def concat_heads(self, att_results: torch.Tensor):
		"""
 			in: [batch*heads, sequence,  depth_v]

			out:
				(batch x  sequence  x  depth_v*heads )
		"""
		# seperate heads from batches
		sep_batches = att_results.view(self.batch_size, self.num_heads, self.seq, self.depth_v)

		# switch collumns to keep order for merge
		# concat heads along last dimension (i.e keep sequences in tact) -> out: (batch x  sequence  x  depth_v*heads )
		multi_att_concat = sep_batches.permute(0,2,1,3).contiguous().view(self.batch_size, self.seq, self.depth_v*self.num_heads)

		return multi_att_concat

	def get_qkv(self, previous_output: torch.Tensor, project_by_parts=False, encoder_output=None):
		"""
		In:
			torch.float of size 
			(batch_size x sequence_size x embed_size ) 

		Out: 
			3 tensors of size 
			(Batch x num_heads x seq x depth_qk) 
			(Batch x num_heads x seq x depth_qk) 
			(Batch x num_heads x seq x depth_v)		
		
		"""

		if project_by_parts == False:
			projected_results = self.project(previous_output)

		else: 
			projected_results = self.project_by_parts_linear(previous_output, encoder_output)

		Q, K , V =  self.split_qkv(projected_results)

		return Q, K, V

	def project(self, previous_output: torch.Tensor):
		"""
		In:
			torch.float of size 
			(batch_size x sequence_size x embed_size ) 

		Out: 
			( batch_size x sequence_size x proj_depth )
		"""

		self.batch_size = previous_output.shape[0]
		self.seq = previous_output.shape[1]

		# ( batch_size x sequence_size x embed_size ) ->( batch_size x embed_size x sequence_size )
		# Align channels for 1D convolution
		prev_transpose = previous_output.transpose(2, 1)

		# !D coblution proj_depth times to extract proj_depth featrures, then transpose to original dim order
		projected_results = self.projection(prev_transpose).transpose(1,2) # out : ( batch_size x sequence_size x proj_depth )

		return projected_results

	def project_by_parts_linear(self, previous_output, encoder_output=None): 
		"""

		Optional
		Equaivalent to self.project() ==
	    	Equivalent to convolution by 1x1xsequence_len convolution when previous_ouput's last 2 dimension are transposed

		In:
			torch.float of size 
			(batch_size x sequence_size x embed_size ) 

		Out: 
			( batch_size x sequence_size x proj_depth )

		"""
		
		# allow for encoder input for decoder
		if type(encoder_output) == type(None): 
			encoder_output = previous_output
			print("No encoder input, will do normal projection by parts")

		self.batch_size = previous_output.shape[0]
		self.seq = previous_output.shape[1]


		Q_proj = self.linear_q(previous_output) # ( batch_size x sequence_size  x self.depth_qk*self.num_heads)
		V_proj = self.linear_v(encoder_output) # ( batch_size x sequence_size  x self.depth_qk*self.num_heads)
		K_proj = self.linear_k(encoder_output) # ( batch_size x sequence_size  x self.depth_v *self.num_heads)

		projected_results = torch.cat([Q_proj,V_proj,K_proj], dim=2)  #( batch_size x sequence_size x proj_depth )

		return projected_results

	def split_qkv(self, projected_results): 
		"""
		In:
			torch.float of size ( batch_size x sequence_size x proj_depth= (depth_qk + depth_qk + depth_v)*num_heads ) 

		Out: 
			3 tensors of size 
			(Batch x num_heads x seq x depth_qk) (Batch x num_heads x seq x depth_qk) (Batch x num_heads x seq x depth_v)
		"""
		
		self.batch_size = projected_results.shape[0]
		self.seq = projected_results.shape[1]

		# split in QKV along 3rd dimension == slice up filters results into 3 chunks == split across channels to extract tuple  (q, k, v)
		# out: (Batch x seq x depth_q*num_heads) ... (Batch x seq x depth_v*num_heads) 
		qkv = torch.split(projected_results, [self.depth_qk*self.num_heads, self.depth_qk*self.num_heads, self.depth_v*self.num_heads], 2)

 		# in (Batch x seq x depth_q*num_heads) -> (Batch x seq x num_heads x depth_q) ->
		# out : -> (Batch x num_heads x seq x depth_q)
		split_heads = lambda unsplit: unsplit.contiguous().view(self.batch_size, self.seq, self.num_heads, -1).permute(0,2,1,3)
		Q, K, V = split_heads(qkv[0]),  split_heads(qkv[1]),  split_heads(qkv[2])

		return Q, K, V


class Decoder_MultiHeadedAttention(MultiHeadedAttention): 
	# TODO mask

	def __init__(self, encoder_output):
		"""
		Child class of MultiHeaded Attention
		Does everything Multiheaded attention does but takes in Encoder stack output for Q and K values

		input: 
				encoder output  ( batch_size x sequence_size x embed_size ) 
		"""

		super().__init__()

		# ( batch_size x sequence_size x embed_size ) 
		self.encoder_output = encoder_output

	@TensorInfo.show__tensor_sizes
	def forward(self, decoder_previous_output):
		"""
		Arg/Input:

    	    previous_output: Output of previous sub_layer or batch of output embeddings initially  
		Input shape

        previous_output :  ( batch_size x sequence_size x embed_size ) 

		"""
		# ( batch_size x sequence_size x embed_size )
		residual = decoder_previous_output

		# Linear transfer both encoder results and previous steps result to get Q,K,V
		# decoder_Q, _, _ 		= self.get_qkv(previous_output)
		# _, encoder_K, encoder_V = self.get_qkv(self.encoder_output)
		# Below avoids unnecessary calculation but less readable
		decoder_Q, encoder_K, encoder_V =  self.get_qkv(decoder_previous_output, project_by_parts=True, encoder_output = self.encoder_output)
		
		att_results = self.scaled_dot_attention(decoder_Q, encoder_K, encoder_V)  # out: [batch*heads, length_q,  depth_v]

		multi_att_concat = self.concat_heads(att_results)

		# linear transform of contact that brings values back to d_model dimensions ==  embed_size
		output = self.final_linear(multi_att_concat)  #out: ( batch_size x sequence_size x d_model )
		
		add_norm = self.norm(residual + output)
		
		return add_norm

