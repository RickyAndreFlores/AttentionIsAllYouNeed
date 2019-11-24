import torch
from torch import nn
from math import sqrt
from helper import TensorPrep, TensorInfo

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


	@TensorInfo.show__tensor_sizes
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

		# scaled dot product attention
		self.scaled_dot_attention = Attention()

		# ending linear transform 
		self.final_linear = torch.nn.Linear(self.depth_v*self.num_heads, self.d_model )

		self.norm = nn.LayerNorm(self.d_model)


	@TensorInfo.show__tensor_sizes
	def forward(self, previous_output: torch.Tensor , mask: bool = False): 
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

		att_results = self.scaled_dot_attention(Q, K, V)  # out: [batch*heads, length_q,  depth_v]

		multi_att_concat = self.format_concat(att_results)

		# linear transform of contact that brings values back to d_model dimensions ==  embed_size
		output = self.final_linear(multi_att_concat)  #out: ( batch_size x sequence_size x d_model )
		
		add_norm = self.norm(residual + output)
		
		return add_norm
		
		# TODO add dropout in appropriate locations
	
	def format_concat(att_results: torch.Tensor):
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

	def get_qkv(self, previous_output: torch.Tensor):
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
		projected_results = self.project(previous_output)

		Q, K , V =  self.split_qkv(projected_results)

		return Q, K, V

	def project(previous_output: torch.Tensor):
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

	def split_qkv(projected_results): 
		"""
		In:
			torch.float of size ( batch_size x sequence_size x proj_depth= (depth_qk + depth_qk + depth_v)*num_heads ) 

		Out: 
			3 tensors of size 
			(Batch x num_heads x seq x depth_qk) (Batch x num_heads x seq x depth_qk) (Batch x num_heads x seq x depth_v)
		"""
		
		self.batch_size = previous_output.shape[0]
		self.seq = previous_output.shape[1]

		# split in QKV along 3rd dimension == slice up filters results into 3 chunks == split across channels to extract tuple  (q, k, v)
		# out: (Batch x seq x depth_q*num_heads) ... (Batch x seq x depth_v*num_heads) 
		qkv = torch.split(projected_results, [self.depth_qk*self.num_heads, self.depth_qk*self.num_heads, self.depth_v*self.num_heads], 2)

 		# in (Batch x seq x depth_q*num_heads) -> (Batch x seq x num_heads x depth_q) ->
		# out : -> (Batch x num_heads x seq x depth_q)
		split_heads = lambda unsplit: unsplit.contiguous().view(self.batch_size, self.seq, self.num_heads, -1).permute(0,2,1,3)
		Q, K, V = split_heads(qkv[0]),  split_heads(qkv[1]),  split_heads(qkv[2])

		return Q, K, V


class FeedForward(nn.Module):

	def __init__(self, d_model=512, hidden=2048):

		super().__init__()


		self.position_wise_linear = nn.Sequential(        
			nn.Conv1d( d_model, hidden, kernel_size=1),
			nn.ReLU(),
			nn.Conv1d( hidden, d_model, kernel_size=1), 
		)

		self.norm = nn.LayerNorm(d_model)

	@TensorInfo.show__tensor_sizes
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

	def __init__(self, N_layers: int=6): 
		"""
		Create N layers of encoder layer

		Encoder layer consits of:
			Sublayer 1: 
				multiheaded attention 
				residual add then norm
			Sublayer 2: 
				Positionwise feed forward 
				residual add then norm

		"""
		super().__init__()

		layers = []
		for _ in range(N_layers): 
			layers.append(MultiHeadedAttention())
			layers.append(FeedForward())

		# layers = [ MultiHeadedAttention(), FeedForward() for _ in range(N_layer) ]
		self.encoder_layers = nn.Sequential(*layers) 

	@TensorInfo.show__tensor_sizes
	def forward(self, input_embedding: torch.Tensor): 
		"""
		input :   ( batch_size x sequence_size x d_model )
		output:   ( batch_size x sequence_size x d_model )


		Go through stack of N decoder layers, given (input_embeddings + positional_encoding)
		"""
		
		return 	self.encoder_layers(input)
 


class positional_encodings(nn.Module):

	def __init__(self, num_words: int, d_model: int = 512):
		
		super().__init__()

		self.num_words = num_words
		self.d_model = d_model

		# generate table
		positional_encoding_values: torch.FloatTensor = self.get_pos_values()

		# make look up table
		self.positional_encoding_lookup = nn.Embedding.from_pretrained(positional_encoding_values, freeze=True)

	@TensorInfo.show__tensor_sizes
	def forward(self, indexes: torch.LongTensor): 
		"""

		// from pytorch docs 
		Input: (*), LongTensor of arbitrary shape containing the indices to extract

		Output: (*, H) , where * is the input shape and H=embedding_dim


		"""
		return self.positional_encoding_lookup(indexes)

	def get_pos_values(self): 
		"""
		out:(n_positions x d_model)
			where each value in the even (2*i)   columns is encoded with it's respective PE_sin(row,col) value
			where each value in the odds (2*i+1) columns is encoded with it's respective PE_cos(row,col) value

		"""
		
		#shape (n_positions x d_model)
		encodings: torch.float = torch.randn(self.num_words, self.d_model)

		num_i = self.d_model // 2

		# for every word position
		for pos in range(self.num_words): 
			# iterate through every 2 columns 
			for i in range(num_i):
				encodings[pos][2*i]     = self.pe_sin(pos, 2*i)   		# evens
				encodings[pos][2*i + 1] = self.pe_cos(pos, 2*i + 1)		# odds

			# if odd
			if self.d_model % 2 == 1:
				# get last collumn 
				encodings[pos][-1]     = self.pe_sin(pos, num_i)

		return encodings

	def pe_sin(self, pos, col): 
		# col = 2i = even col 
		exp = col / self.d_model
		return torch.sin(pos / torch.pow(torch.Tensor([10000.0]), exp))

	def pe_cos(self, pos, col): 
		# col = 2i + 1 = odd col 
		exp = col / self.d_model
		return torch.cos(pos / torch.pow(torch.Tensor([10000.0]), exp))


class Masked_MultiHeadedAttention(MultiHeadedAttention): 


	def __init__(self):
		"""
		Child class of MultiHeaded Attention

		Does everything Multiheaded attention does but with a mask

		"""
		super().__init__()
		print(self.__class__)

	def forward(self, input):
		"""
		TODO -> implement mask 

		Arg/Input:

    	    previous_output: Output of previous sub_layer or batch of output embeddings initially  
		Input shape

        previous_output :  ( batch_size x sequence_size x embed_size ) 

		"""
		MultiHeadedAttention.forward(self, input)
		

class Decoder_MultiHeadedAttention(MultiHeadedAttention): 

	def __init__(self,  encoder_Q, encoder_K):
		"""
		Child class of MultiHeaded Attention

		Does everything Multiheaded attention does takes in Encoder stack output for Q and K values
		"""
		super().__init__()
		print(self.__class__)

		self.encoder_Q = encoder_Q
		self.decoder_K = encoder_K


	def forward(self, decoder_previous_output):
		"""

		Arg/Input:

    	    previous_output: Output of previous sub_layer or batch of output embeddings initially  
		Input shape

        previous_output :  ( batch_size x sequence_size x embed_size ) 

		"""
		# ( batch_size x sequence_size x embed_size )
		residual = decoder_previous_output

		_, _, V = self.get_qkv(previous_output)

		att_results = self.scaled_dot_attention(self.encoder_Q, self.decoder_K, V)  # out: [batch*heads, length_q,  depth_v]

		multi_att_concat = self.format_concat(att_results)

		# linear transform of contact that brings values back to d_model dimensions ==  embed_size
		output = self.final_linear(multi_att_concat)  #out: ( batch_size x sequence_size x d_model )
		
		add_norm = self.norm(residual + output)
		
		return add_norm



# TODO decoder
# TODO mask
class Decoder(nn.Module): 

	def __init__(self, encoder_Q, encoder_K, N_layers: int = 6):
		super().__init__()


		layers = []
		for _ in range(N_layers): 
			layers.append(Masked_MultiHeadedAttention())  #TODO masked
			layers.append(Decoder_MultiHeadedAttention(encoder_Q, encoder_K)) 
			layers.append(FeedForward())

		# layers = [ MultiHeadedAttention(), FeedForward() for _ in range(N_layer) ]
		self.decoder_layers = nn.Sequential(*layers) 



	def forward(self, prev_outputs): 

		outputs =  self.decoder_layers(prev_outputs)


		return outputs


# TODO finish transformer
class Transformer(nn.Module):

	def __init__(self,  sequence_input,
						embedding_size:int = 512, 
						num_heads:int = 6, 
						depth_qk: int = 64, 
						depth_v: int = 64):
		
		super().__init__()

		# TODO see what values a needed where / adjust or remove initialize defautl values 
		self.d_model =			embedding_size 		# Size of embeddings
		self.num_heads = 		num_heads
		self.depth_qk = 		depth_qk
		self.depth_v = 		    depth_v

		self.seq_len = len(sequence_input)

		# Initiliaze embeddings and their look up table
		self.word_embeddings = nn.Embedding(num_words, d_model)  
		self.positional_encoding = positional_encodings(num_words, d_model)  #map values to sin function in paper

		self.encoder = Encoder()
		# Do a single pass through encoder stack 
		encoder_Q, encoder_K = self.get_output_QK()
		# Initialize encoder with output of stack 
		self.decoder = Decoder(encoder_Q, encoder_K) 


		self.get_probabilites = nn.Sequential( [ 
			# in:( batch_size x sequence_size x embed_size )
			# out:( batch_size x sequence_size x output_size )
			nn.Linear(d_model, self.seq_len), 
			# softmax over sequence -- need to remember/recheck TODO
			nn.Softmax(dim=1) 
			])
		
		
	def get_output_QK(self, sequence_input): 

		combined_output = self.encoder_forward(sequence_input)

	def encoder_forward(self, sequence_input):
		"""
		Get output of encoder layers stack

		"""

		# torch.LongTensor
		indexes: torch.LongTensor = input2idx(sequence_input)		
		# Add positional encoding information
		input_embedding = self.word_embeddings(indexes) + self.positional_encoding(indexes)

		encoder_output = self.encoder(input_embedding)

		return encoder_output
		
	def forward(self, prev_outputs): 
		"""
		In: 
			Sequence_input: sentence 
			prev_output: output embeddings based on previous step's decoder output probabilites 

		"""

		# encoder_Q, encoder_K = self.get_encoder_output()

		# Takes in encoder outputs to extract Q, K and prev_output = (output_embeddings + positional)
		output_probabilities = self.decoder(prev_outputs, encoder_output)

		# TODO might integrate this function with decoder
		output_embeddings = self.get_output_embed(output_probabilities)


		return output_embeddings


		
	def get_output_embed(self, output_probabilities): 
		"""
		returns output-embeddings ready for input to decoder
		"""
		
		# TODO word to index look up input2idx(input) for output_probs / other language 

		# shift max to right

		decoder_input = self.get_

		# torch.LongTensor
		indexes: torch.LongTensor = input2idx(sequence_input)
		# Add positional encoding information
		self.input_embedding = self.word_embeddings(indexes) + self.positional_encoding(indexes)
	
		return self.input_embedding
		# torch.LongTensor
		indexes: torch.LongTensor = input2idx(sequence_input)
		# Add positional encoding information
		input_embedding = self.word_embeddings(indexes) + self.positional_encoding(indexes)
		return input_embedding





	# def get_input_embed(self, sequence_input, use_same_input: bool = True): 
	# 	"""
	# 	output input embeddings ready for encoder
	# 	"""
		
	# 	# TODO word to index look up input2idx(input) 
	# 	if use_same_input:
			
	# 		if self.input_embeddings == None: 

	# 			# torch.LongTensor
	# 			indexes: torch.LongTensor = input2idx(sequence_input)

	# 			# Add positional encoding information
	# 			self.input_embedding = self.word_embeddings(indexes) + self.positional_encoding(indexes)
			
	# 		return self.input_embedding

	# 	else: 
	# 		# torch.LongTensor
	# 		indexes: torch.LongTensor = input2idx(sequence_input)

	# 		# Add positional encoding information
	# 		input_embedding = self.word_embeddings(indexes) + self.positional_encoding(indexes)

	# 		return input_embedding
	