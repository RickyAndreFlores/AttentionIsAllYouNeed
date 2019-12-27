import torch
from torch import nn
from helper import TensorInfo

class Positional_Encodings(nn.Module):
	"""
	creates positional encodings for a given batch of source and target sequences 
	encoding are sin and cos values that are depedent on the position and embedding dimension

	"""
	cache = {}

	def __init__(self, d_model: int = 512, device = torch.device('cuda')):
		
		super().__init__()

		self.d_model = d_model
		self.device = device

		self.src_seq_len = None
		self.target_seq_len = None
		self.batch_size = None

		# # values don't matter for init
		# self.src_pos_enc_lookup 	= nn.Embedding(1, d_model)
		# self.target_pos_enc_lookup  = nn.Embedding(1, d_model)

		# self.src_pos_enc_lookup.weight.requires_grad = False
		# self.target_pos_enc_lookup.weight.requires_grad = False

	@TensorInfo.show__tensor_sizes
	def forward(self,  src_sequences: torch.Tensor, target_sequences:torch.Tensor, src_or_target: str = 'both'): 

		batch_size, src_seq_len, target_seq_len = self.get_size_data(src_sequences, target_sequences)

		if batch_size == self.batch_size and src_seq_len == self.src_seq_len and target_seq_len == self.target_seq_len: 
			print("same one")
			encodings = self.return_encodings(src_or_target)

		else:
			
			self.batch_size, self.src_seq_len, self.target_seq_len = batch_size, src_seq_len, target_seq_len
			self.create_lookup()
			# Get encodings that fit batch size x sequence_leng
			self.src_encodings, self.target_encodings = self.create_seq_positional_encodings(src_sequences, target_sequences) #out: (batch x src/target seq x d_model )
			
			encodings = self.return_encodings(src_or_target)

		return encodings
	
	def return_encodings(self, src_or_target: str = 'both'):

		if src_or_target == 'src':
			return self.src_encodings

		elif src_or_target == 'target':
			return self.target_encodings

		elif src_or_target == 'both':
			return self.src_encodings, self.target_encodings

	def create_lookup(self):

		# generate table
		src_values = self.get_pos_values(self.src_seq_len)
		target_values = self.get_pos_values(self.target_seq_len)

		# TODO, make extendable 
		# make look up table
		self.src_pos_enc_lookup 	= nn.Embedding.from_pretrained(src_values, freeze=True).to(self.device)
		self.target_pos_enc_lookup  = nn.Embedding.from_pretrained(target_values, freeze=True).to(self.device)


	def get_encodings(self, src_or_target: str): 
		
		if src_or_target == 'src':
			return self.src_encodings
		elif src_or_target == 'target':
			return self.target_encodings
				  
	def get_pos_values(self, seq_len): 
		"""
		out:(n_positions x d_model)
			where each value in the even (2*i)   columns is encoded with it's respective PE_sin(row,col) value
			where each value in the odds (2*i+1) columns is encoded with it's respective PE_cos(row,col) value

		"""
		
		#shape (n_positions x d_model)
		encodings: torch.float = torch.randn(seq_len, self.d_model)


		# for every word position
		for pos in range(seq_len): 
			# iterate through every 2 columns 
			for i in range(self.d_model // 2):

				if (pos, i) in Positional_Encodings.cache: 
					
					even, odd = Positional_Encodings.cache[(pos, i)]

					encodings[pos][2*i]     = even		# evens
					encodings[pos][2*i + 1] = odd		# odds

				else:

					encodings[pos][2*i]     = self.pe_sin(pos, 2*i)   		# evens
					encodings[pos][2*i + 1] = self.pe_cos(pos, 2*i + 1)		# odds

					Positional_Encodings.cache[(pos, i)] = (encodings[pos][2*i], encodings[pos][2*i + 1]  )


			# if odd
			if self.d_model % 2 == 1:
				# get last collumn 
				encodings[pos][-1]     = self.pe_sin(pos, self.d_model // 2)

		return encodings

	def pe_sin(self, pos, col): 
		# col = 2i = even col 
		exp = col / self.d_model
		return torch.sin(pos / torch.pow(torch.Tensor([10000.0]), exp))

	def pe_cos(self, pos, col): 
		# col = 2i + 1 = odd col 
		exp = col / self.d_model
		return torch.cos(pos / torch.pow(torch.Tensor([10000.0]), exp))

	def get_size_data(self, src_sequences: torch.Tensor, target_sequences: torch.Tensor): 
		
		batch_size = src_sequences.shape[0]

		# They have different max seq length 
		src_seq_len    = src_sequences.shape[1]
		target_seq_len = target_sequences.shape[1]

		return batch_size, src_seq_len, target_seq_len

	def create_seq_positional_encodings(self, src_sequences: torch.Tensor, target_sequences: torch.Tensor):

		# TODO improve positional encoding so we don't repeat calculation
		# positional_encodings_lookup = create_positional_encodings(src_sequences: torch.Tensor, target_sequences: torch.Tensor)
		
		self.src_positions_matrix = self.create_positions_matrix(self.src_seq_len).to(self.device)
		self.target_positions_matrix = self.create_positions_matrix(self.target_seq_len).to(self.device)

		src_encodings 	  = self.src_pos_enc_lookup(self.src_positions_matrix)
		target_encodings = self.target_pos_enc_lookup(self.target_positions_matrix)
		
		return src_encodings, target_encodings

	def create_positions_matrix(self, seq_len: int):

		# get position vector  
		positions_range = torch.LongTensor([position for position in range(seq_len)])		
		# Broadcast it to all entries so it corresponds with batch input
		positions_matrix = torch.zeros((self.batch_size, seq_len), dtype=torch.long) + positions_range

		return positions_matrix




class Mask(nn.Module): 
	"""
	Class for creating the approrpriate mask

	Achieve the same result from just masking the sequence directly, so we do that here

	Each call of mask unmasks the next word in predicted sequence
	starts with the first token <sos> unmasked


	Notes: 
		In some implementations mask is done with a triangular matrix filled with zeros and ones in the approriate position
		element wise multiplied onto Q*K^T, or in other words a mask.

		What they use: 
		1 0 0
		1 1 0
		1 1 1
		
		These implementations argue that this triangle matrix prevents future information from being used in the decoder.
		I argue that this is wrong. 

		Eachr row in Q, K, and V can be thought of as representing features of a word (derived from it's word embedding)

		where . == dot product, and wi = word i 
		QK^T = 
	
		w1.w1  w1.w2  w1.w3
		w2.w1  w2.w2  w3.w3
		w3.w1  w3.w2  w3.w3
		
		if we use the mask for the first sequence  
		
		1 0 0
		1 1 0
		1 1 1
		
		we get 

		w1.w1  0  	  0    		#row1
		w2.w1  w2.w2  0			#row2
		w3.w1  w3.w2  w3.w3		#row3

		and QK*V = 

		entire_sentence_dim1.row1 entire_sentence_dim2.row1 entire_sentence_dim3.row1
		entire_sentence_dim1.row2 entire_sentence_dim2.row2 entire_sentence_dim3.row2
		entire_sentence_dim1.row3 entire_sentence_dim2.row3 entire_sentence_dim3.row3


		So you see, in this masked self attention, we are still feeding information of future words.

		Futhermore these implementations initalize the decoder witht the target sentence, so these leaks of the true answer
		might actually lead to better performance not because of the model, but because it's given the answer. 


		So if we use target sentence as input, and word one is the <sos> token (since it is shifted by one position), then 
		the output of self attention isn't just looking at <sos> its getting information about the target translation!

		So that is why I argue it is wrong.

		The appropriate mask of Q.K^T is actually: 

		for word 1 
		1  0  0
		0  0  0 
		0  0  0 


		w1.w1  0  	0
		0  	   0  	0 
		0  	   0 	0 


		for word 2:
			
		1  1  0
		1  0  0 
		0  0  0 


		w1.w1  w1.w2  0
		w2.w1  0	  0
		0      0  	  0
		
		etc...


		this in turn is equivalent to simply zero masking out the future words from the input 
		
		for word 1 if we have  [<sos>, 0,  0,  0, ...]
		
		and each row of Q and K correspond to the words features then 


		w1.w1  w1.w2  w1.w3
		w2.w1  w2.w2  w3.w3
		w3.w1  w3.w2  w3.w3

	"""

	def __init__(self, device=torch.device('cuda')):
		
		super().__init__()
		
		self.device = device
		self.umask_up_to = 1
		self.finished = False


	def forward(self, sequence_embedding):


		assert self.finished == False

		# get shapes info
		batch_size = sequence_embedding.shape[0]
		seq_size   = sequence_embedding.shape[1]
		embed_size = sequence_embedding.shape[2]

		# 0 == dont apply mask, 1 == apply mask
		unmask = 1
		mask   = 0

		sequence_mask = torch.zeros( seq_size, embed_size).to(self.device)

		for unmasked_idx in range(self.umask_up_to):
			
			# brodcasted across embed dimension
			sequence_mask[unmasked_idx] = unmask
		
		blank_slate = torch.zeros(batch_size, seq_size, embed_size).to(self.device)
		batch_mask  = blank_slate  + sequence_mask
		
		masked_embeddings = batch_mask * sequence_embedding
		
		# Next iteration show the following predicted word 
		self.umask_up_to += 1 


		# reached end of sequence
		max_index = self.umask_up_to - 1
		if max_index >= seq_size:
			self.finished = True

		return masked_embeddings

	def reset(self):

		# print("Reset mask index counter")
		self.umask_up_to = 1 		
		self.finished = False
		

class Pad_Mask(nn.Module): 


	def __init__(self, str_to_index, d_model, device = torch.device('cuda')): 
		
		super().__init__()
		
		self.device = device

		# TODO change str_to_index input to simply the pad tokens
		self.pad   = {'src': str_to_index.src['<pad>'], 'target':  str_to_index.target['<pad>'] }


		self.d_model = d_model

	def forward(self, masked_embeddings, sequence_index, src_or_target: str): 

		"""
		maked_embedding : shape = ( batch x seq x embed )
		maked_embedding : shape = ( batch x seq)
		src_or_target: str	    = either 'src' or 'target'


		"""

		pad_mask = self.get_pad_mask(sequence_index, src_or_target).to(self.device)


		return masked_embeddings * pad_mask


	def get_pad_mask(self, sequence_index, src_or_target: str): 


		pad_mask = (sequence_index != self.pad[src_or_target]).unsqueeze(2).expand(-1,-1, self.d_model).type(torch.FloatTensor)

		return pad_mask

