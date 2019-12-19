import torch
from torch import nn
from math import sqrt
from helper import TensorInfo

 

class Positional_Encodings(nn.Module):

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


# TODO mask
# TODO finish transformer
class Transformer(nn.Module):

	def __init__(self,  sequence_input,
						vocab, 
						embedding_size:int = 512, 
						num_heads:int = 6, 
						depth_qk: int = 64, 
						depth_v: int = 64):
		
		super().__init__()


		self.vocab = vocab
		self.vocab_size = len(vocab)

		# TODO see what values a needed where / adjust or remove initialize defautl values 
		self.d_model =			embedding_size 		
		self.num_heads = 		num_heads
		self.depth_qk = 		depth_qk
		self.depth_v = 		    depth_v

		self.seq_len = len(sequence_input)

		# Initiliaze embeddings and their look up table
		self.word_embeddings_lookup = nn.Embedding(self.vocab_size, self.d_model )  
		self.positional_encodings_lookup = Positional_Encodings(self.vocab_size, self.d_model)  #map values to sin function in paper

		self.decoder = self.setupDecoder(sequence_input)

		self.get_probabilites = nn.Sequential( [ 
			# in:( batch_size x sequence_size x embed_size )
			# out:( batch_size x sequence_size x 1 )
			nn.Linear(d_model, 1), 
			# softmax over sequence -- need to remember/recheck TODO
			nn.Softmax(dim=1) 
			])
		
	def forward(self, prev_outputs, mask): 
		"""
		In: 
			prev_output: output embeddings based on previous step's decoder output probabilites 

		"""

		# Takes in encoder outputs to extract Q, K and prev_output = (output_embeddings + positional)
		decoder_output = self.decoder(prev_outputs, mask)

		# TODO might integrate this function with decoder
		output_embeddings = self.get_output_embed(output_probabilities)


		return output_embeddings

	def predict_next_word(self,  prev_outputs, mask): 

		# Takes in encoder outputs to extract Q, K and prev_output = (output_embeddings + positional)
		decoder_output = self.decoder(prev_outputs, mask)

		# TODO might integrate this function with decoder
		output_embeddings = self.get_output_embed(output_probabilities)

	
	def runEncoder(self, input_embeddings):
		self.encoder = Encoder()
		encoder_output = Encoder(input_embeddings)

		return encoder_output

	def setupDecoder(self, sequence_input): 

		# Initialize and run encoder
		input_embeddings =  self.get_encoder_inputs(sequence_input)
		encoder_output = self.runEncoder(input_embeddings)

		# Initialize encoder with output of encoder layers stack 
		return Decoder(encoder_output) 

	def get_encoder_inputs(self, sequence_input):
		"""
		in:
			sequence of words

		out:
			output of encoder layers stack
			size = ( batch_size x sequence_size x d_model )

		"""

		# torch.LongTensor
		indexes: torch.LongTensor = self.input2idx(sequence_input)		
		# Add positional encoding information
		input_embedding = self.word_embeddings_lookup(indexes) + self.positional_encodings_lookup(indexes)

		return input_embedding

	def get_right_shfted(self, output_embed):

		output_embed[1:] = output_embed[:-1]
		output_embed[0] = 0

		return output_embed
	
	def get_decoder_inputs(self, output_probabilities): 
		"""
		returns output-embeddings ready for input to decoder
		"""
		
		# TODO word to index look up input2idx(input) for output_probs / other language 

		# shift max to right
		decoder_input = self.get_right_shfted(decoder_embeds)

		# torch.LongTensor
		indexes: torch.LongTensor = input2idx(sequence_input)
		# Add positional encoding information
		self.input_embedding = self.word_embeddings(indexes) + self.positional_encoding(indexes)
	
		return input_embedding

	
	def input2idx(self, input_sequence):

		return torch.LongTensor([self.vocab[word] for word in input_sequence ])
		