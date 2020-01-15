import torch
from torch import nn
from math import sqrt
from helper import TensorInfo
from dataclasses import dataclass
from EncoderDecoder import Encoder, Decoder
from PositionsMasks import Mask, Pad_Mask, Positional_Encodings
 
# TODO make compatabile with batch second
class Transformer(nn.Module):

	def __init__(self,	source_vocab_len, 
						target_vocab_len,
						str_to_index,
						index_to_str, 
						embedding_size:int = 512, 
						num_heads:int = 6, 
						depth_qk: int = 64, 
						depth_v: int = 64, 
						device = torch.device('cuda')):
		
		super().__init__()

		# TODO see what values a needed where / adjust or remove initialize defautl values 
		self.d_model =			embedding_size 		
		self.num_heads = 		num_heads
		self.depth_qk = 		depth_qk
		self.depth_v = 		    depth_v
		self.device = 			device

		# Initiliaze embeddings and their look up table
		self.src_embeddings_lookup 	  = nn.Embedding(source_vocab_len, self.d_model).to(self.device)
		self.target_embeddings_lookup = nn.Embedding(target_vocab_len, self.d_model).to(self.device)

		self.str_to_index  = str_to_index
		self.index_to_str  = index_to_str

		# self.start = {'src': str_to_index.src['<sos>'], 'target':  str_to_index.target['<sos>'] }
		self.end   = {'src': str_to_index.src['<eos>'], 'target':  str_to_index.target['<eos>'] }

		self.mask = Mask(device=self.device).to(self.device)
		self.pad_mask = Pad_Mask(self.str_to_index, self.d_model, device=self.device).to(self.device)

		# test
		self.encoder = Encoder(N_layers=5).to(self.device)
		self.decoder = Decoder(N_layers=5).to(self.device)

		# in:( batch_size x sequence_size x embed_size ) ->out:( batch_size x sequence_size x target_vocab_len )
		self.linear_to_vocab_dim = nn.Linear(embedding_size, target_vocab_len).to(self.device)

		self.positional_encodings = Positional_Encodings(self.d_model, self.device).to(self.device)
		
			
	@TensorInfo.show__tensor_sizes
	def forward(self, src_sequences: torch.Tensor, target_sequences: torch.Tensor): 
		"""
		In: 

			sequence_idx_input: batch of sequences to be translated as their indicies within the vocab
				["Hello", "world"] -> [4, 2]

				shape :  (batch x seq_length)

		"""

		# self.get_size_data(src_sequences, target_sequences)

		self.src_positional, self.target_positional = self.positional_encodings(src_sequences, target_sequences, 'both')


		# mask everything besides first position (<sos>)
		self.mask.reset()

		encoder_input_embeddings = self.get_input_embeddings(src_sequences, 'src')
		encoder_output_embeddings = self.encoder(encoder_input_embeddings) 					# out: batch_size, seq, embedding_size

		# mask everything except <sos> token embedding on first iteration
		# Mask will hide all words, target sequence can be replaced random tensor so long as each sequence starts with '<sos>'
		predicted_sequences = target_sequences

		
		while self.mask.finished == False:
				
			# mask everything except <sos> token embedding on first iteration
			decoder_input_embeddings =  self.get_input_embeddings(predicted_sequences, 'target')
			decoder_output 	= self.decoder(decoder_input_embeddings, encoder_output_embeddings) 	# out: batch_size, seq, embedding_size

			# Turn decoder output into target vocab rankings
			predicted_rankings  = self.linear_to_vocab_dim(decoder_output) 	# out-size: batch_size, seq, target_vocab_len

		self.print_translated_text(predicted_rankings, target_sequences)
		
		return predicted_rankings

	def get_input_embeddings(self, sequences_indicies, src_or_target: str):
		"""

			get embeddings from sequence indicies, and mask future and padded values if necessary

			sequences_indicies:  (batch x sequence_length)
			src_or_target:		 "src" or "target"

		"""


		if src_or_target == 'target':
			# add positional information out: batch_size, seq, embedding_size
			input_embeddings_with_positions = self.target_embeddings_lookup(sequences_indicies) + self.target_positional

			# mask everything except <sos> token embedding on first iteration
			masked_input_embeddings_with_pos_enc = self.mask( input_embeddings_with_positions)

		else:
			
			# add positional information out: batch_size, seq, embedding_size
			masked_input_embeddings_with_pos_enc = self.src_embeddings_lookup(sequences_indicies) + self.src_positional


		padded_masked_input_embeddings = self.pad_mask(masked_input_embeddings_with_pos_enc, sequences_indicies, src_or_target).type(torch.FloatTensor).to(self.device)

		
		return padded_masked_input_embeddings
	

	#TODO move to helper
	def print_translated_text(self, predicted_rankings, true_sequences, num_printed=3):

		
		# Unnecessary but reflect what the loss function is doing
		logsoft = nn.LogSoftmax(dim=-1)
		predicted_rankings = logsoft(predicted_rankings)
		
		# choose highest ranking word as output
		predicted_sequences = predicted_rankings.argmax(dim=-1)    		# out-size: batch_size, seq 

		for seq_i, sequence in enumerate(predicted_sequences):
			
			if seq_i >= num_printed: 
				break

			print("\nTranslated text:")

			for word_index in predicted_sequences[seq_i]: 
				print('', self.index_to_str.target[word_index], end=" ")

			print("\nTrue text:")

			for true_index in true_sequences[seq_i]: 
				print('', self.index_to_str.target[true_index], end=" ")

		
		print()
