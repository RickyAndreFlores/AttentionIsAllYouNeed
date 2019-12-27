from Attention import Attention, MultiHeadedAttention, Decoder_MultiHeadedAttention
import torch
from torch import nn
from helper import TensorInfo


class FeedForward(nn.Module):

	def __init__(self, d_model=512, hidden=2048, device=torch.device('cuda')):

		super().__init__()

		self.device = device

		self.position_wise_linear = nn.Sequential(        
			nn.Conv1d( d_model, hidden, kernel_size=1),
			nn.ReLU(),
			nn.Conv1d( hidden, d_model, kernel_size=1), 
		).to(self.device)

		self.dropout = nn.Dropout().to(self.device)

		self.norm = nn.LayerNorm(d_model).to(self.device)

	@TensorInfo.show__tensor_sizes
	def forward(self, multi_atten_results: torch.Tensor):
		"""
		input :   ( batch_size x sequence_size x d_model==embed_size )

			middle: (batch_size x hidden x d_model)

		output:   ( batch_size x sequence_size x d_model )
		"""
		
		residual = multi_atten_results

		# output -> ( batch_size x d_model x sequence_size)
		output = self.position_wise_linear(multi_atten_results.permute(0,2,1))
		
		# if model.train(mode=True). TODO check that this condition is checked automatically
		output = self.dropout(output)

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

		layer_list = []
		for _ in range(N_layers): 
			layer_list.append(MultiHeadedAttention())
			layer_list.append(FeedForward())

		self.layers = nn.Sequential(*layer_list) 

	@TensorInfo.show__tensor_sizes
	def forward(self, input_embedding: torch.Tensor): 
		"""
		input :   ( batch_size x sequence_size x d_model==embed_size )
		output:   ( batch_size x sequence_size x d_model )


		Go through stack of N decoder layers, given (input_embeddings + positional_encoding)
		"""
		
		return self.layers(input_embedding)


class Decoder(nn.Module):
	"""
	inputs: 

		decoder_output_prev: torch.Tensor   ( batch_size x sequence_size x d_model )
		encoder_output: torch.Tensor		( batch_size x sequence_size x d_model )
		mask: torch.Tensor

	"""

	def __init__(self, N_layers: int = 6):

		super().__init__()

		self.layers_list = []
		for _ in range(N_layers): 
			self.layers_list.append(MultiHeadedAttention())  
			self.layers_list.append(Decoder_MultiHeadedAttention()) 
			self.layers_list.append(FeedForward())


		self.layers = nn.Sequential(*self.layers_list) 

	@TensorInfo.show__tensor_sizes
	def forward(self,  decoder_output_prev: torch.Tensor, encoder_output: torch.Tensor): 
		
		input_embed = decoder_output_prev

		
		for current_layer in self.layers: 

			# Continue through layers
			if current_layer.__class__.__name__ == 'MultiHeadedAttention':
				
				input_embed = current_layer(input_embed)

			elif current_layer.__class__.__name__ == 'Decoder_MultiHeadedAttention':
				
				# this sublayer requires the output of the encoder
				input_embed = current_layer(input_embed, encoder_output)
			
			elif current_layer.__class__.__name__ == 'FeedForward':
				
				# no mask for feedforward
				input_embed = current_layer(input_embed)

		# for readability
		decoder_stack_output = input_embed
		
		return decoder_stack_output