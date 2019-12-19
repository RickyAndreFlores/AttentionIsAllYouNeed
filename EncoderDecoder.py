from Attention import Attention, MultiHeadedAttention, Decoder_MultiHeadedAttention
import torch
from torch import nn
from helper import TensorInfo


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
		
		return self.encoder_layers(input_embedding)


class Decoder(nn.Module): 

	def __init__(self, encoder_output: torch.Tensor, N_layers: int = 6):
		super().__init__()

		self.encoder_output = encoder_output

		layers = []
		for _ in range(N_layers): 
			layers.append(MultiHeadedAttention())  #TODO masked
			layers.append(Decoder_MultiHeadedAttention(encoder_output)) 
			layers.append(FeedForward())

		# layers = [ MultiHeadedAttention(), FeedForward() for _ in range(N_layer) ]
		self.decoder_layers = nn.Sequential(*layers) 

	def forward(self,  output_pred_embeds: torch.Tensor, mask: torch.Tensor): 

		decoder_stack_output =  self.decoder_layers(output_pred_embeds, mask)

		return decoder_stack_output