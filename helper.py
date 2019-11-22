import torch

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
						{'queries': {'length': , 'depth': }, 'keys': {'length': , 'depth': }, 'values': {'length': , 'depth': }}

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