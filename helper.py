import torch
from functools import wraps

class TensorPrep():

	@staticmethod
	def attention_get_dims(attention):
		
		@wraps(attention)
		def wrapper(*args, **kwargs): 
			"""
					In
								queries: [batch, heads, length_q, depth_k]
								keys:    [batch, heads, length_kv, depth_k]
								values:  [batch, heads, length_kv, depth_v]
					out
						{'queries': {'length': , 'depth': }, 'keys': {'length': , 'depth': }, 'values': {'length': , 'depth': }}

			"""
			print("get dims")
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



	# @staticmethod
	# def residual_addnorm(sublayer):


	# 	@wraps(attention)
	# 	def wrapper(*args, **kwargs): 

	# 		residual = *args

	# 		# wont work to well must be learned 
	# 		norm = torch.nn.LayerNorm(residual.shape[-1])

	# 		output = sublayer(*args, **kwargs)

	# 		return norm(residul + residual)

		
	# 	return wrapper


class TensorInfo(): 

	indent_lvl = 0
	one_pass = False

	@classmethod
	def show__tensor_sizes(cls, func):
		"""
		Print to console the input and ouput tensor shapes 

		"""
		@wraps(func)
		def wrapper(*args, **kwargs): 

			
			if not cls.one_pass:
				indent =  "\t"*cls.indent_lvl
				print("\n" + indent + "Input tensor sizes:")
				cls.indent_lvl +=1

				for ar in args:
					if type(ar) == torch.Tensor:
						print(indent, ar.shape, ar.grad_fn) 

			# run functions
			result = func(*args, **kwargs)


			if not cls.one_pass:

				cls.indent_lvl -=1
				if cls.indent_lvl == 0:
					cls.one_pass = True

				print(indent + "output tensor sizes")

				if type(result) == torch.Tensor:
					print(indent, result.shape, result.grad_fn)
				elif type(result) == tuple:
					for ar in result:
						if type(ar) == torch.Tensor:
							print(indent, ar.shape, ar.grad_fn) 


			print()
			return result 
		
		return wrapper