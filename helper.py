import torch
from functools import wraps

class TensorInfo(): 

	indent_lvl = 0
	one_pass = False

	@classmethod
	def show__tensor_sizes(cls, func):
		"""
		Print to console the input and ouput tensor shapes 
		
		Tab if a another method with this wrapper is called within the original call
		Untab once that finishes

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