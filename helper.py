import torch
from functools import wraps

from consts import DISPLAY_BOOL

class TensorInfo(): 

	indent_lvl = 0
	one_pass = False
	
	show_logs = DISPLAY_BOOL

	@classmethod
	def show__tensor_sizes(cls, func, display = show_logs):
		"""
		Print to console the input and ouput tensor shapes 
		
		Tab if a another method with this wrapper is called within the original call
		Untab once that finishes

		"""


		@wraps(func)
		def wrapper(*args, **kwargs): 
			
			if display:
				module_name = args[0].__class__.__name__

				if not cls.one_pass:
					indent =  "\t"*cls.indent_lvl
					print("\n" + indent + module_name + " INPUT tensors or args")

					cls.indent_lvl +=1


					for ar in args:
						if type(ar) == torch.Tensor:
							print(indent + " ", ar.shape, "grad func:", ar.grad_fn) 
						else:
							
							print(indent + "  arg",  type(ar))  


			# run functions
			result = func(*args, **kwargs)

			if display:

				if not cls.one_pass:

					cls.indent_lvl -=1
					if cls.indent_lvl == 0:
						cls.one_pass = True

					print(indent + module_name + " OUTPUT tensor sizes")

					if type(result) == torch.Tensor:
						print(indent, result.shape, result.grad_fn)
					elif type(result) == tuple:
						for ar in result:
							if type(ar) == torch.Tensor:
								print(indent, ar.shape, ar.grad_fn) 


			return result 
		
		return wrapper