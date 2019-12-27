from Transformer import Transformer
from Dataset import Dataset
import torch.nn as nn
import torch
import os

PATH = os.environ['ATT_PATH']

	
def print_params(model_state):
	for param in model_state:
		print(param)

def train():

	# Following the original papers hyperparameters
	betas = (0.90 , 0.98)
	num_epochs = 1000

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print("Loading dataset...")
	dataset = Dataset(device=device, batch_size=200)
	print("dataset loaded")


	transformer = Transformer(dataset.src_vocab_len, dataset.target_vocab_len, dataset.str_to_index, dataset.index_to_str)
	transformer.to(device)

	transformer.train(mode=True)

	LOAD = True

	try:
		if LOAD:

			load_dict = torch.load(PATH  + "model_save.pt")

			try:
				del load_dict['positional_encodings.target_pos_enc_lookup.weight']
				del load_dict['positional_encodings.src_pos_enc_lookup.weight']
			except KeyError:
				print('no need to delete')
			# load_dict['linear_to_vocab_dim.weight'] = load_dict['get_probabilites.0.weight']
			# load_dict['linear_to_vocab_dim.bias']   = load_dict['get_probabilites.0.bias']
			# del load_dict['get_probabilites.0.weight']
			# del load_dict['get_probabilites.0.bias']
			
			transformer.load_state_dict(load_dict)
	
			torch.save(transformer.state_dict(), PATH  + "model_save.pt")

			print("Model loaded")

	except FileNotFoundError: 
		print("File not found error, continue blank slate")


	optimizer = torch.optim.Adam(transformer.parameters(), betas=betas)	
	# TODO :  update learning rate according to funtion in paper
	# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer)

	train_steps_per_epoch = dataset.steps_per_epoch["train"]
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr= 8.0, steps_per_epoch= train_steps_per_epoch, epochs=num_epochs)


	loss_fn = nn.CrossEntropyLoss()
	loss = 0 

	loss_values = []
	# values_epoch = []

	update_step = 1

	# TODO update graph.png
	for e in range(num_epochs):

		print("Begin epoch", e)

		for i, batch in enumerate(dataset.train_iterator):

			#  (src_seq_length x batch) -> (batch x src_seq_length)
			src_sequences    = batch.src.permute(1,0)
			target_sequences = batch.trg.permute(1,0)

			optimizer.zero_grad()

			print("batch",i, "Epoch", e)
			#  (batch_size, target_vocab_len, seq), 
			perumted_predictions = transformer(src_sequences, target_sequences)
			loss += loss_fn(perumted_predictions, target_sequences)

			print("Loss = ", str(float(loss)) + "\n")

			if  i %  1  == 0:

				training_step =  e*train_steps_per_epoch + i

				# add to record of loss
				loss_values.append([training_step, float(loss)] )


			if i % 20 == 0: 
				torch.save(transformer.state_dict(), PATH  + "model_save.pt")
			
				with open( PATH + "values.txt", "w") as list_file: 
					list_file.write(str(loss_values))
	
			# update weights every int(update_step)s iterartions
			if i % 1 == 0: 

				loss.backward()
				# scheduler.step()
				# print("scheduler")
				optimizer.step()

				loss = 0 


		scheduler.step()

		# values_epoch.append([e, loss] )
		
		# if e % 1 == 0: 
		# 	torch.save(transformer.state_dict(), PATH  + "model_save.pt")

		# 	with open( PATH + "values_epoch.txt", "w") as list_file: 
		# 		list_file.write(str(values_epoch))

train()




def filter_no_grad(model_state): 
	
	for param in model_state:
		print(param.requires_grad)

	

def visualize(i, loss):

	pass






# TODO make sure this implementation is correct
def learning_rate(d_model, step_num, warmup_steps=4000):

	learning_rate = d_model**(-0.5) * min(step_num**(-0.5) , step_num*warmup_steps**(-1.5))  

	return learning_rate


# TODO complete
def eval(): 
	pass

	