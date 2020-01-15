from Transformer import Transformer
from Dataset import Dataset
import torch.nn as nn
import torch
import os
import math

PATH = os.environ['ATT_PATH']

def print_params(model_state):
	for param in model_state:
		print(param)

def train():

	# Following the original papers hyperparameters
	betas = (0.90 , 0.98)
	num_epochs = 10

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print("Loading dataset...")
	dataset = Dataset(device=device, batch_size=10)
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
	
			transformer.load_state_dict(load_dict)
	
			print("Model loaded")

	except FileNotFoundError: 
		print("File not found error, continue blank slate")


	optimizer = torch.optim.Adam(transformer.parameters(), betas=betas, lr=1.5)

	train_steps_per_epoch = dataset.steps_per_epoch["train"]

	total_steps = train_steps_per_epoch * num_epochs
	ramp_up_steps = 30
	ramp_up_percentage =  ramp_up_steps / total_steps 

	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.45, total_steps=2000, pct_start=ramp_up_percentage)

	loss_fn = nn.CrossEntropyLoss(reduction='sum').to(device=device)
	loss = 0 

	loss_values = []
	update_step = 1


	for e in range(num_epochs):

		print("-----------Begin epoch", e)

		for i, batch in enumerate(dataset.train_iterator):

			#  (src_seq_length x batch) -> (batch x src_seq_length)
			src_sequences    = batch.src.permute(1,0)
			target_sequences = batch.trg.permute(1,0)


			
			print("batch",i, "Epoch", e)

			optimizer.zero_grad()
		
			#  (batch_size, target_vocab_len, seq), 
			predicted_rankings = transformer(src_sequences, target_sequences)
			perumted_predictions = predicted_rankings.permute(0,2,1) # out-size: batch_size, target_vocab_len, seq
		
			loss = loss_fn(perumted_predictions, target_sequences)

			print( "learning rate = ", scheduler.get_lr() )
			print("Loss = ", str(float(loss)) + "\n")

			
			# save model
			if i % 50 == 0: 
				torch.save(transformer.state_dict(), PATH  + "model_save.pt")

				print("saved model")
				with open( PATH + "values.txt", "w") as list_file: 
					list_file.write(str(loss_values))
	
				

			loss.backward()

			optimizer.step()

			scheduler.step()


		
train()


# TODO implement beam search
def beam_search():
	pass


