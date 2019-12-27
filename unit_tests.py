import torch

seq = 3
batch_size = 2
embedding_size:int = 512
num_heads:int = 6
depth_q: int = 64
depth_k: int = 64
depth_v: int = 64

device = torch.device('cuda')
previous = torch.randn(batch_size, seq, embedding_size)

src_sequences_idx = torch.randn(2,4).to(device)
target_sequences_idx = torch.randn(2,7).to(device)

src_sequences_embed = torch.randn(2,4, 5).to(device)
target_sequences_embed = torch.randn(2,7, 5).to(device)

def unit_test_attention(): 
	from Attention import Attention

	att = Attention()


	x_list = [torch.randn(batch_size, num_heads, seq, depth_k), torch.randn(batch_size, num_heads, seq, depth_k), torch.randn(batch_size, num_heads, seq, depth_v)]

	y = att(*x_list)

	# print("y", att(keys=x_list[1], queries=x_list[0],values=x_list[2], mask=True))
	print("y", y.shape , "== size:",   num_heads,"*",batch_size, ",", seq, ",", depth_v)

	return y

# unit_test_attention()


def unit_mult_attention(): 
	from Attention import MultiHeadedAttention

	multi = MultiHeadedAttention()

	previous = torch.randn(batch_size, seq, embedding_size)

	y = multi(previous)

	# print("result", y)

# unit_mult_attention()


def unit_feed_forward(): 
	from EncoderDecoder import FeedForward

	ff = FeedForward()
 
	y = ff(previous)

# unit_feed_forward()

def unit_encoder(): 
	from EncoderDecoder import Encoder

	en = Encoder()

	emeddings = torch.randn(batch_size, seq, embedding_size)
	
	out = en(emeddings)

# unit_encoder()


def unit_pos_encoder():

	from PositionsMasks import Positional_Encodings

	src_sequences_idx = torch.randn(2,4).to(device)
	target_sequences_idx = torch.randn(2,7).to(device)

	positional_encodings = Positional_Encodings(embedding_size)

	src_encodings, trg_encodings = positional_encodings(src_sequences_idx, target_sequences_idx)

	
	print('src', src_sequences_idx.shape)
	print('target', target_sequences_idx.shape)

	print('src', src_encodings.shape)
	print('target', trg_encodings.shape)
	
	print('src', src_encodings)
	print('target', trg_encodings)
	
	print('src', positional_encodings.src_positions_matrix)
	print('target', positional_encodings.target_positions_matrix)

	positional_encodings(src_sequences_idx, target_sequences_idx)

	src_sequences_idx = torch.randn(1,4).to(device)
	target_sequences_idx = torch.randn(1,7).to(device)

	positional_encodings(src_sequences_idx, target_sequences_idx)

	src_sequences_idx = torch.randn(2,5).to(device)
	target_sequences_idx = torch.randn(1,7).to(device)

	positional_encodings(src_sequences_idx, target_sequences_idx)

unit_pos_encoder()


def unit_decoder_attention(): 
	from Attention import Decoder_MultiHeadedAttention


	encoder_output = torch.randn(batch_size, seq, embedding_size)
	previous = torch.randn(batch_size, seq, embedding_size)

	decoder_multi = Decoder_MultiHeadedAttention()

	print(decoder_multi.__class__.__name__)

	y = decoder_multi(previous, encoder_output)


# unit_decoder_attention()



def unit_decoder(): 
	from EncoderDecoder import Decoder

	decoder = Decoder()

	emeddings = torch.randn(batch_size, seq, embedding_size)
	encoder_emeddings = torch.randn(batch_size, seq, embedding_size)

	out = decoder(emeddings, encoder_emeddings, None)

# unit_decoder()


def mask(): 

	from Transformer import Mask
	from dataclasses import dataclass

	print(src_sequences_embed)



	mask = Mask()

	for i in range(src_sequences_embed.shape[1] + 1): 

		masked_embed = mask(src_sequences_embed)

		print(mask.batch_mask)
		print("masked_embed", masked_embed)
	
# mask()


def pad_mask(): 

	from Transformer import Pad_Mask, Mask
	from dataclasses import dataclass


	src_sequences_embed = torch.randn(2,4, 5)

	# print(src_sequences_embed)

	token_dict = { '<pad>': 0, '<sos>': 1 , '<eos>': 2 }

	@dataclass
	class stoi(): 
		src: dict
		target: dict

	stoi_ex = stoi(token_dict, token_dict)

	pad_mask = Pad_Mask(stoi_ex, 5)

	seq_indx = torch.Tensor([ [0,1,2,0], [1,3,0,1] ])
	print(seq_indx)
	print("pad_mask")
	pad_result = pad_mask.get_pad_mask(seq_indx, 'src' )


	mask = Mask()

	for i in range(src_sequences_embed.shape[1]): 

		masked_embed = mask(src_sequences_embed)

		print("masked_embed", masked_embed, "\n")

		masked_pad_embed = pad_mask(masked_embed, seq_indx, 'target')
		print(pad_result)
		print("	masked_pad\n", masked_pad_embed , "\n")



# pad_mask()