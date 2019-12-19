from Transformer import Transformer
import torch.nn as nn


# TODO complete
def train(): 

    vocab = {"hi":0, "my":1, "name":2, "is": 3}

    target_vocab = {"hola": 0, "mi": 1, "nombre": 2, "es": 3}
    att_transformer = Transformer("Hi my name is", vocab, target_vocab ) 
    

    criterion = nn.CrossEnthropyLoss


def loss(criterion, output_embeddings, target, num_pred):
    
    index = len(output_embeddings) - num_pred
    
    # shift it all the way to the right
    output_embeddings[-index:] = output_embeddings[:num_pred]
    # mask values not yet predicted
    output_embeddings[:index] = 0
    target[:index] = 0
    

    loss = criterion(output_embeddings, target) 

    