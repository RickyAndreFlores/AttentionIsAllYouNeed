# from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch
from dataclasses import dataclass


class Dataset(): 
    """
    Given a dataset ( default is Multi30k) provide various useful datastructures

    For looping through train/valid/test sets use self.train_iterator, self.valid_iterator, self.test_iterator 
        example:
        For i, batch in enumerate(dataset.train_iterator):
            source_batch = batch.src  # size = (pad_seq_for_this_batch x batch_size)
            target_batch = batch.trg 

    For coverting indexes to strings or vice versa use  self.str_to_index, self.index_to_str

    and other data is available
  
    """

    from torchtext.datasets import Multi30k
    default_datset = Multi30k

    def __init__(self, dataset = default_datset, device = torch.device('cuda'),  source_lang='de', target_lang='en', batch_size = 20):
        
        
        self.dataset = dataset
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.batch_size = batch_size

        

        self.source_field = self.build_field(self.source_lang)
        self.target_field = self.build_field(self.target_lang)

        self.train_data, self.valid_data, self.test_data = self.create_datasets()

        # Size data 
        self.train_len, self.vaild_len, self.test_len = len(self.train_data),  len(self.valid_data),  len(self.test_data)
        num_batches = lambda dataset_size: int(dataset_size / self.batch_size)
        self.steps_per_epoch =  { "train":  num_batches(self.train_len), "vaild": num_batches(self.vaild_len), "test": num_batches(self.test_len) } 


        self.vocab_built = False # This will be changesd in next call 
       
        self.build_vocab()

        # conditonal is more for readability / clearly show dependence of previous action
        if self.vocab_built: 

            self.str_to_index, self.index_to_str  = self.create_string_index_converters()
            
            self.src_vocab_len    = len(self.source_field.vocab)
            self.target_vocab_len = len(self.target_field.vocab)


        self.train_iterator, self.valid_iterator, self.test_iterator = self.create_iterator(device)

    def build_field(self,  tokenizer_language: str,
                            tokenize: str="spacy",   
                            init_token: str = '<sos>',
                            eos_token: str = '<eos>',
                            lower: bool = True):

        # Function that tokenizes, converts to numbers, lowercases, adds extra tokens
        field = Field(tokenize = tokenize,
                    tokenizer_language= tokenizer_language,
                    init_token = init_token,
                    eos_token = eos_token,
                    lower = lower)

        return field


    def create_datasets(self):

        # apply above functions to dataset Multi30k,then split them into train/valid/test
        return self.dataset.splits(exts = ( '.' + self.source_lang, '.' +  self.target_lang),
                                 fields = (self.source_field,  self.target_field))


    def build_vocab(self): 

        # Go through data with SRC field, and register in vocab (record its index)
        self.source_field.build_vocab(self.train_data, min_freq = 2)
        #  Same as above but with TRG field
        self.target_field.build_vocab(self.train_data, min_freq = 2)

        self.vocab_built = True



    def create_string_index_converters(self):

        @dataclass
        class Converter():
            target: list or dict
            src: list or dict


        str_to_index = Converter(self.target_field.vocab.stoi, self.source_field.vocab.stoi)
        index_to_str = Converter(self.target_field.vocab.itos, self.source_field.vocab.itos)

        return str_to_index, index_to_str 

    def create_iterator(self, device): 


        # create iterators that outputs a batch of data samples 
        return BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size = self.batch_size,
            device = device)


