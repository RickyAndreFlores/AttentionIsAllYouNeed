import torch
from torch import nn
from math import sqrt


class Attention(nn.Module): 

    def __init__(self, D_in, H):

        super.__init__()
        
        self.linear_q = torch.nn.Linear(D_in, H)
        self.linear_k = torch.nn.Linear(D_in, H)
        self.linear_v = torch.nn.Linear(D_in, H)
        self.softmax  = torch.nn.Softmax(dim=2)

        pass

    def forward(self, Queries: torch.Tensor, Keys: torch.Tensor, Values: torch.Tensor, mask: bool): 
        
        Q = self.linear_q(Queries)
        K = self.linear_k(Keys)
        V = self.linear_v(Values)

        # math.sqrt if not tensor
        shrinking_weight = sqrt(Q.size()[0])

        # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
        a = torch.bmm(Q, K.transpose(2,1)) / shrinking_weight

        if mask:
            # TODO mask 
            print("mask")

        # Softmax values along dimension 2 
        
        v_filter = self.softmax(a)
        results = torch.bmm(v_filter, V)

        return results


class MultiHeadedAttention(Attention): 

    def __init__(self, D_in, H):
        super().__init__()


    def forward(self, Queries: torch.Tensor, Keys: torch.Tensor, Values: torch.Tensor, mask: bool): 

        Q, K, V = None # TODO some formated version 

        Attention.forward(self, Q, K, V, mask)
        
        
        # Attention()  in parralell - inlcudes lineawr  already
        # TODO concat
        # TODO other linear 


        #TODO  add residual and norm
        
        pass

class FeedForward(nn.Module):

    def __init__(self, in_channels, out_channels):


        self.position_wise_linear = nn.Sequential(        
            nn.Conv1d( in_channels, out_channels, kernel_size=1),
            nn.Conv1d( in_channels, out_channels, kernel_size=1), 
            nn.LayerNorm()
        )
        

    def forward(self, atten_results: torch.Tensor):

        output = self.position_wise_linear(atten_results)


        # TODO add attention residual and norm 


        return


class Encoder(FeedForward, MultiHeadedAttention): 

    def __init__(self): 
        FeedForward.__init__(self)
        MultiHeadedAttention.__init__(self)

    def forward(self): 
        
        outputs = MultiHeadedAttention.forward( )
        queries, keys = FeedForward()

        return queries, keys

class Decoder(Encoder): 

    def __init__(self):
        super().__init__()

    def forward(self): 

        masked_values = super().MultiHeadedAttention(Q, K, V, mask =true)
        encoded_q, encoded_K = Encoder.forward()

        out = super().MultiheadedAttention(encoded_K, encoded_q, masked_values)
        results = super().FeedForward(out)



class Transformer(Decoder):

    def __init__(self):
        super().__init__()
        self.get_probabilites = nn.Sequential( [ 
            nn.Linear(), 
            nn.Softma()
        ])

    def forward(self): 

        results = super().forward()
        output_prob = self.get_probabilites(results)
        
        return output_prob
        