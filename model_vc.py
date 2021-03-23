import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class ConvNorm1d(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, 
                 padding=None, dilation=1, bias=True, w_init_gain='linear'): 
        super(ConvNorm1d, self).__init__() 
        if padding is None: 
            assert(kernel_size % 2 == 1) 
            padding = int(dilation * (kernel_size - 1) / 2) 
 
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=kernel_size, stride=stride, 
                                    padding=padding, dilation=dilation, 
                                    bias=bias) 
 
        torch.nn.init.xavier_uniform_( 
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)) 

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class ConvNorm2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm2d, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class ConvT2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='relu'):
        super(ConvT2d, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
            
        self.conv = torch.nn.ConvTranspose2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)),

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal 

## "4.2. The Content Encoder"
#class Encoder(nn.Module):
#    """Encoder module:
#    """
#    def __init__(self, dim_neck, dim_emb, freq):
#        super(Encoder, self).__init__()
#        self.dim_neck = dim_neck
#        self.freq = freq
#        convolutions = []
#        for i in range(3):
#        # "the input to the content encoder is the 80-dimensional mel-spectrogram of X1 concatenated with the speaker embedding" - I think the embeddings are copy pasted from a dataset, as the Speaker Decoder is pretrained and may not actually appear in this implementation?
#            conv_layer = nn.Sequential(
#        # "the input to the content encoder is the 80-dimensional mel-spectrogram of X1 concatenated with the speaker embedding. The concatenated features are fed into three 5 × 1 convolutional layers, each followed by batch normalization and ReLU activation. The number of channels is 512"
#                ConvNorm(80+dim_emb if i==0 else 512,
#                         512,
#                         kernel_size=5, stride=1,
#                         padding=2,
#                         dilation=1, w_init_gain='relu'),
#                nn.BatchNorm1d(512))
#            convolutions.append(conv_layer)
#        self.convolutions = nn.ModuleList(convolutions)
#        
#        # "Both the forward and backward cell dimensions are 32, so their (LSTMs) combined dimension is 64."
#        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)
#
#        # c_org is speaker embedding
#    def forward(self, x, c_org):
#        x = x.squeeze(1).transpose(2,1)
#        #pdb.set_trace()
#        # broadcasts c_org to a compatible shape to merge with x
#        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
#        x = torch.cat((x, c_org), dim=1)
#        for conv in self.convolutions:
#            x = F.relu(conv(x))
#        x = x.transpose(1, 2)
#        
#        self.lstm.flatten_parameters()
#        # lstms output 64 dim
#        outputs, _ = self.lstm(x)
#        # backward is the first half of dimensions, forward is the second half
#        # pdb.set_trace()
#        out_forward = outputs[:, :, :self.dim_neck]
#        out_backward = outputs[:, :, self.dim_neck:]
#
#        # pdb.set_trace()
#        codes = []
#        
#        # for each timestep, skipping self.freq frames
#        for i in range(0, outputs.size(1), self.freq):
#            # remeber that i is self.freq, not increments of 1)
#            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))
#        
#        # if self.freq is 32, then codes is a list of 4 tensors of size 64
#        return codes

# "4.2. The Content Encoder"
class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, dim_emb, freq):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        self.freq = freq
        convolutions = []
        for i in range(4):
        # "the input to the content encoder is the 80-dimensional mel-spectrogram of X1 concatenated with the speaker embedding" - I think the embeddings are copy pasted from a dataset, as the Speaker Decoder is pretrained and may not actually appear in this implementation?
            conv_layer = nn.Sequential(
        # "the input to the content encoder is the 80-dimensional mel-spectrogram of X1 concatenated with the speaker embedding. The concatenated features are fed into three 5 × 1 convolutional layers, each followed by batch normalization and ReLU activation. The number of channels is 512"
                ConvNorm2d(1 if i==0 else 64 if i==1 else 128 if i==2 else 256,
                         64 if i==0 else 128 if i==1 else 256 if i==2 else 512,
                         kernel_size=3, stride=1,
                         padding=1,
                         dilation=1, w_init_gain='relu')
                ,nn.BatchNorm2d(64 if i==0 else 128 if i==1 else 256 if i==2 else 512)
                ,nn.ReLU()
                ,nn.MaxPool2d((4,1))
            )
                
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        
        # "Both the forward and backward cell dimensions are 32, so their (LSTMs) combined dimension is 64."
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

        # c_org is speaker embedding
    def forward(self, x, c_org):
        x = x.transpose(2,1).unsqueeze(1) # after this transpose, tensor should be shape (batch, feature, time)
        # broadcasts c_org to a compatible shape to merge with x
        c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1)).unsqueeze(1)
        x = torch.cat((x, c_org), dim=2)
        saved_enc_outs = [x] ###
        for conv in self.convolutions:
            x = conv(x)
            saved_enc_outs.append(x) ###
        x = x.squeeze(2).transpose(-1,-2)
        self.lstm.flatten_parameters()
        # lstms output 64 dim
        outputs, _ = self.lstm(x)
        # backward is the first half of dimensions, forward is the second half
        saved_enc_outs.append(outputs.transpose(2,1)) ###
        out_forward = outputs[:, :, :self.dim_neck] #takes the first half of outputs
        out_backward = outputs[:, :, self.dim_neck:]

        codes = []
        
        # for each timestep, skipping self.freq frames
        for i in range(0, outputs.size(1), self.freq):
            # remeber that i is self.freq, not increments of 1)
            codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))
        # this saves codes as an entire tensor instead of list
        #saved_enc_outs.append(torch.cat(codes, dim=0)) 
        # if self.freq is 32, then codes is a list of 4 tensors of size 64
        return codes, saved_enc_outs

#class Decoder(nn.Module):
#    """Decoder module:
#    """
#    def __init__(self, dim_neck, dim_emb, dim_pre):
#        super(Decoder, self).__init__()
#        
#        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
#        
#        convolutions = []
#        for i in range(3):
#            conv_layer = nn.Sequential(
#                ConvNorm(dim_pre,
#                         dim_pre,
#                         kernel_size=5, stride=1,
#                         padding=2,
#                         dilation=1, w_init_gain='relu'),
#                nn.BatchNorm1d(dim_pre))
#            convolutions.append(conv_layer)
#        self.convolutions = nn.ModuleList(convolutions)
#        
#        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
#        self.linear_projection = LinearNorm(1024, 80)
#
#    def forward(self, x):
#        
#        #self.lstm1.flatten_parameters()
#        x, _ = self.lstm1(x)
#        x = x.transpose(1, 2)
#        
#        for conv in self.convolutions:
#            x = F.relu(conv(x))
#        x = x.transpose(1, 2)
#        
#        outputs, _ = self.lstm2(x)
#        
#        decoder_output = self.linear_projection(outputs)
#
#        return decoder_output   

class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre):
        super(Decoder, self).__init__()
        
        self.lstm1 = nn.LSTM(dim_neck*2+dim_emb, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvT2d(1 if i==0 else 64 if i==1 else 64,
                        64 if i==0 else 64 if i==1 else 64,
                        kernel_size=5,
                        stride=1,
                        padding=2),
                nn.BatchNorm2d(64 if i==0 else 64 if i==1 else 64))
            convolutions.append(conv_layer) 
        self.convolutions = nn.ModuleList(convolutions)         

        self.lstm2 = nn.LSTM(dim_pre*64, 1024, 2, batch_first=True)
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        #self.lstm1.flatten_parameters()
        saved_dec_outs = [x.transpose(2,1)]
        x, _ = self.lstm1(x)
        saved_dec_outs.append(x.transpose(2,1)) ###
        x = x.transpose(1, 2).unsqueeze(1)
        saved_dec_outs.append(x) ###
        for i, conv in enumerate(self.convolutions):
            x = F.relu(conv(x))
            saved_dec_outs.append(x) ###
        x = x.reshape(x.size(0), x.size(1)*x.size(2), -1).transpose(1,2)
        outputs, _ = self.lstm2(x)
        saved_dec_outs.append(outputs.transpose(2,1)) ###
        decoder_output = self.linear_projection(outputs)
        saved_dec_outs.append(decoder_output.transpose(2,1)) ###
        return decoder_output, saved_dec_outs

# Still part of Decoder as indicated in paper Fig. 3 (c) - last two blocks 
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm1d(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm1d(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm1d(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    

#class Postnet(nn.Module):
#    """Postnet
#        - Five 1-d convolution with 512 channels and kernel size 5
#    """
#
#    def __init__(self):
#        super(Postnet, self).__init__()
#        self.convolutions = nn.ModuleList()
#
#        self.convolutions.append(
#            nn.Sequential(
#                ConvNorm2d(256, 128,
#                         kernel_size=5, stride=1,
#                         padding=(0,2),
#                         dilation=1, w_init_gain='tanh'),
#                nn.BatchNorm2d(128),
#                nn.MaxPool2d((3,1))),
#        )
#
#        for i in range(1, 5 - 1):
#            self.convolutions.append(
#                nn.Sequential(
#                    ConvNorm2d(128 if i==1 else 64,
#                             64,
#                             kernel_size=5, stride=1, padding=(1,2) if i<1 else 2,
#                             dilation=1, w_init_gain='tanh'),
#                    nn.BatchNorm2d(64),
#                    nn.MaxPool2d((2,1) if i==1 else (1,1))),
#            )
#
#        self.convolutions.append(
#            nn.Sequential(
#                ConvNorm2d(64, 1,
#                         kernel_size=5, stride=1,
#                         padding=(0,2),
#                         dilation=1, w_init_gain='linear'))
#
#            )
#
#    def forward(self, x):
#        for i in range(len(self.convolutions) - 1):
#            print(i, x.shape)
#            x = torch.tanh(self.convolutions[i](x))
#        x = self.convolutions[-1](x)
#        pdb.set_trace()
#        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_emb, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb, freq)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x, c_org, c_trg):

        # codes is a LIST of tensors                
        codes, saved_enc_outs = self.encoder(x, c_org)
        # if no c_trg given, then just return the formatted encoder codes
        if c_trg is None:
            # concatenates the by stacking over the last (in 2D this would be vertical) dimensio by stacking over the last (in 2D this would be vertical) dimension. For lists it means the same
            return torch.cat(codes, dim=-1)
        # list of reformatted codes        
        tmp = []
        for code in codes:
            # reformatting tmp from list to tensor, and resample it at the specified freq
            tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        code_exp = torch.cat(tmp, dim=1)
        # concat reformated encoder output with target speaker embedding
        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,x.size(1),-1)), dim=-1)
        mel_outputs, saved_dec_outs = self.decoder(encoder_outputs)
        # then put mel_ouputs through remaining postnet section of NN
        # the postnet process produces the RESIDUAL information that gets added to the mel output
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        #pdb.set_trace() 
        # add together, as done in Fig. 3 (c) ensuring the mel_out_psnt is same shape (2,128,80). new mel_out_psnt will be the same
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
       
        #insert channel dimension into tensors to become (2,1,128,80)
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1), saved_enc_outs, saved_dec_outs

