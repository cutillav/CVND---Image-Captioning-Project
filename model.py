import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        embed_size : Size of embedding
        hidden_size : Number of nodes in the hidden layer
        vocab_size : The size of vocabulary or output size
        num_layers : Number of LSTM layers        
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size,hidden_size = hidden_size,num_layers = num_layers,batch_first = True)
        self.linear = nn.Linear(in_features = hidden_size, out_features = vocab_size)
        
    
    def forward(self, features, captions):
        # Remmove the final token - (batch_size, caption_length - 1)
        captions = captions[:, :-1]
        
        # Embed the captions
        embeddings = self.embed(captions)
        
        # Concatenate features and vectors
        final_embeddings = torch.cat((features.unsqueeze(dim = 1), embeddings), dim = 1)
        
        # Get the output and the state from lstm
        lstm_output, _ = self.lstm(final_embeddings)
        
        # Get the final words
        outputs = self.linear(lstm_output)
        
        return outputs


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass