import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv1d(kargs['input_depth'], 64, 50, stride=6, padding=(50//2)),
            nn.ReLU(),
            nn.MaxPool1d(8, stride=8, padding=(8//2)),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, 8, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, 8, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4, padding=(4//2)),
        )
        
        self.cnn2 = nn.Sequential(
            nn.Conv1d(kargs['input_depth'], 64, 400, stride=50, padding=(400//2)),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=4, padding=(4//2)),
            nn.Dropout(0.5),
            nn.Conv1d(64, 128, 6, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, 6, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, 6, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2, padding=(2//2)),
        )

    def forward(self, x):
        out1 = self.cnn1(x)
        out2 = self.cnn2(x)
        
        flatten_out1 = torch.flatten(out1, start_dim=1)
        flatten_out2 = torch.flatten(out2, start_dim=1)

        flatten_out = torch.cat((flatten_out1, flatten_out2), dim = -1)
        out = F.dropout(flatten_out, 0.5)
        return out

class Encoder(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        self.lstm = nn.LSTM(2 * kargs['num_units'], kargs['num_units'], num_layers = kargs['lstm_layers'], bidirectional = True, batch_first = True)
    
    def forward(self, x):
        encoder_outputs, (encoder_state, _) = self.lstm(x)
        # 마지막 레이어의 양방향 hidden state 가져오기
        hidden_state = encoder_state[-2:, :, :]

        # 양방향 hidden state 연결
        hidden_state = hidden_state.transpose(0, 1).contiguous().view(x.size(0), -1)

        return encoder_outputs, hidden_state

class Attention(nn.Module):
    def __init__(self, **kargs):
        super(Attention, self).__init__()
        self.We = nn.Linear(2 * kargs['num_units'], 2 * kargs['num_units'],)
        self.Wh = nn.Linear(2 * kargs['num_units'], 2 * kargs['num_units'],)

    # encoder_hidden과 decoder_hidden과 shape이 같아야 함
    def forward(self, encoder_hidden, decoder_hidden):
        WE = self.We(encoder_hidden)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        WH = self.Wh(decoder_hidden)
        x = WE + WH
        f = torch.tanh(x)
        alpha = torch.nn.functional.softmax(f, dim=-1)
        c = alpha * encoder_hidden
        c = torch.sum(c, dim=1)
        return c

class Decoder(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        
        self.attention = Attention(**kargs)
        self.lstm = nn.LSTM(1+(2*kargs['num_units']), kargs['num_units'], num_layers = kargs['lstm_layers'], bidirectional = True, batch_first = True)
        self.classes = nn.Linear(2 * kargs['num_units'], kargs['max_time_step'])
        # self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, decoder_input, prev_decoder_hidden, encoder_hidden):
        c = self.attention(encoder_hidden, prev_decoder_hidden)
        decoder_input = decoder_input.view(decoder_input.shape[0], -1)
        x = torch.cat([c, decoder_input], dim=-1).unsqueeze(1)
        x, h = self.lstm(x) # .squeeze(1)
        prediction = torch.nn.functional.softmax(self.classes(x))
        return prediction, h

class Model(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        
        self.cnn = CNN(**kargs)
        self.encoder = Encoder(**kargs)
        self.decoder = Decoder(**kargs)
        
    def forward(self, x, dec_input):
        cnnout = []
        for i in range(x.shape[1]):
            _cnnout = self.cnn(x[:,i])
            _cnnout = _cnnout.unsqueeze(1)
            cnnout.append(_cnnout)
        cnnout = torch.cat(cnnout, dim = 1)

        out, hidden = self.encoder(cnnout)
        out, h = self.decoder(dec_input, hidden, out)
        
        return out, h