import os
from datetime import datetime
from dataloader import SeqDataLoader
import time
import numpy as np
from imblearn.over_sampling import SMOTE
import torch
from SleepEEGNet import Model
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

num_folds = 20
data_dir = 'data/data2013/eeg_fpz_cz'
classes = ['W', 'N1', 'N2', 'N3', 'REM']
checkpoint_dir = 'checkpoints-seq2seq-sleep-EDF'
output_dir = 'outputs_2013/outputs_eeg_fpz_cz'

kargs = {
    'epochs': 120,
    'batch_size': 20,
    'num_units': 128,
    'embed_size': 10,
    'input_depth': 3000,
    'n_channels': 100,
    'bidirectional': False,
    'use_attention': True,
    'lstm_layers': 2,
    'attention_size': 64,
    'beam_width': 4,
    'use_beamsearch_decode': False,
    'max_time_step': 10,
    'output_max_length': 10 + 2,
    'akara2017': True,
    'test_step': 5
}
def MFE(y, y_hat, class_labels):
    # y and y_hat are tensors of shape (batch_size,)
    # class_labels is a list of unique class labels [W, N, ...]

    loss_MFE = 0.0
    loss_MSFE = 0.0
    for c_i in class_labels:
        idx = (y == c_i)  # Get the indices of samples in class c_i
        y_c_i = y[idx]
        y_hat_c_i = y_hat[idx]
        class_loss = ((y_c_i - y_hat_c_i) ** 2).mean()  # Compute the mean of squared differences
        loss_MFE += class_loss
        loss_MSFE += class_loss ** 2

    return torch.tensor(loss_MFE).to('cuda'), torch.tensor(loss_MSFE).to('cuda')

def train():
    model = Model(**kargs)
    path, channel_ename = os.path.split(data_dir)
    traindata_dir = os.path.join(os.path.abspath(os.path.join(data_dir, os.pardir)),'traindata/')

    fold_idx = 0
    for fold_idx in range(num_folds):
        data_loader = SeqDataLoader(data_dir, num_folds, fold_idx, classes=classes)
        X_train, y_train, X_test, y_test = data_loader.load_data(seq_len=kargs['max_time_step'])

        char2numY = dict(list(zip(classes, list(range(len(classes))))))        

        char2numY['<SOD>'] = len(char2numY)
        char2numY['<EOD>'] = len(char2numY)
        num2charY = dict(list(zip(list(char2numY.values()), list(char2numY.keys()))))
        fname = os.path.join(traindata_dir,'trainData_'+channel_ename+'_SMOTE_all_10s_f'+str(fold_idx)+'.npz')


        # over-sampling: SMOTE:
        X_train = np.reshape(X_train,[X_train.shape[0]*X_train.shape[1],-1])
        y_train= y_train.flatten()

        nums = []
        for cl in classes:
            nums.append(len(np.where(y_train == char2numY[cl])[0]))

        if (os.path.exists(traindata_dir) == False):
            os.mkdir(traindata_dir)

        if (os.path.isfile(fname)):
            X_train, y_train,_ = data_loader.load_npz_file(fname)
        else:
            # oversampling
            n_osamples = nums[2] - 7000
            ratio = {0: n_osamples if nums[0] < n_osamples else nums[0], 1: n_osamples if nums[1] < n_osamples else nums[1],
                        2: nums[2], 3: n_osamples if nums[3] < n_osamples else nums[3], 4: n_osamples if nums[4] < n_osamples else nums[4]}

            sm = SMOTE(random_state=12, sampling_strategy=ratio)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            data_loader.save_to_npz_file(X_train, y_train,data_loader.sampling_rate,fname)

            print(n_osamples)
            print(ratio)

        X_train = X_train[:(X_train.shape[0] // kargs['max_time_step']) * kargs['max_time_step'], :]
        y_train = y_train[:(X_train.shape[0] // kargs['max_time_step']) * kargs['max_time_step']]

        X_train = np.reshape(X_train,[-1,X_test.shape[1],X_test.shape[2]])
        y_train = np.reshape(y_train,[-1,y_test.shape[1],])


        # shuffle training data_2013
        permute = np.random.permutation(len(y_train))
        X_train = X_train[permute]
        y_train = y_train[permute]

        # add '<SOD>' to the beginning of each label sequence, and '<EOD>' to the end of each label sequence (both for training and test sets)
        # y_train= [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_train]
        # y_train = np.array(y_train)

        # y_test= [[char2numY['<SOD>']] + [y_ for y_ in date] + [char2numY['<EOD>']] for date in y_test]
        # y_test = np.array(y_test)

        # decoder_input = torch.zeros(y_train.shape[0], 7)  # Assuming <SOD> token representation
        # decoder_input[:, char2numY['<SOD>']] = 1

        learning_rate = 1e-3
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        
        # decoder_input = F.one_hot(torch.tensor(char2numY['<SOD>']), num_classes=7).float()
        # decoder_input = decoder_input.expand(X_train.shape[0], -1)  # <SOD>로 시작
        # decoder_input = torch.ones((7547, 1)) + 4

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # # Define the embedding layer
        # embedding_layer = nn.Embedding(num_embeddings=len(char2numY), embedding_dim=10).to(device)

        # # Use the embedding layer
        # decoder_emb_inputs = embedding_layer(torch.tensor(y_train[:, :-1]).to(device)).float()
        decoder_emb_inputs = torch.ones((y_train.shape[0], 1)) + 4
        decoder_emb_inputs = decoder_emb_inputs.to(device).float()
        # y_train = F.one_hot(torch.tensor(y_train, dtype=torch.long), num_classes=7).float().to(device)

        X_train = torch.tensor(X_train).unsqueeze(-1).to(device).float()
        # _y=torch.tensor(y_train[:, 1:]).to(device).float()
        _y = torch.tensor(y_train).to(device).float()

        
        if torch.cuda.is_available():
            model.to(device)
            print('net is operated by cuda')

        # 데이터를 TensorDataset으로 묶습니다.
        train_data = TensorDataset(X_train, decoder_emb_inputs, _y)

        # DataLoader로 배치 사이즈를 설정합니다.
        # batch_size를 원하는 배치 크기로 설정해주세요.
        batch_size = 512
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        mse = nn.MSELoss()
        model.train()
        for i in tqdm(range(kargs['epochs'])):
            for X_batch, decoder_emb_inputs_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred, _ = model(X_batch, decoder_emb_inputs_batch)
                pred = pred.squeeze()

                _loss = mse(y_batch, pred)
                print(_loss)
                _loss.backward(retain_graph=True)
                optimizer.step()
# 데이터 확인, 스케쥴러, loss function, optimizer, 파일저장
    return 0


if __name__ == '__main__':
    train()