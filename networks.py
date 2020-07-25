import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MIMO_encoder(nn.Module):
    def __init__(self,num_bits=8):
        super().__init__()
        self.num_input=int(num_bits+num_bits*num_bits*2)
        self.num_output=2*num_bits #8 for Real channel, 8 for Imaginary channel
        self.mlp=nn.Sequential(
            nn.Linear(self.num_input,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,num_bits*2)
        )

    def forward(self,input):
        x=self.mlp(input)
        x_norm=torch.norm(x,p=2,dim=1).unsqueeze(1).repeat(1,x.shape[1])
        x_normalize=x/x_norm
        return x_normalize

class MIMO_decoder(nn.Module):
    def __init__(self,num_bits=8):
        super().__init__()
        in_channels=num_bits*2
        self.mlp=nn.Sequential(
            nn.Linear(in_channels,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_bits),
            nn.Sigmoid()
            #nn.Softmax(dim=1)
        )
    def forward(self,receive_signal):
        decode_result=self.mlp(receive_signal)
        return decode_result

class neural_MIMO(nn.Module):
    def __init__(self,num_bits=8):
        super().__init__()
        self.num_input=num_bits
        self.encoder=MIMO_encoder(num_bits=num_bits)
        self.decoder=MIMO_decoder(num_bits=num_bits)
        self.criterion= nn.BCELoss()

    def compute_cross_entropy_loss(self,decode_signal,target):
        loss=self.criterion(decode_signal,target)
        return loss

    def forward(self,binary_inputs,CSI_Re,CSI_Im):
        batch_size=binary_inputs.shape[0]
        CSI_vec=torch.cat([CSI_Re.view(batch_size,-1),CSI_Im.view(batch_size,-1)],dim=1)
        encoder_input=torch.cat([binary_inputs,CSI_vec],dim=1)
        encode_output=self.encoder(encoder_input)
        encode_Re=encode_output[:,0:self.num_input].unsqueeze(2)
        encode_Im=encode_output[:,self.num_input:self.num_input*2].unsqueeze(2)
        #precoding_matrix=encode_output[:,self.num_input*2:].view(-1,self.num_input,self.num_input,2)
        #precode_matrix_Re=precoding_matrix[:,:,:,0]
        #precode_matrix_Im=precoding_matrix[:,:,:,1]
        #precode_Re=torch.bmm(precode_matrix_Re,encode_Re)-torch.bmm(precode_matrix_Im,encode_Im)
        #precode_Im=torch.bmm(precode_matrix_Re,encode_Im)+torch.bmm(precode_matrix_Im,encode_Re)

        #print(CSI_Re.shape)
        #print(CSI_Re.shape,encode_Re.shape)
        Re_Raleigh=torch.matmul(CSI_Re,encode_Re)-torch.matmul(CSI_Im,encode_Im)
        Im_Raleigh=torch.matmul(CSI_Im,encode_Re)+torch.matmul(CSI_Re,encode_Im)
        Raleigh_signal=torch.cat([Re_Raleigh.squeeze(2),Im_Raleigh.squeeze(2)],dim=1)
        #Raleigh_signal=torch.cat([encode_Re.squeeze(2),encode_Im.squeeze(2)],dim=1)

        EBN0=torch.randint(0,20,(batch_size,)).unsqueeze(1).repeat(1,Raleigh_signal.shape[1]).float()
        EBN0=torch.pow(10,EBN0)
        #print(EBN0)
        thermal_noise=math.sqrt(1 / 2)*torch.randn(Raleigh_signal.shape)
        if torch.cuda.is_available():
            EBN0=EBN0.cuda()
            thermal_noise=thermal_noise.cuda()
        #noise_power = torch.mean(thermal_noise ** 2, dim=1)
        #print(noise_power)
        random_nunmber=torch.rand(1)
        if random_nunmber>0.1:
            thermal_noise=thermal_noise/torch.sqrt(EBN0)
        else:
            thermal_noise=0 #feed 0 noise to the channel to train
        #noise_power=torch.mean(thermal_noise**2,dim=1)
        #print(noise_power)
        Receive_signal=Raleigh_signal+thermal_noise
        decode_signal=self.decoder(Receive_signal)
        return decode_signal

class neural_MIMO_fixSNR(nn.Module):
    def __init__(self,num_bits=8):
        super().__init__()
        self.num_input=num_bits
        self.encoder=MIMO_encoder(num_bits=num_bits)
        self.decoder=MIMO_decoder(num_bits=num_bits)
        self.criterion= nn.BCELoss()

    def forward(self,binary_inputs,CSI_Re,CSI_Im,EBN0):
        batch_size=binary_inputs.shape[0]
        CSI_vec=torch.cat([CSI_Re.view(batch_size,-1),CSI_Im.view(batch_size,-1)],dim=1)
        encoder_input=torch.cat([binary_inputs,CSI_vec],dim=1)
        encode_output=self.encoder(encoder_input)
        encode_Re=encode_output[:,0:self.num_input].unsqueeze(2)
        encode_Im=encode_output[:,self.num_input:self.num_input*2].unsqueeze(2)

        Re_Raleigh=torch.matmul(CSI_Re,encode_Re)-torch.matmul(CSI_Im,encode_Im)
        Im_Raleigh=torch.matmul(CSI_Im,encode_Re)+torch.matmul(CSI_Re,encode_Im)
        Raleigh_signal=torch.cat([Re_Raleigh.squeeze(2),Im_Raleigh.squeeze(2)],dim=1)

        EBN0=torch.tensor(EBN0).float().expand_as(Raleigh_signal)
        if torch.cuda.is_available():
            EBN0=EBN0.cuda()
        thermal_noise=math.sqrt(1 / 2)*torch.randn(Raleigh_signal.shape)
        if torch.cuda.is_available():
            EBN0=EBN0.cuda()
            thermal_noise=thermal_noise.cuda()
        thermal_noise=thermal_noise/torch.sqrt(EBN0)
        noise_power=torch.mean(thermal_noise**2)
        Receive_signal=Raleigh_signal+thermal_noise
        decode_signal=self.decoder(Receive_signal)
        return decode_signal

if __name__=="__main__":
    binary_input=torch.rand((10,8))
    binary_input=binary_input>0.5
    binary_input=binary_input.float().cuda()
    print(binary_input)

    model=neural_MIMO(num_bits=8).cuda()
    decode_signal=model(binary_input)
    #print(decode_signal.shape)
    print(decode_signal)
    loss=model.compute_cross_entropy_loss(decode_signal,binary_input)
    print("cross entropy loss is",loss)