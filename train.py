import os,sys
import argparse
sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', '-e', type=str, required=True, help='experiment name')
parser.add_argument('--debug', action='store_true', help='specify debug mode')
parser.add_argument('--batch_size', type=int,default=128)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--nepoch',type=int,default=100)
parser.add_argument('--num_frames',type=int,default=100)
parser.add_argument('--num_bits',type=int,default=8)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import datetime
from torch.optim import Adam
from tensorboardX import SummaryWriter
from networks import neural_MIMO
import time
from dataset import MIMO_dataset
from torch.utils import data
import numpy as np
import math

def worker_init_fn_seed(worker_id):
    seed = datetime.datetime.now().second
    seed += worker_id
    np.random.seed(seed)
    #print(worker_id)

def train(args):
    start_t=time.time()

    log_dir=os.path.join("./checkpoints",args.exp_name)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)

    tb_logger=SummaryWriter(log_dir)

    dataset = MIMO_dataset(num_bits=args.num_bits)
    dataloader=data.DataLoader(dataset=dataset,
                               batch_size=args.batch_size,
                               num_workers=16,
                               worker_init_fn= worker_init_fn_seed,
                               drop_last=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=neural_MIMO(num_bits=args.num_bits).to(device)
    model.train()
    optimizer=Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999))
    iter=0
    #print(args.nepoch)

    CSI_Re = math.sqrt(1 / 2) * torch.randn((args.num_frames, args.num_bits, args.num_bits))
    CSI_Im = math.sqrt(1 / 2) * torch.randn((args.num_frames, args.num_bits, args.num_bits))
    np.save(os.path.join(log_dir,'CSI_Re.npy'),CSI_Re.numpy())
    np.save(os.path.join(log_dir, 'CSI_Im.npy'), CSI_Im.numpy())

    #current_CSI_Im = math.sqrt(1 / 2) * torch.randn((8, 8)).unsqueeze(0).repeat(args.batch_size, 1, 1)
    #current_CSI_Re = math.sqrt(1 / 2) * torch.randn((8, 8)).unsqueeze(0).repeat(args.batch_size, 1, 1)
    for e in range(args.nepoch):
        for batch_id,(binary_input,target) in enumerate(dataloader):
            random_frame_index=np.random.randint(0,args.num_frames)

            '''
            current_CSI_Re=CSI_Re[random_frame_index].unsqueeze(0).repeat(args.batch_size,1,1)
            current_CSI_Im = CSI_Im[random_frame_index].unsqueeze(0).repeat(args.batch_size, 1, 1)
            '''
            '''
            current_CSI_Im = math.sqrt(1 / 2) * torch.randn((args.num_bits, args.num_bits)).unsqueeze(0).repeat(args.batch_size, 1, 1)
            current_CSI_Re = math.sqrt(1 / 2) * torch.randn((args.num_bits, args.num_bits)).unsqueeze(0).repeat(args.batch_size, 1, 1)
            '''
            current_CSI_Im = math.sqrt(1 / 2) * torch.randn((args.batch_size,args.num_bits, args.num_bits))
            current_CSI_Re = math.sqrt(1 / 2) * torch.randn((args.batch_size,args.num_bits, args.num_bits))
            optimizer.zero_grad()
            binary_input=binary_input.float().cuda()
            target=target.float().cuda()
            #print(binary_input)
            batch_size=binary_input.shape[0]
            if torch.cuda.is_available():
                current_CSI_Im=current_CSI_Im.cuda()
                current_CSI_Re=current_CSI_Re.cuda()
            decode_output=model(binary_input,current_CSI_Re,current_CSI_Im)
            #print(decode_output[0])
            loss=model.compute_cross_entropy_loss(decode_output,binary_input)
            loss.backward()
            optimizer.step()

            detect_output=decode_output>0.5
            detect_output=detect_output.float()
            BER=torch.sum(torch.abs(detect_output-binary_input))/batch_size/args.num_bits

            msg = "{:0>8},{}:{},[{}/{}],{}: {}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(dataloader),
                "total_loss",
                loss.item()
            )
            print(msg)

            tb_logger.add_scalar('loss',loss.item(),iter)
            tb_logger.add_scalar('BER',BER.item(),iter)
            iter += 1
        if (e + 1) % 10 == 0:
            model_save_dir = os.path.join(log_dir, 'epoch_%d.pth' % (e))
            torch.save(model.state_dict(), model_save_dir)

if __name__=="__main__":
    train(args)



