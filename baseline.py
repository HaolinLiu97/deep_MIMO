import os,sys
import argparse
sys.path.append('../')

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, required=True, help='weight file path')
parser.add_argument('--batch_size', type=int,default=16)
parser.add_argument('--gpu', type=str, default='1')
parser.add_argument('--nepoch',type=int,default=100)
parser.add_argument('--num_frames',type=int,default=100)
parser.add_argument('--num_bits',type=int,default=2)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import datetime
from torch.optim import Adam
from tensorboardX import SummaryWriter
from networks import neural_MIMO_fixSNR
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

def test(args):
    dataset = MIMO_dataset(num_bits=args.num_bits,isTrain=False)
    dataloader=data.DataLoader(dataset=dataset,
                               batch_size=args.batch_size,
                               num_workers=16,
                               worker_init_fn= worker_init_fn_seed,
                               drop_last=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model=neural_MIMO_fixSNR(num_bits=args.num_bits).to(device)
    model.load_state_dict(torch.load(args.resume))
    model.eval()

    BER_list=[]
    for EBN0 in range(0,21):
        iter = 0
        error_count=0
        for batch_id,(binary_input,target) in enumerate(dataloader):
            current_CSI_Im = math.sqrt(1 / 2) * torch.randn((args.batch_size,args.num_bits, args.num_bits))
            current_CSI_Re = math.sqrt(1 / 2) * torch.randn((args.batch_size,args.num_bits, args.num_bits))
            binary_input=binary_input.float().cuda()
            batch_size=binary_input.shape[0]
            if torch.cuda.is_available():
                current_CSI_Im=current_CSI_Im.cuda()
                current_CSI_Re=current_CSI_Re.cuda()
            decode_output=model(binary_input,current_CSI_Re,current_CSI_Im,EBN0)
            #loss=model.compute_cross_entropy_loss(decode_output,binary_input)

            detect_output=decode_output>0.5
            detect_output=detect_output.float()
            error_count+=torch.sum(torch.abs(detect_output-binary_input))
            iter += 1
        BER=error_count/args.batch_size/args.num_bits/iter
        BER_list.append(BER.cpu().numpy())
        print("BER under EBN0 %d is %f"%(EBN0,BER))
    BER_array=np.array(BER_list)
    exp_name = args.resume.split("/")[-2]
    save_dir=os.path.join("./","test_results",exp_name)
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir,"BER.npy"),BER_array)

if __name__=="__main__":
    test(args)



