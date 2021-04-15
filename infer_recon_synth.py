import time, os, pdb, pickle, argparse, shutil, yaml, torch, math, time, pdb, datetime, pickle
import utils #file
from solver_encoder import Solver 
from data_loader import pathSpecDataset
from torch.utils.data import DataLoader
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
import torch.nn.functional as F
import importlib

# tailor config, define other 
model_name = 'vteautovcWithAvgVtesNoCdLoss'
cudnn.benchmark = True
use_avg_vte = True
convert_style = True
autovc_model_saves_dir = '/homes/bdoc3/my_data/autovc_data/vte-autovc/model_saves/'
autovc_model_dir = autovc_model_saves_dir + model_name
config = pickle.load(open(autovc_model_dir +'/config.pkl','rb'))
ckpt_iters = 500000
config.which_cuda = 0
config.batch_size = 1
config.autovc_ckpt = autovc_model_dir +'/ckpts/ckpt_' +str(ckpt_iters) +'.pth.tar'
avg_embs = np.load(os.path.dirname(config.emb_ckpt) +'/averaged_embs.npy')
config.vte_ckpt = '/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar'
# additional config attrs
style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
male_idx = range(0,11)
female_idx = range(11,20)
config.device = torch.device(f'cuda:{config.which_cuda}' if torch.cuda.is_available() else 'cpu')
with open(config.spmel_dir +'/spmel_params.yaml') as File:
    spmel_params = yaml.load(File, Loader=yaml.FullLoader)
    config

subdir_for_wavs = autovc_model_dir +'/generated_wavs/' +str(ckpt_iters) +'iters'
if os.path.exists(subdir_for_wavs)==False:
    os.makedirs(subdir_for_wavs)

# import path to use autovc_model_dir's .py
import sys
sys.path.insert(1, autovc_model_dir) # usually the cwd is priority, so index 1 is good enough for our purposes here
print(sys.path)
from this_model_vc import Generator

config.batch_size=1
# setup dataloader, models
vocalSet = pathSpecDataset(config, spmel_params)
vocalSet_loader = DataLoader(vocalSet, batch_size=config.batch_size, shuffle=True, drop_last=False)
G = utils.setup_gen(config, Generator)
vte = utils.setup_vte(config, spmel_params)

import sys
sys.path.insert(1, '/homes/bdoc3/my_data/autovc_data') # usually the cwd is priority, so index 1 is good enough for our purposes here
print(sys.path)
from hparams import hparams
# importlib.reload(synthesis)

import torch
import librosa
import soundfile as sf
import pickle
from synthesis import build_model
from synthesis import wavegen

model = build_model().to(config.device)
checkpoint = torch.load("/homes/bdoc3/my_data/autovc_data/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])

find_male = True
num_examples = 8
counter = 0
while counter < num_examples:
    data_iter = iter(vocalSet_loader)
    x_real, org_style_idx, singer_idx = next(data_iter)
    x_real = x_real.to(config.device)
    original_spmel = x_real[0].cpu().detach().numpy()
    if find_male == True:
        gender_idx = male_idx
    else:
        gender_idx = female_idx
    if singer_idx in gender_idx:
        find_male = not find_male
        counter += 1
        if use_avg_vte == True:
            emb_org = torch.tensor(avg_embs[org_style_idx]).to(config.device).unsqueeze(0)
        else:
            x_real_chunked = x_real.view(x_real.shape[0]*config.chunk_num, x_real.shape[1]//config.chunk_num, -1)
            pred_style_idx, all_tensors = vte(x_real_chunked)
            emb_org = all_tensors[-1]

        saved_enc_outs = None
        saved_dec_outs = None
        all_spmels = [original_spmel]

        if convert_style == False:
            _, x_identic_psnt, _, saved_enc_outs, saved_dec_outs = G(x_real, emb_org, emb_org)
            all_spmels.append(x_identic_psnt.squeeze(1)[0].cpu().detach().numpy())
        else:
            for trg_style_idx in range(len(avg_embs)):
                emb_trg = torch.tensor(avg_embs[trg_style_idx]).to(config.device).unsqueeze(0)
                _, x_identic_psnt, _, _, _ = G(x_real, emb_org, emb_trg)
                all_spmels.append(x_identic_psnt.squeeze(1)[0].cpu().detach().numpy())

        plt.figure(figsize=(20,5))
        for j in range(len(all_spmels)):
            plt.subplot(1,len(all_spmels),j+1)
            if j == 0: plt.title('original_' +str(singer_idx))    
            else: plt.title(str(style_names[j-1] +'_' +str(singer_idx)))
            plt.imshow(np.rot90(all_spmels[j]))
        plt.savefig(subdir_for_wavs +'/example' +str(counter) +'_spmels')

        for k, spmel  in enumerate(all_spmels):
            # x_identic_psnt = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
            waveform = wavegen(model, c=spmel)   
            #     librosa.output.write_wav(name+'.wav', waveform, sr=16000)
            if k == 0:
                sf.write(subdir_for_wavs +'/example' +str(counter) +'_original.wav', waveform, samplerate=16000)
            else:
                sf.write(subdir_for_wavs +'/example' +str(counter) + '_' +style_names[k-1] +'.wav', waveform, samplerate=16000)
