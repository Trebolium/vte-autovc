from model_vc import Generator
from vte_model import Vt_Embedder
from collections import OrderedDict
import torch
import math, os, pickle
import utils
from scipy.signal import medfilt
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time, pdb
import datetime

# SOLVER IS THE MAIN SETUP FOR THE NN ARCHITECTURE. INSIDE SOLVER IS THE GENERATOR (G)
class Solver(object):

    def __init__(self, data_loader, config, spmel_params):
        """Initialize configurations."""
    
        self.config = config
        self.spmel_params = spmel_params
        # Data loader.
        self.data_loader = data_loader

        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_neck = config.dim_neck
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        self.which_cuda = config.which_cuda

        # Training configurations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.emb_ckpt = config.emb_ckpt
        self.file_name = config.file_name
        self.one_hot = config.one_hot
        self.psnt_loss_weight = config.psnt_loss_weight 
        self.prnt_loss_weight = config.prnt_loss_weight 
        self.adam_init = config.adam_init
        self.style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
        self.singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.which_cuda}' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        self.ckpt_freq = config.ckpt_freq
        self.spec_freq = config.spec_freq

        self.singer_id_embs = torch.FloatTensor([embs[1] for embs in pickle.load(open('singer_id_embs.pkl','rb'))])
        # Build the model and tensorboard.
        self.build_model()

        self.avg_embs = np.load(os.path.dirname(config.emb_ckpt) +'/averaged_embs.npy')

    def build_model(self):
       
        self.vte =  Vt_Embedder(self.config, self.spmel_params)
        for param in self.vte.parameters():
            param.requires_grad = False
        self.vte_optimizer = torch.optim.Adam(self.vte.parameters(), 0.0001)
        self.vte_checkpoint = torch.load(self.emb_ckpt)
        new_state_dict = OrderedDict()
        for i, (key, val) in enumerate(self.vte_checkpoint['model_state_dict'].items()):
#            if key.startswith('class_layer'):
#                continue
            new_state_dict[key] = val 
        self.vte.load_state_dict(new_state_dict)

        for state in self.vte_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(self.device)

        self.vte.to(self.device)
        self.vte.eval()

        self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.adam_init)
        if self.config.ckpt_model!='':
#            _, _, ckpt_list = next(os.walk(self.config.ckpt_model +'/ckpts'))
#            ckpt_name = ''
#            for i in ckpt_list:
#                if i.endswith('.tar'): ckpt_name = i
#            ckpt_path = os.path.join(self.config.ckpt_model, 'ckpts', ckpt_name)
            ckpt_path = os.path.join('/homes/bdoc3/my_data/autovc_data/autoStc', self.config.ckpt_model)
            g_checkpoint = torch.load(ckpt_path)
            self.G.load_state_dict(g_checkpoint['model_state_dict'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
            # fixes tensors on different devices error
            # https://github.com/pytorch/pytorch/issues/2830
            for state in self.g_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.previous_ckpt_iters = g_checkpoint['iteration']
            tester=2
        else:
            self.previous_ckpt_iters = 0
        self.G.to(self.device)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    
    #=====================================================================================================================================#
   
        

    def train(self, writer):
        # Set data loader.
        data_loader = self.data_loader
        hist_arr = np.array([0,0,0])
        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
        last_save = file_path = hist_file_path =  'delete.txt' 
        losses_list = [0., 0., 0.]
        # Start training.
        print('Start training...')
        start_time = time.time()
        log_list = []

        for i in range(self.previous_ckpt_iters, self.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch data.
            # THE NEXT(DATA_ITER) FUNCTION USES THE DATALOADERS __GETITEM__ FUNCTION, SEEMINGLY
            # ITER AND NEXT FUNCTIONS WORK TOGETHER TO PRODUCE A COLLATED BATCH OF EXAMPLES 
            try:
                x_real, style_idx, singer_idx = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, style_idx, singer_idx = next(data_iter)
        
            x_real = x_real.to(self.device).float() 
            x_real_chunked = x_real.view(x_real.shape[0]*self.config.chunk_num, x_real.shape[1]//self.config.chunk_num, -1)
            # =================================================================================== #
            #                               2. Train the generator                                #
            # =================================================================================== #

            # informs generator to be in train mode 

            pred_style_idx, all_tensors = self.vte(x_real_chunked)
            emb_org = all_tensors[-1]
            #emb_org = torch.tensor(self.avg_embs[style_idx]).to(self.device)
            #pdb.set_trace()
            self.G = self.G.train()
            # x_identic_psnt consists of the original mel + the residual definiton added ontop
            x_identic, x_identic_psnt, code_real, _, _ = self.G(x_real, emb_org, emb_org)
            # SHAPES OF X_REAL AND X_INDETIC/PSNT ARE NOT THE SAME AND MAY GIVE INCORRECT LOSS VALUES
            residual_from_psnt = x_identic_psnt - x_identic
            # pdb.set_trace()
            x_identic = x_identic.squeeze(1)
            x_identic_psnt = x_identic_psnt.squeeze(1)
            residual_from_psnt = residual_from_psnt.squeeze(1)

            g_loss_id = F.l1_loss(x_real, x_identic)   
            g_loss_id_psnt = F.l1_loss(x_real, x_identic_psnt)   
            
            # Code semantic loss. For calculating this, there is no target embedding
            code_reconst = self.G(x_identic_psnt, emb_org, None)
            # gets the l1 loss between original encoder output and reconstructed encoder output
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            # Backward and optimize.
            # interesting - the loss is a sum of the decoder loss and the melspec loss
            g_loss = (self.prnt_loss_weight * g_loss_id) + (self.psnt_loss_weight * g_loss_id_psnt) #+ ((self.lambda_cd  * (i / 100000)) * g_loss_cd)
            self.reset_grad()
            g_loss.backward()
            #pdb.set_trace()
            self.g_optimizer.step()

            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
            losses_list[0] += g_loss_id.item()
            losses_list[1] += g_loss_id_psnt.item()
            losses_list[2] += g_loss_cd.item()

            tb_freq = ((self.num_iters-self.previous_ckpt_iters)//1500)
            if (i+1) % tb_freq == 0: 
                writer.add_scalar(f"Loss_id/Train", losses_list[0]/tb_freq, i)
                writer.add_scalar(f"Loss_id_psnt/Train", losses_list[1]/tb_freq, i)
                #writer.add_scalar(f"Loss_cd/Train", losses_list[2]/tb_freq, i)
                losses_list = [0.,0.,0.]
                writer.flush()
                print('writer flushed')

            if i==0:
                hist_arr = np.array([g_loss_id.item(), g_loss_id_psnt.item(), g_loss_cd.item()])
            else:
                temp_arr = np.array([g_loss_id.item(), g_loss_id_psnt.item(), g_loss_cd.item()])
                hist_arr = np.vstack((hist_arr, temp_arr))
            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #
            #pdb.set_trace()

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                log_list.append(log)

            if (i+1) % self.spec_freq == 0:
                # save x and x_hat images
                x_real = x_real.cpu().data.numpy()
                x_identic = x_identic.cpu().data.numpy()
                x_identic_psnt = x_identic_psnt.cpu().data.numpy()
                residual_from_psnt = residual_from_psnt.cpu().data.numpy()
                specs_list = []
                for arr in x_real:
                    specs_list.append(arr)
                for arr in x_identic:
                    specs_list.append(arr)
                for arr in residual_from_psnt:
                    specs_list.append(arr)
                for arr in x_identic_psnt:
                    specs_list.append(arr)
                columns = 2
                rows = 4
                fig, axs = plt.subplots(4,2)
                fig.tight_layout()
                for j in range(0, columns*rows):
                    spec = np.rot90(specs_list[j])
                    fig.add_subplot(rows, columns, j+1)
                    if j == 5 or j == 6:
                        spec = spec - np.min(spec)
                        plt.clim(0,1)
                    plt.imshow(spec)
                    try:
                        name = 'Singer ' +str(singer_idx[j%2]) +', Style ' +str(style_idx[j%2])
                    except:
                        pdb.set_trace()
                    plt.title(name)
                    plt.colorbar()
                plt.savefig(self.config.data_dir +'/' +self.file_name +'/image_comparison/' +str(i+1) +'iterations')
                plt.close()
                
            if (i+1) % self.ckpt_freq == 0:
                print('Saving model...')
                checkpoint = {'model_state_dict' : self.G.state_dict(),
                    'optimizer_state_dict': self.g_optimizer.state_dict(),
                    'iteration': i+1,
                    'loss': loss}
#                if os.path.exists(last_save):
#                    os.remove(last_save)
                torch.save(checkpoint, self.config.data_dir +'/' +self.file_name +'/ckpts/' +'ckpt_' +str(i+1) +'.pth.tar')
                last_save = self.config.data_dir +'/' +self.file_name +'/ckpts/' +'ckpt_' +str(i+1) +'.pth.tar'
                # plotting history since last checkpoint downsampled by 100
                print('Saving loss visuals...')
                num_cols=1
                num_graph_vals = 200
                down_samp_size = math.ceil(self.ckpt_freq/num_graph_vals)
                modified_array = hist_arr[-self.ckpt_freq::down_samp_size,:]
#                if os.path.exists(file_path):
#                    os.remove(file_path)
                file_path = self.config.data_dir +'/' +self.file_name +'/ckpts/' +'ckpt_' +str(i+1) +'_loss_since_ckpt.png'
                labels = ['iter_steps','loss','loss_id','loss_id_psnt','loss_cd']
                utils.saveContourPlots(modified_array, file_path, labels, num_cols) 
                #if (i+1) % (self.ckpt_freq*2) == 0:
                print('saving loss visuals of all history...')
                down_samp_size = math.ceil(i/num_graph_vals)
                modified_array = hist_arr[::down_samp_size,:]
                if os.path.exists(hist_file_path):
                    os.remove(hist_file_path)
                hist_file_path = self.config.data_dir +'/' +self.file_name +'/ckpts/' +'ckpt_' +str(i+1) +'_loss_since_training_session.png'
                utils.saveContourPlots(modified_array, hist_file_path, labels, num_cols) 
                with open(self.config.data_dir +'/' +self.config.file_name +'/log_list.pkl', 'wb') as File:
                    pickle.dump(log_list, File)
        writer.close()
