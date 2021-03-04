import os, pdb, pickle, argparse, shutil, yaml
from solver_encoder import Solver
from data_loader import get_loader, pathSpecDataset
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def str2bool(v):
    return v.lower() in ('true')

def overwrite_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    

def main(config):
    # For fast training.
    cudnn.benchmark = True

    with open(config.spmel_dir +'/spmel_params.yaml') as File:
        spmel_params = yaml.load(File, Loader=yaml.FullLoader)
    vocalSet = pathSpecDataset(config, spmel_params)
    vocalSet_loader = DataLoader(vocalSet, batch_size=config.batch_size, shuffle=True, drop_last=False)
    # Data loader.
    #vcc_loader = get_loader(config)
    # pass dataloader and configuration params to Solver NN
    if config.file_name == 'defaultName' or config.file_name == 'deletable':
        writer = SummaryWriter('testRuns/test')
    else:
        writer = SummaryWriter(comment = '_' +config.file_name)

    solver = Solver(vocalSet_loader, config, spmel_params)
    solver.train(writer)
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # use configurations from a previous model
    parser.add_argument('--config_file', type=str, default='', help='path to config file to use')
    parser.add_argument('--data_dir', type=str, default='/homes/bdoc3/my_data/autovc_data/vte-autovc', help='path to config file to use')
    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    parser.add_argument('--one_hot', type=str2bool, default=False, help='Toggle 1-hot mode')
    parser.add_argument('--shape_adapt', type=str2bool, default=True, help='adjust shapes of tensors to match automatically')
    parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda to use')
    
    # Training configuration.
    parser.add_argument('--file_name', type=str, default='defaultName')
    parser.add_argument('--spmel_dir', type=str, default='/homes/bdoc3/my_data/phonDet/spmel_autovc_params_unnormalized')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--adam_init', type=float, default=0.0001, help='Define initial Adam optimizer learning rate')
    parser.add_argument('--train_size', type=int, default=21, help='Define how many speakers are used in the training set')
    parser.add_argument('--len_crop', type=int, default=192, help='dataloader output sequence length')
    parser.add_argument('--chunk_seconds', type=float, default=0.5, help='dataloader output sequence length')
    parser.add_argument('--chunk_num', type=int, default=6, help='dataloader output sequence length')
    parser.add_argument('--psnt_loss_weight', type=float, default=1.0, help='Determine weight applied to postnet reconstruction loss')
    parser.add_argument('--prnt_loss_weight', type=float, default=1.0, help='Determine weight applied to pre-net reconstruction loss')
 
    # Miscellaneous.
    parser.add_argument('--load_ckpts', type=str, default='', help='toggle checkpoint load function')
    parser.add_argument('--emb_ckpt', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar', help='toggle checkpoint load function')
    parser.add_argument('--ckpt_freq', type=int, default=10000, help='frequency in steps to mark checkpoints')
    parser.add_argument('--spec_freq', type=int, default=10000, help='frequency in steps to print reconstruction illustrations')
    parser.add_argument('--log_step', type=int, default=10)
    config = parser.parse_args()
    if config.config_file != '':
        config = pickle.load(open(config.config_file, 'rb'))
    
    if config.one_hot==True:
        config.dim_emb=config.train_size
    
    print(config)
    # pdb.set_trace()
    overwrite_dir(config.data_dir +'/model_saves/' +config.file_name)
    os.makedirs(config.data_dir +'/model_saves/' +config.file_name +'/ckpts')
    os.makedirs(config.data_dir +'/model_saves/' +config.file_name +'/generated_wavs')
    os.makedirs(config.data_dir +'/model_saves/' +config.file_name +'/image_comparison')
    with open(config.data_dir +'/model_saves/' +config.file_name +'/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)

    main(config)
