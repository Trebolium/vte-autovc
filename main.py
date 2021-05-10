import os, pdb, pickle, random argparse, shutil, yaml
from solver_encoder import Solver
from data_loader import VctkFromMeta, PathSpecDataset, SpecChunksFromPkl
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from shutil import copyfile

def str2bool(v):
    return v.lower() in ('true')

def overwrite_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    

def main(config):
    # For fast training.
    cudnn.benchmark = True
    random.seed(1)
    with open(config.spmel_dir +'/spmel_params.yaml') as File:
        spmel_params = yaml.load(File, Loader=yaml.FullLoader)
    if config.use_loader == 'PathSpecDataset':
        dataset = PathSpecDataset(config, spmel_params)

    elif config.use_loader == 'SpecChunksFromPkl':
        dataset = SpecChunksFromPkl(config, spmel_params)
        test_song_idxs = random.sample(range(len(dataset), (len(dataset)//0.2)) 
        train_song_idxs = range(len(dataset) - test_song_idxs
        train_sampler = SubsetRandomSampler(train_song_idxs)
        test_sampler = SubsetRandomSampler(test_song_idxs)
    elif config.use_loader == 'VctkFromMeta':
        dataset = VctkFromMeta(config)
    else: raise NameError('use_loader string not valid')
    pdb.set_trace()

    train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=False)
    test_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, drop_last=False)
    #data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, drop_last=False)
    # Data loader.
    #vcc_loader = get_loader(config)
    # pass dataloader and configuration params to Solver NN
    if config.file_name == 'defaultName' or config.file_name == 'deletable':
        writer = SummaryWriter('testRuns/test')
        #writer = SummaryWriter(filename_suffix = config.file_name)
    else:
        writer = SummaryWriter(comment = '_' +config.file_name)
        #writer = SummaryWriter(filename_suffix = config.file_name)
        
    solver = Solver(data_loader, config, spmel_params)
    solver.train(writer)
    
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # use configurations from a previous model
    parser.add_argument('--use_ckpt_config', type=str2bool, default=False, help='path to config file to use')
    parser.add_argument('--exclude_test', type=str2bool, default=True, help='take singer ids to exclude from the VTEs config.test_list')
    parser.add_argument('--use_loader', type=str, default='pathSpecDataset', help='take singer ids to exclude from the VTEs config.test_list')
    parser.add_argument('--ckpt_model', type=str, default='', help='path to config file to use')
    parser.add_argument('--data_dir', type=str, default='/homes/bdoc3/my_data/autovc_data/autoStc', help='path to config file to use')
    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=32)
    parser.add_argument('--one_hot', type=str2bool, default=False, help='Toggle 1-hot mode')
    parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda to use')
    
    # Training configuration.
    parser.add_argument('--file_name', type=str, default='defaultName')
    parser.add_argument('--spmel_dir', type=str, default='/homes/bdoc3/my_data/phonDet/spmel_autovc_params_unnormalized')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--adam_init', type=float, default=0.0001, help='Define initial Adam optimizer learning rate')
    parser.add_argument('--train_size', type=int, default=20, help='Define how many speakers are used in the training set')
    parser.add_argument('--len_crop', type=int, default=192, help='dataloader output sequence length')
    parser.add_argument('--chunk_seconds', type=float, default=0.5, help='dataloader output sequence length')
    parser.add_argument('--chunk_num', type=int, default=6, help='dataloader output sequence length')
    parser.add_argument('--psnt_loss_weight', type=float, default=1.0, help='Determine weight applied to postnet reconstruction loss')
    parser.add_argument('--prnt_loss_weight', type=float, default=1.0, help='Determine weight applied to pre-net reconstruction loss')
 
    # Miscellaneous.
    parser.add_argument('--emb_ckpt', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar', help='toggle checkpoint load function')
    parser.add_argument('--ckpt_freq', type=int, default=10000, help='frequency in steps to mark checkpoints')
    parser.add_argument('--spec_freq', type=int, default=10000, help='frequency in steps to print reconstruction illustrations')
    parser.add_argument('--log_step', type=int, default=10)
    config = parser.parse_args()

    if config.ckpt_model != '':
#        ckpt_path = os.path.join(config.data_dir, config.ckpt_model, 'ckpts')
#        for file_object in os.scandir(ckpt_path):
#            if file_object.name.endswith('.pth.tar'):
#                config.autovc_ckpt = file_object.path
        if config.use_ckpt_config == True:
            num_iters = config.num_iters
            file_name = config.file_name
            autovc_ckpt = config.autovc_ckpt
            emb_ckpt = config.emb_ckpt
            ckpt_model = config.ckpt_model
            ckpt_freq = config.ckpt_freq
            config = pickle.load(open(os.path.join(config.data_dir, config.ckpt_model, 'config.pkl'), 'rb'))
            config.ckpt_model = ckpt_model
            config.num_iters = num_iters
            config.file_name = file_name
            config.autovc_ckpt = autovc_ckpt
            config.emb_ckpt = emb_ckpt
            config.ckpt_freq = ckpt_freq

    if config.exclude_test == True: config.exclude_list = pickle.load(open(os.path.dirname(config.emb_ckpt) +'/config_params.pkl', 'rb')).test_list.split(' ')
    else: config.exclude_list = []

    if config.one_hot==True:
        config.dim_emb=config.train_size
    
    print(config)
    if config.file_name == config.ckpt_model:
        raise Exception("Your file name and ckpt_model name can't be the same")
    overwrite_dir(config.data_dir +'/' +config.file_name)
    os.makedirs(config.data_dir +'/' +config.file_name +'/ckpts')
    os.makedirs(config.data_dir +'/' +config.file_name +'/generated_wavs')
    os.makedirs(config.data_dir +'/' +config.file_name +'/image_comparison')
    with open(config.data_dir +'/' +config.file_name +'/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)
    open(config.data_dir +'/' +config.file_name +'/config.txt', 'a').write(str(config))
    copyfile('./model_vc.py',(config.data_dir +'/' +config.file_name +'/this_model_vc.py'))
    copyfile('./solver_encoder.py',(config.data_dir +'/' +config.file_name +'/solver_encoder.py'))
    main(config)
