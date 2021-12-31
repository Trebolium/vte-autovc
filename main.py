import os, pdb, pickle, random, argparse, shutil, yaml
from solver_encoder import Solver
#from solver_encoder_singerid_embs import Solver
from data_loader import VctkFromMeta, PathSpecDataset, SpecChunksFromPkl
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from shutil import copyfile


def str2bool(v):
    return v.lower() in ('true')

def overwrite_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    
def new_song_idx(dataset):
    # finds the index for each new song in dataset
    new_Song_idxs = []
    song_idxs = list(range(255))
    for song_idx in song_idxs:
        for ex_idx, ex in enumerate(dataset):
            if ex[1] == song_idx:
                new_Song_idxs.append(ex_idx)
                break
    return new_Song_idxs

def main(config):
    singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']
    cudnn.benchmark = True # For fast training.
    random.seed(1)

    with open('/homes/bdoc3/my_data/spmel_data/vocalSet_subset_unnormed/uuuuuspmel_params.yaml') as File:
        spmel_params = yaml.load(File, Loader=yaml.FullLoader)

    all_idxs = [i for i in range(20)] # assumes dataset is in the order of singer_names as seen above
    dataset = PathSpecDataset(config, spmel_params)
    vte_dir_config_path = config.emb_ckpt[:-30] +'/config_params.pkl'
    vte_dir_config = pickle.load(open(vte_dir_config_path,'rb'))
    vocalset_test_ids = vte_dir_config.test_list.split(' ')
    vocalset_test_idxs = [singer_names.index(i) for i in vocalset_test_ids]

    if config.use_loader == 'vocal':
        train_song_idxs = [i for i in all_idxs if i not in vocalset_test_idxs]
        train_sampler = SubsetRandomSampler(train_song_idxs)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=True)
        d_idx_list = list(range(len(dataset)))
    elif config.use_loader == 'medley':
        dataset = SpecChunksFromPkl(config, spmel_params)
        d_idx_list = list(range(len(dataset)))
        train_song_idxs = random.sample(d_idx_list, int(len(dataset)*0.8)) 
        train_sampler = SubsetRandomSampler(train_song_idxs)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=True)
    elif config.use_loader == 'vctk':
        dataset = VctkFromMeta(config)
        d_idx_list = list(range(len(dataset)))
        train_song_idxs = random.sample(d_idx_list, int(len(dataset)*0.8)) 
        train_sampler = SubsetRandomSampler(train_song_idxs)
        train_loader = DataLoader(dataset, batch_size=config.batch_size, sampler=train_sampler, shuffle=False, drop_last=True)
    else: raise NameError('use_loader string not valid')


    if config.eval_all == True:
        medleydb = SpecChunksFromPkl(config, spmel_params)
        vocalset = PathSpecDataset(config, spmel_params)
        vctk = VctkFromMeta(config)
        datasets = [medleydb, vocalset, vctk]
        print('Finished loading the datasets...')
        test_loaders = []
        ds_ids_train_idxs = []
        d_idx_list = list(range(len(datasets)))
        ds_labels = ['medley', 'vocal', 'vctk']
        for i, ds in enumerate(datasets):
            random.seed(1) # reinstigating this at every iteration ensures the same random numbers are for each dataset
            current_ds_size = len(ds)
            d_idx_list = list(range(current_ds_size))
            if i==0:
                train_song_idxs = random.sample(d_idx_list, int(current_ds_size*0.8))
                test_song_idxs = [x for x in d_idx_list if x not in train_song_idxs]
                ds_ids_train_idxs.append((ds_labels[i], [(x[2][:-10]) for x in ds], train_song_idxs))
            elif i==1:
                train_song_idxs = [i for i in all_idxs if i not in vocalset_test_idxs]
                test_song_idxs = vocalset_test_idxs
                ds_ids_train_idxs.append((ds_labels[i], [(x[2].split('_')[0]) for x in ds], train_song_idxs))
            elif i==2:
                train_song_idxs = random.sample(d_idx_list, int(current_ds_size*0.8))
                test_song_idxs = [x for x in d_idx_list if x not in train_song_idxs]
                # save all singer_ids and the idx of only those we'll use for trainingtrain_song_idxs
                ds_ids_train_idxs.append((ds_labels[i], [(x[2]) for x in ds], train_song_idxs))
            test_sampler = SubsetRandomSampler(test_song_idxs)
            test_loader = DataLoader(ds, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, drop_last=True)
            test_loaders.append((ds_labels[i], test_loader))
        with open('dataset_ids_train_idxs.pkl','wb') as File:
            pickle.dump(ds_ids_train_idxs, File)
    else:
        test_song_idxs = [x for x in d_idx_list if x not in train_song_idxs]
        config.test_idxs = test_song_idxs
        test_sampler = SubsetRandomSampler(test_song_idxs)
        test_loaders = DataLoader(dataset, batch_size=config.batch_size, sampler=test_sampler, shuffle=False, drop_last=True)

    solver = Solver(train_loader, config, spmel_params)
    current_iter = solver.get_current_iters()
    log_list = []
    while current_iter < config.num_iters:
        current_iter, log_list = solver.iterate('train', train_loader, current_iter, config.train_iter, log_list)
        for ds_label, test_loader in test_loaders:
            current_iter, log_list = solver.iterate(f'test_{ds_label}', test_loader, current_iter, int(config.train_iter*0.2), log_list)
    solver.closeWriter()
    with open(config.data_dir +'/' +config.file_name +'/log_list.pkl', 'wb') as File:
        pickle.dump(log_list, File)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # use configurations from a previous model
    parser.add_argument('--use_ckpt_config', type=str2bool, default=False, help='path to config file to use')
    parser.add_argument('--use_loader', type=str, default='vocal', help='take singer ids to exclude from the VTEs config.test_list')
    parser.add_argument('--ckpt_model', type=str, default='', help='path to the ckpt model want to use')
    parser.add_argument('--data_dir', type=str, default='/homes/bdoc3/my_data/autovc_data/autoStc', help='path to config file to use')
    parser.add_argument('--which_embs', type=str, default='vt-live', help='path to config file to use')
    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    parser.add_argument('--one_hot', type=str2bool, default=False, help='Toggle 1-hot mode')
    parser.add_argument('--with_cd', type=str2bool, default=False, help='Toggle 1-hot mode')
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
    parser.add_argument('--patience', type=float, default=30, help='Determine weight applied to pre-net reconstruction loss')
    parser.add_argument('--eval_all', type=str2bool, default=True, help='determines whether to evaluate with one DataLoader or all DataLoaders')
 
    # Miscellaneous.
    parser.add_argument('--emb_ckpt', type=str, default='/homes/bdoc3/phonDet/results/newStandardAutovcSpmelParamsUnnormLatent64Out256/best_epoch_checkpoint.pth.tar', help='toggle checkpoint load function')
    parser.add_argument('--ckpt_freq', type=int, default=50000, help='frequency in steps to mark checkpoints')
    parser.add_argument('--spec_freq', type=int, default=10000, help='frequency in steps to print reconstruction illustrations')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--train_iter', type=int, default=500)
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

    if config.one_hot==True:
        config.dim_emb=config.train_size
    
    print(config)
    if config.file_name == config.ckpt_model:
        raise Exception("Your file name and ckpt_model name can't be the same")
    if not config.ckpt_freq%int(config.train_iter*0.2) == 0 or not config.ckpt_freq%int(config.train_iter*0.2) == 0:
        raise Exception(f"ckpt_freq {config.ckpt_freq} and spec_freq {config.spec_freq} need to be a multiple of test_iter {int(config.train_iter*0.2)}")
    overwrite_dir(config.data_dir +'/' +config.file_name)
    os.makedirs(config.data_dir +'/' +config.file_name +'/ckpts')
    os.makedirs(config.data_dir +'/' +config.file_name +'/generated_wavs')
    os.makedirs(config.data_dir +'/' +config.file_name +'/image_comparison')
    with open(config.data_dir +'/' +config.file_name +'/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)
    open(config.data_dir +'/' +config.file_name +'/config.txt', 'a').write(str(config))
    copyfile('./model_vc.py',(config.data_dir +'/' +config.file_name +'/this_model_vc.py'))
    copyfile('./solver_encoder.py',(config.data_dir +'/' +config.file_name +'/solver_encoder.py'))
    copyfile('./main.py',(config.data_dir +'/' +config.file_name +'/main.py'))
    main(config)
