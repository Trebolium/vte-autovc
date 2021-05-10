import os, pdb, random, shutil, pickle
#import soundfile as sf

#src_root = '/import/c4dm-datasets/MUSDB18HQ/test'
src_root = '/import/research_c4dm/ss404/V2'
des_dir = '/homes/bdoc3/vte-autovc/external_examples_medleydb'
tracklist = pickle.load(open('tracklist.pkl','rb'))

num_choices = 20

if not os.path.exists(des_dir):
    os.makedirs(des_dir)
else:
    shutil.rmtree(des_dir)
    os.mkdir(des_dir)

chosens = []
for choice in range(num_choices):
    chosen_path = random.choice(tracklist)
    shutil.copy(chosen_path, des_dir + '/' +os.path.basename(chosen_path))
#for song_dir in os.scandir(src_root):
#    if not song_dir.name.startswith('._'):
#        for audio_dir in os.scandir(song_dir.path):
#            if not audio_dir.name.startswith('._'):
#                if audio_dir.name.endswith('RAW'):
#                    for audio_file in os.scandir(audio_dir.path):
#                        if audio_file.name.endswith('01_01.wav'):
#                            if not audio_file.name.startswith('._'):
#                                print(audio_file.name, audio_file.path)
#                                des_name = audio_file.name[:-4] +'_' +song_dir.name +'.wav'
#                                shutil.copy(audio_file.path, os.path.join(des_dir, des_name))
