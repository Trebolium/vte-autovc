import os, pdb, shutil
#import soundfile as sf

#src_root = '/import/c4dm-datasets/MUSDB18HQ/test'
src_root = '/import/research_c4dm/ss404/V2'
des_dir = '/homes/bdoc3/vte-autovc/external_examples_medleydb'

#pdb.set_trace
#for src_dir in os.scandir(src_root): 
#    for f in os.scandir(src_dir.path):
#        if f.name.startswith('voc'):
#            print(f.path, f.name)
#            shutil.copy(f.path, des_dir +'/' +f.name[:-4] +'_' +src_dir.name +'.wav')

#for song_dir in os.scandir(src_root):
#    if not song_dir.name.startswith('._'):
#        for audio_dir in os.scandir(song_dir.path):
#            if not audio_dir.name.startswith('._'):
#                if audio_dir.name.endswith('wav'):
#                    des_name = audio_dir.name[:-4] +'_' +song_dir.name +'.wav'
#                    shutil.copy(audio_dir.path, os.path.join(des_dir, des_name))


for song_dir in os.scandir(src_root):
    if not song_dir.name.startswith('._'):
        for audio_dir in os.scandir(song_dir.path):
            if not audio_dir.name.startswith('._'):
                if audio_dir.name.endswith('RAW') and not audio_dir.name.startswith('.'):
                    for audio_file in os.scandir(audio_dir.path):
                        if audio_file.name.endswith('01_01.wav'):
                            print(audio_file.name, audio_file.path)
                            des_name = audio_file.name[:-4] +'_' +song_dir.name +'.wav'
                            shutil.copy(audio_file.path, os.path.join(des_dir, des_name))
