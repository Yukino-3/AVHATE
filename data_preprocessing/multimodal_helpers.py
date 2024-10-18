from utils.utils import load_config
import numpy as np
import pickle
import os

config = load_config('configs/configs.yaml')


def read_audio_data():
    modality_data = {}
    if config["AUDIO_FEATURE_EXTRACTOR"] == "MFCC":
        allAudList = [name[:-7] for name in os.listdir(config["MFCC_FOLDER"]) if "hate" in name] # if condition to avoid including .DS folder automatically created
        for i in allAudList:
            with open(f"{config['MFCC_FOLDER']}/{i}_mfcc.p", 'rb') as fp:
                modality_data[i] = np.array(pickle.load(fp))
    return modality_data


def read_video_data():
    modality_data = {}
    allVidList = [name[:-6] for name in os.listdir(config["VIT_FOLDER"]) if "hate" in name] # if condition to avoid including .DS folder automatically created
    for i in allVidList:
        with open(f"{config['VIT_FOLDER']}/{i}_vit.p", 'rb') as fp:
            modality_data[i] = np.array(pickle.load(fp))
    return modality_data

def read_data_for_modality(modality):
    if modality == 'AUD':
        modality_data = read_audio_data()
    elif modality == 'VID':
        modality_data = read_video_data()
    return modality_data


