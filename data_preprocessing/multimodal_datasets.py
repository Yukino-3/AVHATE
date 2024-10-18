import torch
from torch.utils.data import Dataset
from data_preprocessing.multimodal_helpers import read_data_for_modality
from utils.utils import load_config

config = load_config('configs/configs.yaml')

class MultimodalDataset(Dataset):
    def __init__(self, folders, labels, modalities):
        self.folders = folders
        self.labels = labels 
        self.load_modalities_data(modalities)       

    def load_modalities_data(self, modalities):
        self.modalities_data = {}
        for modality in modalities:
            self.modalities_data[modality] = read_data_for_modality(modality)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        sample = {}

        for modality in self.modalities_data.keys():
            sample[modality + '_features'] = self.get_features_for_modality(folder, modality)
        y = torch.LongTensor([self.labels[idx]])

        return sample, y

    def get_features_for_modality(self, folder, modality):
        return torch.tensor(self.modalities_data[modality][folder])   