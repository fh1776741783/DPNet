import glob, os, sys, pdb, time

import numpy
import pandas as pd
import numpy as np
import cv2
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

import config

# symptom_class = ['dr','normal','hm','disc swelling and haemorrhage','disc swelling','um',
#                  'pm','maculopathy','cataract','vkh','optic atrophy','osteoma','rao',
#                  'mocd','amd','serous-exudative rd'] # 窄角

symptom_class = ['isolated drusen','normal','laser spots','ah','floaters','vitreous opacity','aneurysms',
                 'vasculitis','maculopathy','brvo','cataract','choroidal diseases','fundus neoplasm',
                 'coats','optic abnormalities','crvo','pdr','erm','fevr','hm','trd','fibrosis',
                 'lens dislocation','mh','myelinated nerve fiber','pm',
                 'peripheral retinal degeneration','rd','retinal breaks','retinal white dots','rp',
                 'silicone oil','surgery-air','surgery-band:buckle','surgery-medicine','vkh',
                 'isolated vessel tortuosity','chorioretinitis']#38疾病修改后

# pathology_class = ['microaneurysm','laser spots','exudates','retinal haemorrhage','tf','vessel tortuosity',
#                    'disc swelling and haemorrhage','disc swelling','gray-brown retinal mass',
#                    'ppa','drusen','cataract','serous-exudative rd','retinal folds','macular edema','optic atrophy',
#                    'retinal opacities','cotton-wool spots','macular haemorrhage',
#                    'hyperpigmentary']  #窄角

pathology_class = ['white fundus mass','vessel attenuation','isolated chorioretinal atrophy','vascular occlusion',
                   'peripheral retinal degeneration','floaters','macular edema','isolated vessel tortuosity','cataract','erm','preretinal fibrosis',
                   'exudates','vitreous haemorrhage','laser spots','retinal white dots',
                   'congenital disc abnormality','retinal haemorrhage','bone-spicule-pigmentation','retinal opacities','surgery-band:buckle',
                   'hm','geographic macular atrophy','retinal breaks','gray-brown retinal mass','surgery-cryotherapy',
                   'nv','subretinal fibrosis','cotton-wool spots','macular haemorrhage','disc swelling','hemangioma','fundus mass',
                   'macular atrophy','vitreous opacity','isolated drusen','hyaloid remnant','isolated hyperpigmentary','mh','sun-set glow',
                   'optic atrophy','serous-exudative rd','microaneurysm','rd','disc swelling and haemorrhage','yellow subretinal lesions',
                   'vitreous lens','optic nv','silicone oil','choroid coloboma','vasculitis','subretinal haemorrhage',
                   'trd','chrp','ps','choroiditis','ah','myelinated nerve fiber','surgery-air','dragged disc',
                   'nevus','preretinal haemorrhage','aneurysms','retinal folds','surgery-medicine','macular star',
                   'osteoma','dalen fuchs nodules']#38疾病修改后

class TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.transform = transform

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.make_pkl_dir(config.pkl_dir_path)

        # get train_val_df
        if not os.path.exists(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path)):

            self.train_val_df = self.get_train_val_df()
            print('\nself.train_val_df.shape: {}'.format(self.train_val_df.shape))

            # pickle dump the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('{}: dumped'.format(config.train_val_df_pkl_path))

        else:
            # pickle load the train_val_df
            with open(os.path.join(config.pkl_dir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
            print('\n{}: loaded'.format(config.train_val_df_pkl_path))
            print('self.train_val_df.shape: {}'.format(self.train_val_df.shape))

        self.all_classes, self.all_classes_dict,self.all_Pathology_classes,self.all_Pathology_classes_dict = self.choose_the_indices()

        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_dir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_classes_pkl_path))
        else:
            print('\n{}: already exists'.format(config.disease_classes_pkl_path))

        if not os.path.exists(os.path.join(config.pkl_dir_path, config.disease_Pathology_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_dir_path, config.disease_Pathology_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_Pathology_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_Pathology_classes_pkl_path))
        else:
            print('\n{}: already exists'.format(config.disease_Pathology_classes_pkl_path))

        self.new_df = self.train_val_df # this is the sampled train_val data
        print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))
        print('\nself.all_Pathology_classes_dict: {}'.format(self.all_Pathology_classes_dict))

    def resample(self):
        self.the_chosen, self.all_classes, self.all_classes_dict,self.all_Pathology_classes,self.all_Pathology_classes_dict = self.choose_the_indices()
        self.new_df = self.train_val_df.iloc[self.the_chosen, :]
        print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))
        print('\nself.all_Pathology_classes_dict: {}'.format(self.all_Pathology_classes_dict))

    def make_pkl_dir(self, pkl_dir_path):
        if not os.path.exists(pkl_dir_path):
            os.mkdir(pkl_dir_path)

    def get_train_val_df(self):

        # get the list of train_val data
        train_val_list = self.get_train_val_list()

        train_val_df = pd.DataFrame()
        print('\nbuilding train_val_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename = os.path.basename(self.df.iloc[i, 0])
            if filename in train_val_list:
                train_val_df = train_val_df.append(self.df.iloc[i:i + 1, :])

        return train_val_df

    def __getitem__(self, index):
        row = self.new_df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Symptom Labels'], ',')
        Pathology_labels = str.split(row['Pathology Labels'], ',')

        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab = lab.strip().lower()
            if lab in symptom_class:
                lab_idx = self.all_classes.index(lab)
                target[lab_idx] = 1

        Pathology_target = torch.zeros(len(self.all_Pathology_classes))
        for lab in Pathology_labels:
            lab = lab.strip().lower()
            if lab in pathology_class:
                lab_idx = self.all_Pathology_classes.index(lab)
                Pathology_target[lab_idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, Pathology_target,target

    def choose_the_indices(self):

        max_examples_per_class = 10000  # its the maximum number of examples that would be sampled in the training set for any class
        all_classes = {}
        all_Pathology_classes = {}
        length = len(self.train_val_df)
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length), length, replace=False))):
            temp = str.split(self.train_val_df.iloc[i, :]['Symptom Labels'], ',')
            temp_Pathology = str.split(self.train_val_df.iloc[i, :]['Pathology Labels'], ',')

            temp = list(set(temp))
            temp_Pathology = list(set(temp_Pathology))

            # choose if multiple labels
            if len(temp) > 1:
                bool_lis = [False] * len(temp)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_classes:
                        if all_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                # if all lables under upper limit, append
                if sum(bool_lis) == len(temp):
                    # maintain count
                    for t in temp:
                        t = t.strip().lower()
                        if t == '':
                            continue
                        if t not in symptom_class:
                            continue
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:  # these are single label images
                for t in temp:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in symptom_class:
                        continue
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class:  # 500
                            all_classes[t] += 1
            if len(temp_Pathology) > 1:
                bool_lis = [False] * len(temp_Pathology)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp_Pathology):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_Pathology_classes:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                    # if all lables under upper limit, append
                    if sum(bool_lis) == len(temp_Pathology):
                        # maintain count
                        for t in temp_Pathology:
                            t = t.strip().lower()
                            if t == '':
                                continue
                            if t not in pathology_class:
                                continue
                            if t not in all_Pathology_classes:
                                all_Pathology_classes[t] = 1
                            else:
                                all_Pathology_classes[t] += 1
            else:  # these are single label images
                for t in temp_Pathology:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in pathology_class:
                        continue
                    if t not in all_Pathology_classes:
                        all_Pathology_classes[t] = 1
                    else:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            all_Pathology_classes[t] += 1

        return sorted(list(all_classes)), all_classes, sorted(list(all_Pathology_classes)), all_Pathology_classes

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_train.csv')
        print('\n{} found: {}'.format(csv_path, os.path.exists(csv_path)))

        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, '*', '*'))]

        df['Image Index'] = df['image_links'].apply(lambda x: x.split('/')[-1])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Symptom Labels', 'Pathology Labels']]
        return merged_df

    def get_train_val_list(self):
        f = open(os.path.join(self.data_dir, 'train_val_list.txt'), 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.new_df)

class ValDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir

        self.transform = transform

        # full dataframe including train_val and test set
        self.df = self.get_df()
        print('self.df.shape: {}'.format(self.df.shape))

        self.make_pkl_dir(config.pkl_valdir_path)

        # get train_val_df
        if not os.path.exists(os.path.join(config.pkl_valdir_path, config.train_val_df_pkl_path)):

            self.train_val_df = self.get_train_val_df()
            print('\nself.train_val_df.shape: {}'.format(self.train_val_df.shape))

            # pickle dump the train_val_df
            with open(os.path.join(config.pkl_valdir_path, config.train_val_df_pkl_path), 'wb') as handle:
                pickle.dump(self.train_val_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('{}: dumped'.format(config.train_val_df_pkl_path))

        else:
            # pickle load the train_val_df
            with open(os.path.join(config.pkl_valdir_path, config.train_val_df_pkl_path), 'rb') as handle:
                self.train_val_df = pickle.load(handle)
            print('\n{}: loaded'.format(config.train_val_df_pkl_path))
            print('self.train_val_df.shape: {}'.format(self.train_val_df.shape))

        self.all_classes, self.all_classes_dict,self.all_Pathology_classes,self.all_Pathology_classes_dict = self.choose_the_indices()

        if not os.path.exists(os.path.join(config.pkl_valdir_path, config.disease_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_valdir_path, config.disease_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_classes_pkl_path))
        else:
            print('\n{}: already exists'.format(config.disease_classes_pkl_path))

        if not os.path.exists(os.path.join(config.pkl_valdir_path, config.disease_Pathology_classes_pkl_path)):
            # pickle dump the classes list
            with open(os.path.join(config.pkl_valdir_path, config.disease_Pathology_classes_pkl_path), 'wb') as handle:
                pickle.dump(self.all_Pathology_classes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print('\n{}: dumped'.format(config.disease_Pathology_classes_pkl_path))
        else:
            print('\n{}: already exists'.format(config.disease_Pathology_classes_pkl_path))

        self.new_df = self.train_val_df # this is the sampled train_val data
        print('\nself.val_all_classes_dict: {}'.format(self.all_classes_dict))
        print('\nself.val_all_Pathology_classes_dict: {}'.format(self.all_Pathology_classes_dict))

    def resample(self):
        self.the_chosen, self.all_classes, self.all_classes_dict,self.all_Pathology_classes,self.all_Pathology_classes_dict = self.choose_the_indices()
        self.new_df = self.train_val_df.iloc[self.the_chosen, :]
        print('\nself.all_classes_dict: {}'.format(self.all_classes_dict))
        print('\nself.all_Pathology_classes_dict: {}'.format(self.all_Pathology_classes_dict))

    def make_pkl_dir(self, pkl_valdir_path):
        if not os.path.exists(pkl_valdir_path):
            os.mkdir(pkl_valdir_path)

    def get_train_val_df(self):

        # get the list of train_val data
        train_val_list = self.get_train_val_list()

        train_val_df = pd.DataFrame()
        print('\nbuilding train_val_df...')
        for i in tqdm(range(self.df.shape[0])):
            filename = os.path.basename(self.df.iloc[i, 0])
            if filename in train_val_list:
                train_val_df = train_val_df.append(self.df.iloc[i:i + 1, :])

        return train_val_df

    def __getitem__(self, index):
        row = self.new_df.iloc[index, :]

        img = cv2.imread(row['image_links'])
        labels = str.split(row['Symptom Labels'], ',')
        Pathology_labels = str.split(row['Pathology Labels'], ',')
        name = row['Image Index']


        target = torch.zeros(len(self.all_classes))
        for lab in labels:
            lab = lab.strip().lower()
            if lab in symptom_class:
                lab_idx = self.all_classes.index(lab)
                target[lab_idx] = 1

        Pathology_target = torch.zeros(len(self.all_Pathology_classes))
        for lab in Pathology_labels:
            lab = lab.strip().lower()
            if lab in pathology_class:
                lab_idx = self.all_Pathology_classes.index(lab)
                Pathology_target[lab_idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, Pathology_target,target,name

    def choose_the_indices(self):

        max_examples_per_class = 10000  # its the maximum number of examples that would be sampled in the training set for any class

        all_classes = {}
        all_Pathology_classes = {}

        length = len(self.train_val_df)
        print('\nSampling the huuuge training dataset')
        for i in tqdm(list(np.random.choice(range(length), length, replace=False))):
            temp = str.split(self.train_val_df.iloc[i, :]['Symptom Labels'], ',')
            temp_Pathology = str.split(self.train_val_df.iloc[i, :]['Pathology Labels'], ',')

            temp = list(set(temp))
            temp_Pathology = list(set(temp_Pathology))

            # choose if multiple labels
            if len(temp) > 1:
                bool_lis = [False] * len(temp)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_classes:
                        if all_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                # if all lables under upper limit, append
                if sum(bool_lis) == len(temp):
                    # maintain count
                    for t in temp:
                        t = t.strip().lower()
                        if t == '':
                            continue
                        if t not in symptom_class:
                            continue
                        if t not in all_classes:
                            all_classes[t] = 1
                        else:
                            all_classes[t] += 1
            else:  # these are single label images
                for t in temp:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in symptom_class:
                        continue
                    if t not in all_classes:
                        all_classes[t] = 1
                    else:
                        if all_classes[t] < max_examples_per_class:  # 500
                            all_classes[t] += 1
            if len(temp_Pathology) > 1:
                bool_lis = [False] * len(temp_Pathology)
                # check if any label crosses the upper limit
                for idx, t in enumerate(temp_Pathology):
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t in all_Pathology_classes:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            bool_lis[idx] = True
                    else:
                        bool_lis[idx] = True
                    # if all lables under upper limit, append
                    if sum(bool_lis) == len(temp_Pathology):
                        # maintain count
                        for t in temp_Pathology:
                            t = t.strip().lower()
                            if t == '':
                                continue
                            if t not in pathology_class:
                                continue
                            if t not in all_Pathology_classes:
                                all_Pathology_classes[t] = 1
                            else:
                                all_Pathology_classes[t] += 1
            else:  # these are single label images
                for t in temp_Pathology:
                    t = t.strip().lower()
                    if t == '':
                        continue
                    if t not in pathology_class:
                        continue
                    if t not in all_Pathology_classes:
                        all_Pathology_classes[t] = 1
                    else:
                        if all_Pathology_classes[t] < max_examples_per_class:  # 500
                            all_Pathology_classes[t] += 1

        return sorted(list(all_classes)), all_classes, sorted(list(all_Pathology_classes)), all_Pathology_classes

    def get_df(self):
        csv_path = os.path.join(self.data_dir, 'Data_val.csv')
        print('\n{} found: {}'.format(csv_path, os.path.exists(csv_path)))

        all_xray_df = pd.read_csv(csv_path)

        df = pd.DataFrame()
        df['image_links'] = [x for x in glob.glob(os.path.join(self.data_dir, '*', '*'))]

        df['Image Index'] = df['image_links'].apply(lambda x: x.split('/')[-1])
        merged_df = df.merge(all_xray_df, how='inner', on=['Image Index'])
        merged_df = merged_df[['image_links', 'Symptom Labels', 'Pathology Labels','Image Index']]
        return merged_df

    def get_train_val_list(self):
        f = open(os.path.join(self.data_dir, 'train_val_list.txt'), 'r')
        train_val_list = str.split(f.read(), '\n')
        return train_val_list

    def __len__(self):
        return len(self.new_df)
