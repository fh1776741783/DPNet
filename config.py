pkl_dir_path             = 'pickles'
pkl_valdir_path          = 'pickle'
train_val_df_pkl_path    = 'train_val_df.pickle'
test_df_pkl_path         = 'test_df.pickle'
disease_classes_pkl_path = 'disease_classes.pickle'
disease_Pathology_classes_pkl_path = 'disease_Pathology_classes.pickle'
models_dir               = 'models'

from torchvision import transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize([512,648]),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.ColorJitter(brightness=(1,1.3)),
                    transforms.ToTensor(),
                    normalize
                                ])

transform_val = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize([512,648]),
                    transforms.ToTensor(),
                    normalize
                                 ])
