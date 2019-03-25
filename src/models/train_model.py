#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, TensorDataset, Sampler
from torch.utils.data.dataloader import DataLoader
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
import seaborn as sns
from PIL import Image
import sys
sys.path.insert(0, "./chex")
import densenet
import adamw
import cosine_scheduler
import compare_auc_delong_xu 
import h5py

# In[ ]:


#Top level data directory (can be 'CheXpert-v1.0-small', 'CheXpert-v1.0', 'mimic-cxr')
root_dir = 'Data/'
which_data = 'CheXpert-v1.0'
data_dir = f'{root_dir}{which_data}'

#Data is split into two sets (may later create a training-dev set)
phases = ['train', 'val']

#Path to csvfiles on training and validation data
csvpath = {phase: '{}/{}.csv'.format(data_dir, phase) for phase in phases}

#Load data into dictionary of two Pandas DataFrames
dframe = {phase: pd.read_csv(csvpath[phase]) for phase in phases}

#Calculate sizes
dataset_sizes = {phase:len(dframe[phase]) for phase in phases}

print(os.listdir(data_dir))
print(dframe['train'].shape, dframe['val'].shape)


# In[ ]:


# Models to choose from [efficient-densenet, resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 14

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 10

#Number of minibatches to pass before printing out metrics
checkpoint = 200

# Flag for feature extracting. When False, we finetune the whole model, when True we only update the reshaped layer params
feature_extract = False

#The number of training samples
num_samples = dframe['train'].shape[0]

#Class names that will be predicted
class_names = dframe['train'].iloc[:,5:].columns

#indices we will calculate AUC for, as in the CheXpert paper
competition_tasks = torch.ByteTensor([0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0])

#Which approach to use when dealing with uncertain labels (as detailed in the CheXpert paper)
u_approach = 'ones'

# Pick the model
model_name = 'resnet'

#Using pretrained weights
use_pretrained = True

#filename for outputs
filename = f'{which_data}_{model_name}_{u_approach}_{batch_size}'

#Calculating weighting for imbalanced classes (input in loss criterion)
df = dframe['train'].iloc[:,5:].copy()
df = df.replace(-1,0)
pos_weight = torch.Tensor([df[cl].sum()/df.shape[0] for cl in class_names])


# In[ ]:


# Use CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print('Using CUDA') if use_cuda else print('Using CPU')

print(torch.cuda.memory_cached()/1e9)
print(torch.cuda.max_memory_cached()/1e9)
print(torch.cuda.memory_allocated()/1e9)
print(torch.cuda.max_memory_allocated()/1e9)


# In[ ]:


def train_model(model, dataloaders, criterion, optimizer, scheduler, competition_tasks, 
                num_epochs=25, max_fpr=None, u_approach=None, is_inception=False, checkpoint=200):
    
    since = time.time() 
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc = 0.0
    missing = 0;
    losses = {'train':[],'val':[]}; 
    accuracy = {'train':[],'val':[]}; 
    variances = {'train':[],'val':[]}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0; running_auc = 0.0; running_var = 0.0
            
            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                    if u_approach == "ignore":
                        mask = labels.lt(0) #select u labels (-1)
                        loss = torch.sum(loss.masked_select(mask)) #mask out uncertain labels
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
#                         scheduler.batch_step() #USE WITH DENSENET and other scheduler

                # statistics
                running_loss += loss.item() * inputs.size(0)

                #select subset of 5 pathologies of interest
                labels_sub = labels[:,competition_tasks].cpu().squeeze().numpy()
                preds_sub = outputs[:,competition_tasks].detach().cpu().squeeze().numpy()

                if u_approach == "ignore":
                    #mask out the negative values
                    mask_sub = (labels_sub>-1) 
                    for j in range(labels_sub.shape[1]):     
                        label = labels_sub[:,j]
                        pred = preds_sub[:,j]
                        m = mask_sub[:,j]
                        label = label[m]; pred = pred[m]
                        try:
                            tmp = compare_auc_delong_xu.delong_roc_variance(label, pred)
                            running_auc += tmp[0]
                            running_var += np.nansum(tmp[1])
                        except:
                            missing += 1;
                            continue
                else:
                    for j in range(labels_sub.shape[1]):     
                        label = labels_sub[:,j]
                        pred = preds_sub[:,j]
                        tmp = compare_auc_delong_xu.delong_roc_variance(label, pred)
                        running_auc += tmp[0]
                        running_var += np.nansum(tmp[1])

#                 if (i+1) % checkpoint == 0:    # print every 'checkpoint' mini-batches
                if (i+1) % 200 == 0:    # print every 'checkpoint' mini-batches
#                     print('Missed {}'.format(missing))
                    print(f'{phase} Loss: {running_loss / (i+1)} DeLong AUC: {running_auc / (labels_sub.shape[1] * (i+1) * batch_size)} Variance: {running_var / (labels_sub.shape[1] * (i+1) * batch_size)}')
                    
                    losses[phase].append(running_loss / ((i+1) * batch_size))
                    accuracy[phase].append(running_auc / (labels_sub.shape[1] * (i+1) * batch_size))
                    variances[phase].append(running_var / (labels_sub.shape[1] * (i+1) * batch_size))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_auc = running_auc / (dataset_sizes[phase] * labels_sub.shape[1])
            epoch_var = running_var / (dataset_sizes[phase] * labels_sub.shape[1])
            print(f'{phase} Epoch Loss: {epoch_loss} Epoch AUC: {epoch_auc} Epoch Variance: {epoch_var}')
            #With a small validation set would otherwise get no recorded values so:
            if phase == 'val':
                losses[phase].append(epoch_loss)
                accuracy[phase].append(epoch_auc)
                variances[phase].append(epoch_var)  
            
            # deep copy the model
            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val AUC: {best_auc}')
    print(f'Missed {missing} examples.')
    # load best model weights
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy, variances)
#     for phase in ['train', 'val']:
#         with open(f'metrics/{filename}_{phase}.txt','w+') as f:
#             for idx in len(losses[phase]):
#                 f.write(f'{losses[phase][idx]} {accuracy[phase][idx]} {variances[phase][idx]}\n')

    return model, metrics


# In[ ]:


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if model_name == "efficient-densenet":
        """memory-efficient densenet - https://github.com/gpleiss/efficient_densenet_pytorch
        """
        # # 
        growth_rate  = 32    #(int) - how many filters to add each layer (`k` in paper)
        block_config = [6, 12, 24, 16]     #(list of 3 or 4 ints) - how many layers in each pooling block
        compression = 0.5   #Reduction in size
        num_init_features = 2 * growth_rate    #(int) - the number of filters to learn in the first convolution layer
        bn_size =  4   #(int) - mult. factor for number of bottle neck layers (i.e. bn_size * k features in the bottleneck layer)
        drop_rate = 0.     #(float) - dropout rate after each dense layer
        small_inputs = False     #(bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient = True     #(bool) - set to True to use checkpointing. Much more memory efficient, but slower.
        model_ft = densenet.DenseNet(growth_rate=growth_rate, block_config=block_config, compression=compression, 
                         num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate, num_classes=num_classes, 
                        small_inputs=small_inputs, efficient=efficient)
        if use_pretrained:
            state_dict = torch.load('chexpredict/densenet121_effi.pth')
            model.load_state_dict(state_dict, strict=True)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


# In[ ]:


class CheXpertDataset(Dataset):
    """CheXpert dataset."""
    def __init__(self, data_dir, phase, u_approach, num_samples, hdf5path):
        """
        Args:
            chex_frame (DataFrame): Train or validation data
            root_dir (string): Directory with all the images.
            tforms (Torch Transform): pre-processing transforms
            targets (DataFrame): Modifies labels depending on u_approach
        """
        self.data_dir = data_dir
        self.phase = phase
        self.u_approach = u_approach
        self.num_samples = num_samples
        self.hdf5path = hdf5path
        # self.hdf5 = None
        self.hdf5 = h5py.File(self.hdf5path, 'r')
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        
        # if(self.hdf5 is None): #open in thread
        #     self.hdf5 = h5py.File(self.hdf5path, 'r', libver='latest', swmr=True)
        img = self.hdf5[(f"X{idx}")][()]
        labels = self.hdf5[(f"y{idx}")][()]
        #             img = torch.FloatTensor(img)
        #             labels = torch.FloatTensor(labels)
        return img, np.float32(labels)


# #         with h5py.File(self.hdf5path, 'r') as hf:
#         with h5py.File(self.hdf5path, 'r', libver='latest', swmr=True) as hf:
#             img = hf[(f"X{idx}")]
#             labels = hf[(f"y{idx}")]
#             img = torch.FloatTensor(img)
#             labels = torch.FloatTensor(labels)
#             return (img, labels)


# In[ ]:


#Initialize
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)

#preprocess target labels before placing in dataloader
labels_array = {phase: dframe[phase].iloc[:,5:].copy().fillna(0) for phase in phases}

for phase in labels_array.keys():
    if u_approach == 'ones':
        labels_array[phase] = labels_array[phase].replace(-1,1)
    elif u_approach == 'zeros':
        labels[phase] = labels_array[phase].replace(-1,0)
    labels_array[phase] = torch.FloatTensor(labels_array[phase].to_numpy()) #needed when using cross-entropy loss

#Transforms to perform on images
tforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}
hdf5paths = {phase: f'{data_dir}/{phase}_u{u_approach}_inp{input_size}_processed.h5' for phase in phases}
# Create the datasets
datasets = {phase: CheXpertDataset(data_dir=data_dir, phase=phase, u_approach=u_approach, num_samples=dataset_sizes[phase], hdf5path=hdf5paths[phase]) for phase in phases}


# In[ ]:


def proc_images(img_paths, labels_array, data_dir, u_approach, input_size, phases=['train', 'val'], tforms=None):
    """
    Saves compressed, resized images as HDF5 datsets
    Returns
        data.h5, where each dataset is an image or class label
        e.g. X23,y23 = image and corresponding class labels
    """
    for phase in phases:
        print(f'Processing {phase} files...')
        with h5py.File(f'{data_dir}/{phase}_u{u_approach}_inp{input_size}_processed.h5', 'w') as hf: 
            for i,img_path in enumerate(img_paths[phase]):     
                if i % 2000 == 0:
                    print(f"{i} images processed")

                # Images
                #Using Pillow-SIMD rather than Pillow
                img = Image.open(img_path).convert('RGB')
                if tforms:
                    img = tforms[phase](img)
                Xset = hf.create_dataset(
                    name=f"X{i}",
                    data=img,
                    shape=(3, input_size, input_size),
                    maxshape=(3, input_size, input_size),
                    compression="lzf",
                    shuffle="True")
               # Labels
                yset = hf.create_dataset(
#                     name=f"y{i}",
                    name=f"y{i}",
                    data = labels_array[phase][i,:],
                    shape=(num_classes,),
                    maxshape=(num_classes,),
                    compression="lzf",
                    shuffle="True",
                    dtype="i1")
    print('Finished!')
try:
    os.path.isfile(f'{data_dir}/{phases[0]}_u{u_approach}_inp{input_size}_processed.h5') 
    os.path.isfile(f'{data_dir}/{phases[1]}_u{u_approach}_inp{input_size}_processed.h5') 
except:
    img_paths = {phase: root_dir + dframe[phase].iloc[:,0] for phase in phases}
    proc_images(img_paths, labels_array, data_dir, u_approach, input_size, phases=phases, tforms=tforms)


# In[ ]:


# #Accounting for imbalanced classes
# df = dframe['train'].iloc[:,5:].fillna(0).copy()
# # df = df.replace(-1, np.nan)

# #Get a list of the number of positive, negative, and uncertain samples for each class
# class_sample_count = [df[df==t].count() for t in [-1, 0, 1]]
# #Use this to calculate the weight for positive, negative, and uncertain samples for each class
# weights = [num_classes * class_sample_count[t] * (1 / num_samples) for t in range(len(class_sample_count))]
# #Create list of weights as long as the dataset, squeeze to put in sequence? 
# sample_weights = weights[0] * (df==-1) + weights[1] * (df==0) + weights[2] * (df==1)
# sample_weights.head(3)
# sample_weights = sample_weights.to_numpy()
# print(weights[1])
# #Load into sampler
# sampler = None
# # sampler = torch.utils.data.WeightedRandomSampler(sample_weights, 1, replacement=True)


# In[ ]:


# Setting parameters
# num_workers = torch.get_num_threads()
num_workers = 1
params = {'train': {'batch_size': batch_size,
                   'shuffle': True,
                   'num_workers': num_workers,
                   'pin_memory':True}, #Trying this out once more to see if it helps
#                    'sampler': sampler},
          'val': {'batch_size': batch_size,
                 'num_workers': num_workers,
                 'pin_memory':True}}
# if params['train']['sampler'] is not None:
#     params['train']['shuffle'] = False
dataloaders = {phase: DataLoader(datasets[phase], **params[phase]) for phase in phases}


# In[ ]:


# #Show transformed images
# fig = plt.figure()

# for i in range(len(datasets['val'])):
#     sample = datasets['val'][i]
# #     print(i, sample[0].shape, sample[1].shape)
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample[0][0,:,:])

#     if i == 3:
#         plt.show()
#         break


# In[ ]:


# #Checking for shuffling (or not)
# times = []
# t0 = time.time()
# for i, (inputs, labels) in enumerate(dataloaders['train']):
#     if i < 1:
# #         print(i)
#         times.append(time.time()-t0)
#         t0 = time.time()
# #         print('Time to load/transform: {:.3f} s'.format(t1-t0))
# #         print(inputs[1].size(0))
# #         target = inputs[1]
# #         target[target==-1]=0
# #         print(torch.sum(target,dim=0))
# #         target = inputs[1][:,competition_tasks]
# #         target[target==-1]=0
# #         print(torch.sum(target,dim=0))
# #         print(target.shape)
        
# #     else:
#         break
# print('Average time to load up data: {} s'.format(np.round(np.mean(times),4)))
# #With batch size of 64
# # 3.6-4 s for hdf5 datasets
# # 5.3-6 s for original method with Pillow-SIMD
# # 5.5-6.4 s for alternative PIL method
# # 5.7-6.1 s for cv2


# In[ ]:


# # Helper function to show a batch
# def show_img_batch(sample_batched):
#     """Show image with labels for a batch of samples."""
#     images_batch, labels_batch = \
#             sample_batched[0], sample_batched[1]
#     batch_size = len(images_batch)
#     im_size = images_batch.size(2)

#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))

# for i_batch, sample_batched in enumerate(dataloaders['train']):
#     print(i_batch, sample_batched[0].size(), sample_batched[1].size())
    
#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_img_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break


# In[ ]:


try:
    model_ft.load_state_dict(torch.load('models/{}.pt'.format(filename))) #load weights if already completed
except:
    print('Starting from scratch..')

model_ft = model_ft.to(device)

if u_approach == 'ignore': #Use masked binary cross-entropy for first run
    criterion = nn.BCEWithLogitsLoss(reduction='none',pos_weight=pos_weight).to(device)
else:
    criterion = nn.BCEWithLogitsLoss(reduction='sum',pos_weight=pos_weight).to(device)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# # Pick AdamW optimizer - https://github.com/mpyrozhok/adamwr
# optimizer = adamw.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# epoch_size = np.round(num_samples / batch_size) # number of training examples/batch size
# #Cosine annealing: adjusting on batch update rather than epoch - https://github.com/mpyrozhok/adamwr
# scheduler = cosine_scheduler.CosineLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=5, t_mult=1.2)


# In[ ]:


# model = model_ft
# optimizer = optimizer_ft
# scheduler = exp_lr_scheduler
# is_inception = False


# In[ ]:


model_ft, metrics = train_model(model_ft, dataloaders, criterion, optimizer_ft, exp_lr_scheduler, competition_tasks, 
                       num_epochs=num_epochs, u_approach=u_approach, is_inception=(model_name=="inception"), checkpoint=checkpoint)
datasets['train'].hdf5.close()
datasets['val'].hdf5.close()
torch.save(model_ft.state_dict(), f'models/{filename}.pt')


# In[ ]:


# #Visualizing model predictions
# def visualize_model(model, num_images=6):
#     tform = transforms.Compose([transforms.functional.to_pil_image,
#                                 transforms.functional.to_grayscale])
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()

#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 img = tform(inputs[j].cpu())
#                 ax.set_title('First prediction: {}'.format(class_names[preds[j]]))
#                 plt.imshow(img)

#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)
        
        
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    
# model_ft.load_state_dict(torch.load(f'models/{filename}.pt')) #load new weights
# model_ft.to(device)

# visualize_model(model_ft)


# In[ ]:


# #Path to csvfiles on training and validation data


# #Load data into dictionary of two Pandas DataFrames
# mf = {phase: pd.read_csv(metricspath[phase]) for phase in phases}

# plt.title("AUC vs. Number of Training Epochs")
# plt.xlabel("Training Epochs")
# plt.ylabel("DeLong AUC")
# plt.plot(range(1,num_epochs+1),mf['train'],label="train")
# plt.plot(range(1,num_epochs+1),mf['val'],label="val")
# plt.ylim((0,1.))
# # plt.xticks(np.arange(1, num_epochs+1, 1.0))
# plt.legend()
# plt.show()


# In[ ]:


# import re
# import pandas as pd
# import numpy as np
# import torch
# import torchvision
# from torchvision import models, transforms
# print("PyTorch Version: ",torch.__version__)
# print("Torchvision Version: ",torchvision.__version__)
# from PIL import Image

# #Initialize model
# filename = ""
# num_classes = 14
# feature_extract = False
# use_pretrained = True
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)

# #Load trained weights
# model_ft.load_state_dict(torch.load('models/{}.pt'.format(filename)))  
# model_ft.to(device)

# def predict(input_csv_name, output_csv_name, model):
#     '''
#     The input-data-csv-file path will point to a file which is structured exactly like 
#     valid_image_paths.csv, ie where each line is of the form 
#     CheXpert-v1.0/{valid,test}/<STUDYTYPE>/<PATIENT>/<STUDY>/<IMAGE>
#     '''
#     #Input series of image paths
#     img_paths = pd.read_csv(input_csv_name)
    
#     #Output file predictions for the following competition tasks:
#     col_names = ['Study', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
#     #Output dataframe
#     output_csv = pd.DataFrame(index=range(1,len(img_paths)), columns=col_names)
    
#     #Predicting for dummy validation and actual test data:
#     phases = ['valid', 'test']
        
#     #Image preprocessing
#     tforms = transforms.Compose([
#         transforms.Resize(input_size),
#         transforms.CenterCrop(input_size),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
#     #RegEx to determine study number and whether in validation or test data 
#     pattern = re.compile(r'(CheXpert-v1.0\/\w+\/\w+\/study\d)\/view\d_\w+\.jpg')
        
#     was_training = model.training
#     model.eval()
#     with torch.no_grad():
#         for i, img_path in enumerate(img_paths):
#             study = re.findall(pattern, img_path)
#             img = Image.open(img_path).convert('RGB') #Model expects 3 channels
#             img = tforms(img)
#             img = img.to(device)
#             output = model(img)
#             pred = outputs[competition_tasks].cpu().to_numpy()
            
#             output_csv.iloc[i, 0] = study
#             output_csv.iloc[i, 1:5] = pred    
        
#         model.train(mode=was_training)
        
#     output_csv.to_csv(output_csv_name, index=False)
        
        
# predict(input_csv_name, model)

