# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
