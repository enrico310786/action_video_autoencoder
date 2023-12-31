import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml
from data import create_loaders
import numpy as np
import matplotlib.pyplot as plt
from upload_s3 import multiup
import random
import wandb
import seaborn as sns
import gc

from model import SpaceTimeAutoencoder, find_last_checkpoint_file


def train_batch(inputs, model, optimizer, criterion):
    model.train()
    target, outputs, latent = model(inputs)
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


@torch.no_grad()
def val_loss(inputs, model, criterion):
    model.eval()
    target, outputs, latent = model(inputs)
    val_loss = criterion(outputs, target)
    return val_loss.item()


def calculate_errors_and_distributions(device,
                                       model,
                                       dataloader,
                                       label2class,
                                       df_distribution,
                                       type_dataset,
                                       path_save,
                                       number_of_classes,
                                       cfg,
                                       embedding_centroids=None,
                                       latent_centroids=None,
                                       df_anomaly_distances=None):
    '''
    1) Calculate the error reconstruction of the embedding vectors
    2) Calculate the train latent centroids for each class
    3) For each class, calculate the distances of the latent arrays from the corresponding centroids
    4) Calculate the train embedding centroids for each class
    5) For each class, calculate the distances of the embeddings from the corresponding centroids
    6) Look the TSNE distribution of the train embedding arrays and of the train latent arrays
    7) For the anomaly dataset calculate the distance of the embedding from each class centroid of the train set
    '''


    # centroids is the list of the centroids of each class calculated in the train set, both for latent arrays and embedding arrays
    if embedding_centroids is None:
        embedding_centroids = []
    if latent_centroids is None:
        latent_centroids = []

    class_labels = []
    class_names = []
    latents_array = None
    reconstructed_embeddings_array = None
    embeddings_array = None
    model = model.eval()

    with torch.no_grad():
        # cycle on all batches
        for inputs, labels, classes in dataloader:
            if cfg['model']['name_time_model'] == "3d_slowfast":
                inputs = [i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            classes = list(classes)
            embeddings, reconstructed_embeddings, latents = model(inputs)
            #embeddings = model.base_model(inputs)
            #reconstructed_embeddings = model(inputs)
            #latents = model.encoder(embeddings)

            # stack the results
            if latents_array is None:
                latents_array = latents.detach().cpu().numpy()
            else:
                latents_array = np.vstack((latents_array, latents.detach().cpu().numpy()))

            if embeddings_array is None:
                embeddings_array = embeddings.detach().cpu().numpy()
            else:
                embeddings_array = np.vstack((embeddings_array, embeddings.detach().cpu().numpy()))

            if reconstructed_embeddings_array is None:
                reconstructed_embeddings_array = reconstructed_embeddings.detach().cpu().numpy()
            else:
                reconstructed_embeddings_array = np.vstack((reconstructed_embeddings_array, reconstructed_embeddings.detach().cpu().numpy()))

            class_labels.extend(labels.detach().cpu().numpy().tolist())
            class_names.extend(classes)

        # transform to numpy array
        class_labels = np.array(class_labels)

        print('len(class_names): ', len(class_names))
        print('class_labels.shape: ', class_labels.shape)
        print('latents_array.shape: ', latents_array.shape)
        print('reconstructed_embeddings_array.shape: ', reconstructed_embeddings_array.shape)
        print('embeddings_array.shape: ', embeddings_array.shape)

        if len(embedding_centroids) == 0:
            # calculate the centroid of the embedding vectors for each class. Iter over the label of each class
            for idx in label2class.keys():
                filter_idxs = np.where(class_labels == idx)[0]
                #print("filter_idxs: ", filter_idxs)
                embeddings_array_filtered = np.take(embeddings_array, filter_idxs, 0)
                centroid = np.mean(embeddings_array_filtered, axis=0)
                embedding_centroids.append(centroid)
        embedding_centroids = np.array(embedding_centroids)

        if len(latent_centroids) == 0:
            # calculate the centroid of the latent vectors for each class. Iter over the label of each class
            for idx in label2class.keys():
                filter_idxs = np.where(class_labels == idx)[0]
                #print("filter_idxs: ", filter_idxs)
                latents_array_filtered = np.take(latents_array, filter_idxs, 0)
                centroid = np.mean(latents_array_filtered, axis=0)
                latent_centroids.append(centroid)
        latent_centroids = np.array(latent_centroids)

        # iter over the array to find error and distances from the correspondig centroid. The centroid is found using the label of the class
        if type_dataset != "ANOMALY":
            for label, emb, rec_emb, latent in zip(class_labels, embeddings_array, reconstructed_embeddings_array, latents_array):
                #error = (np.square(emb - rec_emb)).mean()
                error = (np.square(emb - rec_emb)).sum()
                latent_dist = np.linalg.norm(latent - latent_centroids[label])
                emb_dist = np.linalg.norm(emb - embedding_centroids[label])
                category = label2class[label]

                df_distribution = df_distribution.append({'CLASS': category,
                                                          'LABEL': label,
                                                          'ERROR': error,
                                                          'LATENT_DISTANCE': latent_dist,
                                                          'EMBEDDING_DISTANCE': emb_dist,
                                                          'TYPE_DATASET': type_dataset}, ignore_index=True)
        elif type_dataset == "ANOMALY":
            # iter over the embedding array
            for class_name, emb, rec_emb in zip(class_names, embeddings_array, reconstructed_embeddings_array):
                error = (np.square(emb - rec_emb)).sum()
                df_distribution = df_distribution.append({'CLASS': class_name,
                                                          'LABEL': None,
                                                          'ERROR': error,
                                                          'LATENT_DISTANCE': None,
                                                          'EMBEDDING_DISTANCE': None,
                                                          'TYPE_DATASET': type_dataset}, ignore_index=True)

                # for each embedding array iter over the centroid to find the distance between the centroid and the array
                for idx in label2class.keys():
                    emb_dist = np.linalg.norm(emb - embedding_centroids[idx])
                    class_centroid = label2class[idx]
                    df_anomaly_distances = df_anomaly_distances.append({'ANOMALY_CLASS': class_name,
                                                                        'CLASS_CENTROID': class_centroid,
                                                                        'DIST_CENTROID': emb_dist}, ignore_index=True)

        if type_dataset == "TRAIN":

            # find TSNE distribution for latent array
            print("find TSNE distribution for latent array")
            tsne = TSNE(2)
            clustered = tsne.fit_transform(latents_array)
            fig = plt.figure(figsize=(12, 10))
            cmap = plt.get_cmap('Spectral', number_of_classes)
            plt.scatter(*zip(*clustered), c=class_labels, cmap=cmap)
            plt.colorbar(drawedges=True)
            fig.savefig(os.path.join(path_save, "TSNE_latent_array_train.png"))

            # find TSNE distribution for embedding array
            print("find TSNE distribution for embedding array")
            tsne = TSNE(2)
            clustered = tsne.fit_transform(embeddings_array)
            fig = plt.figure(figsize=(12, 10))
            cmap = plt.get_cmap('Spectral', number_of_classes)
            plt.scatter(*zip(*clustered), c=class_labels, cmap=cmap)
            plt.colorbar(drawedges=True)
            fig.savefig(os.path.join(path_save, "TSNE_embedding_array_train.png"))

        return df_distribution, latent_centroids, embedding_centroids, df_anomaly_distances

def train_model(cfg,
                device,
                model,
                criterion,
                optimizer,
                lr_scheduler,
                train_loader,
                val_loader,
                best_epoch,
                num_epoch,
                best_val_epoch_loss,
                checkpoint_dir,
                saving_dir_experiments,
                epoch_start_unfreeze=None,
                layer_start_unfreeze=None,
                aws_bucket=None,
                aws_directory=None,
                scheduler_type=None):

    train_losses = []
    val_losses = []

    print("Start training")
    freezed = True
    for epoch in range(best_epoch, num_epoch):

        if epoch_start_unfreeze is not None and epoch >= epoch_start_unfreeze and freezed:
            print("****************************************")
            print("Unfreeze the base model weights")
            if layer_start_unfreeze is not None:
                print("unfreeze the layers greater and equal to layer_start_unfreeze: ", layer_start_unfreeze)
                #in this case unfreeze only the layers greater and equal the unfreezing_block layer
                for i, properties in enumerate(model.named_parameters()):
                    if i >= layer_start_unfreeze:
                        #print("Unfreeze model layer: {} -  name: {}".format(i, properties[0]))
                        properties[1].requires_grad = True
            else:
                # in this case unfreeze all the layers of the model
                print("unfreeze all the layer of the model")
                for name, param in model.named_parameters():
                    param.requires_grad = True

            freezed = False
            print("*****************************************")
            print("Model layer info after unfreezing")

            print("Check layers properties")
            for i, properties in enumerate(model.named_parameters()):
                print("Model layer: {} -  name: {} - requires_grad: {} ".format(i, properties[0],
                                                                                properties[1].requires_grad))
            print("*****************************************")

            pytorch_total_params = sum(p.numel() for p in model.parameters())
            pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("pytorch_total_params: ", pytorch_total_params)
            print("pytorch_total_trainable_params: ", pytorch_total_trainable_params)

            print("*****************************************")

        # define empty lists for the values of the loss of train and validation obtained in the batch of the current epoch
        # then at the end I take the average and I get the final values of the whole era
        train_epoch_losses = []
        val_epoch_losses = []

        # cycle on all train batches of the current epoch by executing the train_batch function
        for inputs, _, _ in tqdm(train_loader, desc=f"epoch {str(epoch)} | train"):
            if cfg['model']['name_time_model'] == "3d_slowfast":
                inputs = [i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)
            #print("inputs[0].size(): ", inputs[0].size())
            #print("inputs[1].size(): ", inputs[1].size())
            batch_loss = train_batch(inputs, model, optimizer, criterion)
            train_epoch_losses.append(batch_loss)
            torch.cuda.empty_cache()
        train_epoch_loss = np.array(train_epoch_losses).mean()

        # cycle on all batches of val of the current epoch by calculating the accuracy and the loss function
        for inputs, _, _ in tqdm(val_loader, desc=f"epoch {str(epoch)} | val"):
            if cfg['model']['name_time_model'] == "3d_slowfast":
                inputs = [i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)
            validation_loss = val_loss(inputs, model, criterion)
            val_epoch_losses.append(validation_loss)
            torch.cuda.empty_cache()
        val_epoch_loss = np.mean(val_epoch_losses)

        wandb.log({'Learning Rate': optimizer.param_groups[0]['lr'],
                   'Train Loss': train_epoch_loss,
                   'Valid Loss': val_epoch_loss})

        print("Epoch: {} - LR:{} - Train Loss: {:.4f} - Val Loss: {:.4f}".format(int(epoch), optimizer.param_groups[0]['lr'], train_epoch_loss, val_epoch_loss))

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        print("Plot learning curves")
        plot_learning_curves(epoch - best_epoch + 1, train_losses, val_losses, checkpoint_dir)

        if best_val_epoch_loss > val_epoch_loss:
            print("We have a new best model! Save the model")
            # update best_val_epoch_loss
            best_val_epoch_loss = val_epoch_loss
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eval_loss': best_val_epoch_loss
            }
            print("Save best checkpoint at: {}".format(os.path.join(checkpoint_dir, 'best.pth')))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'best.pth'),  _use_new_zipfile_serialization=False)
            print("Save latest checkpoint at: {}".format(os.path.join(checkpoint_dir, 'latest.pth')))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'latest.pth'),  _use_new_zipfile_serialization=False)
        else:
            print("Save the current model")
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_eval_loss': best_val_epoch_loss
            }
            print("Save latest checkpoint at: {}".format(os.path.join(checkpoint_dir, 'latest.pth')))
            torch.save(save_obj, os.path.join(checkpoint_dir, 'latest.pth'),  _use_new_zipfile_serialization=False)

        if scheduler_type == "ReduceLROnPlateau":
            print("lr_scheduler.step(val_epoch_loss)")
            lr_scheduler.step(val_epoch_loss)
        else:
            print("lr_scheduler.step()")
            lr_scheduler.step()

        if aws_bucket is not None and aws_directory is not None:
            print('Upload on S3')
            multiup(aws_bucket, aws_directory, saving_dir_experiments)

        torch.cuda.empty_cache()
        gc.collect()
        print("---------------------------------------------------------")

    print("End training")
    return


def plot_learning_curves(epochs, train_losses, val_losses, path_save):
    '''
    Plot learning curves of the training model
    '''
    x_axis = range(0, epochs)

    plt.figure(figsize=(27,9))
    plt.suptitle('Learning curves ', fontsize=18)

    plt.subplot(121)
    plt.plot(x_axis, train_losses, label='Training Loss')
    plt.plot(x_axis, val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Train and Validation Losses', fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)

    plt.savefig(os.path.join(path_save, "learning_curves.png"))



def analyze_error_distribution(df_distribution, dir_save_results):

    print("-------------------------------------------------------------------")
    print("ERROR DISTRIBUTION")
    print("")
    desc = df_distribution[df_distribution['TYPE_DATASET'] == 'TRAIN']['ERROR'].describe()
    print("ERROR distrbution for dataset TRAIN: ")
    print(desc)
    print("")
    desc = df_distribution[df_distribution['TYPE_DATASET'] == 'VAL']['ERROR'].describe()
    print("ERROR distrbution for dataset VAL: ")
    print(desc)
    print("")
    desc = df_distribution[df_distribution['TYPE_DATASET'] == 'TEST']['ERROR'].describe()
    print("ERROR distrbution for dataset TEST: ")
    print(desc)
    print("")
    desc = df_distribution[df_distribution['TYPE_DATASET'] == 'ANOMALY']['ERROR'].describe()
    print("ERROR distrbution for dataset ANOMALY: ")
    print(desc)
    print("-------------------------------------------------------------------")

    # plotting in a one figure
    print("Plot error distribution grouped by dataset")
    plt.figure(figsize=(7, 7))
    sns.boxplot(data=df_distribution, x="TYPE_DATASET", y="ERROR")
    plt.yticks(rotation=90)
    plt.savefig(os.path.join(dir_save_results, "error_distribution.png"))

    print("-------------------------------------------------------------------")
    print("ERROR DISTRIBUTION GROUPED BY CLASS")
    print("")
    desc_grouped = df_distribution[df_distribution['TYPE_DATASET'] == 'TRAIN'].groupby('CLASS')["ERROR"].describe()
    print("ERROR distrbution for dataset TRAIN: ")
    print(desc_grouped)
    print("")
    desc_grouped = df_distribution[df_distribution['TYPE_DATASET'] == 'VAL'].groupby('CLASS')["ERROR"].describe()
    print("ERROR distrbution for dataset VAL: ")
    print(desc_grouped)
    print("")
    desc_grouped = df_distribution[df_distribution['TYPE_DATASET'] == 'TEST'].groupby('CLASS')["ERROR"].describe()
    print("ERROR distrbution for dataset TEST: ")
    print(desc_grouped)
    print("")
    desc_grouped = df_distribution[df_distribution['TYPE_DATASET'] == 'ANOMALY'].groupby('CLASS')["ERROR"].describe()
    print("ERROR distrbution for dataset ANOMALY: ")
    print(desc_grouped)
    print("-------------------------------------------------------------------")

    # boxplot
    print("Plot error distribution grouped by dataset and classes")
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=df_distribution[df_distribution['TYPE_DATASET'] == 'TRAIN'], x="CLASS", y="ERROR")
    plt.xticks(rotation=45)
    plt.title('train set', fontsize=12)

    plt.subplot(2, 2, 2)
    sns.boxplot(data=df_distribution[df_distribution['TYPE_DATASET'] == 'VAL'], x="CLASS", y="ERROR")
    plt.xticks(rotation=45)
    plt.title('val set', fontsize=12)

    plt.subplot(2, 2, 3)
    sns.boxplot(data=df_distribution[df_distribution['TYPE_DATASET'] == 'TEST'], x="CLASS", y="ERROR")
    plt.xticks(rotation=45)
    plt.title('test set', fontsize=12)

    plt.subplot(2, 2, 4)
    sns.boxplot(data=df_distribution[df_distribution['TYPE_DATASET'] == 'ANOMALY'], x="CLASS", y="ERROR")
    plt.xticks(rotation=45)
    plt.title('anomaly set', fontsize=12)

    plt.savefig(os.path.join(dir_save_results, "error_distribution_grouped_by_class.png"))


def analyze_embedding_distances(df_distribution, df_anomaly_distances, dir_save_results):

    print("-------------------------------------------------------------------")
    print("EMBEDDING DISTANCES DISTRIBUTION")
    print("")
    desc = df_distribution[df_distribution['TYPE_DATASET'] == 'TRAIN'].groupby('CLASS')['EMBEDDING_DISTANCE'].describe()
    print("EMBEDDING DISTANCE distrbution for dataset TRAIN: ")
    print(desc)
    print("")
    desc = df_distribution[df_distribution['TYPE_DATASET'] == 'VAL'].groupby('CLASS')['EMBEDDING_DISTANCE'].describe()
    print("EMBEDDING DISTANCE distrbution for dataset VAL: ")
    print(desc)
    print("")
    desc = df_distribution[df_distribution['TYPE_DATASET'] == 'TEST'].groupby('CLASS')['EMBEDDING_DISTANCE'].describe()
    print("EMBEDDING DISTANCE distrbution for dataset TEST: ")
    print(desc)
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")
    print("")
    desc = df_anomaly_distances['DIST_CENTROID'].describe()
    print("EMBEDDING DISTANCE DISTRIBUTION FOR ANOMALY SET: ")
    print(desc)
    print("")
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")
    print("")
    desc = df_anomaly_distances.groupby('CLASS_CENTROID')['DIST_CENTROID'].describe()
    print("EMBEDDING DISTANCE DISTRIBUTION GROUPED BY CLASS CENTROID FOR ANOMALY SET: ")
    print(desc)
    print("")

    # boxplot
    print("Plot embedding distances distribution grouped by dataset and classes")
    plt.figure(figsize=(15, 15))

    plt.subplot(2, 2, 1)
    sns.boxplot(data=df_distribution[df_distribution['TYPE_DATASET'] == 'TRAIN'], x="CLASS", y="EMBEDDING_DISTANCE")
    plt.xticks(rotation=45)
    plt.title('train set', fontsize=12)

    plt.subplot(2, 2, 2)
    sns.boxplot(data=df_distribution[df_distribution['TYPE_DATASET'] == 'VAL'], x="CLASS", y="EMBEDDING_DISTANCE")
    plt.xticks(rotation=45)
    plt.title('val set', fontsize=12)

    plt.subplot(2, 2, 3)
    sns.boxplot(data=df_distribution[df_distribution['TYPE_DATASET'] == 'TEST'], x="CLASS", y="EMBEDDING_DISTANCE")
    plt.xticks(rotation=45)
    plt.title('test set', fontsize=12)

    plt.subplot(2, 2, 4)
    sns.boxplot(data=df_anomaly_distances, x="CLASS_CENTROID", y="DIST_CENTROID")
    plt.xticks(rotation=45)
    plt.title('anomaly set', fontsize=12)

    plt.savefig(os.path.join(dir_save_results, "embedding_distances_distribution_grouped_by_class.png"))


def run_train_test_model(cfg, do_train, do_test, aws_bucket=None, aws_directory=None):

    seed_everything(42)
    checkpoint = None
    best_epoch = 0
    best_val_epoch_loss = float('inf')

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    dataset_path = dataset_cfg['dataset_path']
    path_dataset_train_csv = dataset_cfg['path_dataset_train_csv']
    path_dataset_val_csv = dataset_cfg['path_dataset_val_csv']
    path_dataset_test_csv = dataset_cfg.get("path_dataset_test_csv", None)
    path_dataset_anomaly_csv = dataset_cfg.get("path_dataset_anomaly_csv", None)

    saving_dir_experiments = model_cfg['saving_dir_experiments']
    saving_dir_model = model_cfg['saving_dir_model']
    num_epoch = model_cfg['num_epoch']
    epoch_start_unfreeze = model_cfg.get("epoch_start_unfreeze", None)
    layer_start_unfreeze = model_cfg.get("layer_start_unfreeze", None)
    batch_size = dataset_cfg['batch_size']
    scheduler_type = model_cfg['scheduler_type']
    learning_rate = model_cfg['learning_rate']
    lr_patience = model_cfg.get("lr_patience", None)
    scheduler_step_size = int(model_cfg.get("scheduler_step_size", None))
    lr_factor = model_cfg.get("lr_factor", None)
    T_max = model_cfg.get("T_max", None)
    eta_min = model_cfg.get("eta_min", None)
    number_of_classes = model_cfg['number_of_classes']
    is_slowfast = False
    if model_cfg['name_time_model'] == "3d_slowfast":
        is_slowfast = True
        print("Set is_slowfast to True")

    # load ans shuffle csv dataset
    df_dataset_train = pd.read_csv(path_dataset_train_csv)
    df_dataset_train = df_dataset_train.sample(frac=1).reset_index(drop=True)

    df_dataset_val = pd.read_csv(path_dataset_val_csv)
    df_dataset_val = df_dataset_val.sample(frac=1).reset_index(drop=True)

    df_dataset_test = None
    if path_dataset_test_csv is not None:
        df_dataset_test = pd.read_csv(path_dataset_test_csv)
        df_dataset_test = df_dataset_test.sample(frac=1).reset_index(drop=True)

    df_dataset_anomaly = None
    if path_dataset_anomaly_csv is not None:
        df_dataset_anomaly = pd.read_csv(path_dataset_anomaly_csv)
        df_dataset_anomaly = df_dataset_anomaly.sample(frac=1).reset_index(drop=True)

    # create the directories with the structure required by the project
    print("create the project structure")
    print("saving_dir_experiments: {}".format(saving_dir_experiments))
    saving_dir_model = os.path.join(saving_dir_experiments, saving_dir_model)
    print("saving_dir_model: {}".format(saving_dir_model))
    os.makedirs(saving_dir_experiments, exist_ok=True)
    os.makedirs(saving_dir_model, exist_ok=True)

    # save the config file
    yaml_config_path = os.path.join(saving_dir_model, "config.yaml")
    with open(yaml_config_path, 'w') as file:
        documents = yaml.dump(cfg, file)

    # create the dataloaders
    train_loader, val_loader, test_loader, anomaly_loader = create_loaders(df_dataset_train=df_dataset_train,
                                                                           df_dataset_val=df_dataset_val,
                                                                           df_dataset_test=df_dataset_test,
                                                                           df_dataset_anomaly=df_dataset_anomaly,
                                                                           data_cfg=cfg["data"],
                                                                           dataset_path=dataset_path,
                                                                           batch_size=batch_size,
                                                                           is_slowfast=is_slowfast)

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # create the model
    print("*****************************************")
    print("create the model")
    model = SpaceTimeAutoencoder(cfg["model"]).to(device)
    print("Check layers properties")
    for i, properties in enumerate(model.named_parameters()):
        print("Model layer: {} -  name: {} - requires_grad: {} ".format(i, properties[0],
                                                                        properties[1].requires_grad))
    print("*****************************************")
    checkpoint_dir = saving_dir_model

    if do_train:
        # look if exist a checkpoint
        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir)
        if path_last_checkpoint is not None:
            print("Load checkpoint from path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)

        # Set optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # set the scheduler
        scheduler = None
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                          mode='max',
                                          patience=lr_patience,
                                          verbose=True,
                                          factor=lr_factor)
        elif scheduler_type == "StepLR":
            print("StepLR")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                        step_size=scheduler_step_size,
                                                        gamma=lr_factor)
        elif scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                   T_max=T_max,
                                                                   eta_min=eta_min)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                             T_0=T_max,
                                                                             T_mult=1,
                                                                             eta_min=eta_min)

        # set the loss
        #criterion = nn.MSELoss()
        criterion = nn.MSELoss(reduction='sum')

        if checkpoint is not None:
            print('Load the optimizer from the last checkpoint')
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint["scheduler"])

            print('Latest epoch of the checkpoint: {}'.format(checkpoint['epoch']))
            print('Setting the new starting epoch: {}'.format(checkpoint['epoch'] + 1))
            best_epoch = checkpoint['epoch'] + 1
            print('Setting best best_eval_loss from checkpoint: {}'.format(checkpoint['best_eval_loss']))
            best_val_epoch_loss = checkpoint['best_eval_loss']

        # run train model function
        train_model(cfg=cfg,
                    device=device,
                    model=model,
                    criterion=criterion,
                    optimizer=optimizer,
                    lr_scheduler=scheduler,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    best_epoch=best_epoch,
                    num_epoch=num_epoch,
                    best_val_epoch_loss=best_val_epoch_loss,
                    checkpoint_dir=checkpoint_dir,
                    saving_dir_experiments=saving_dir_experiments,
                    epoch_start_unfreeze=epoch_start_unfreeze,
                    layer_start_unfreeze=layer_start_unfreeze,
                    scheduler_type=scheduler_type,
                    aws_bucket=aws_bucket,
                    aws_directory=aws_directory)
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

    if do_test:

        print("Execute Inference on Train, Val and Test Dataset with best checkpoint")

        path_last_checkpoint = find_last_checkpoint_file(checkpoint_dir=checkpoint_dir, use_best_checkpoint=True)
        if path_last_checkpoint is not None:
            print("Upload the best checkpoint at the path: ", path_last_checkpoint)
            checkpoint = torch.load(path_last_checkpoint, map_location=torch.device(device))
            model.load_state_dict(checkpoint['model'])
            model = model.to(device)

        # go through the lines of the dataset
        class2label = {}
        for index, row in df_dataset_train.iterrows():
            class_name = row["CLASS"]
            label = row["LABEL"]

            if class_name not in class2label:
                class2label[class_name] = label
        #sort the value of the label
        class2label = dict(sorted(class2label.items(), key=lambda item: item[1]))
        label2class = {k: v for (v, k) in class2label.items()}
        print("class2label: ", class2label)
        print("label2class: ", label2class)

        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        # create dataset for distribution
        df_distribution = pd.DataFrame(columns=['CLASS', 'LABEL', 'ERROR', 'LATENT_DISTANCE', 'EMBEDDING_DISTANCE', 'TYPE_DATASET'])
        df_anomaly_distances = pd.DataFrame(columns=['ANOMALY_CLASS', 'CLASS_CENTROID', 'DIST_CENTROID'])

        # 12 - execute the inferences on the train, val and test set
        print("Inference on train dataset")
        df_distribution, train_latent_centroids, train_embedding_centroids, _ = calculate_errors_and_distributions(device,
                                                                                                                model,
                                                                                                                train_loader,
                                                                                                                label2class,
                                                                                                                number_of_classes=number_of_classes,
                                                                                                                cfg=cfg,
                                                                                                                df_distribution=df_distribution,
                                                                                                                type_dataset="TRAIN",
                                                                                                                path_save=checkpoint_dir,
                                                                                                                embedding_centroids=None,
                                                                                                                latent_centroids=None)
        torch.cuda.empty_cache()
        gc.collect()
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        print("Inference on val dataset")
        df_distribution, _, _, _ = calculate_errors_and_distributions(device,
                                                                   model,
                                                                   val_loader,
                                                                   label2class,
                                                                   number_of_classes=number_of_classes,
                                                                   cfg=cfg,
                                                                   df_distribution=df_distribution,
                                                                   type_dataset="VAL",
                                                                   path_save=checkpoint_dir,
                                                                   embedding_centroids=train_embedding_centroids,
                                                                   latent_centroids=train_latent_centroids)
        torch.cuda.empty_cache()
        gc.collect()
        if test_loader is not None:
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")

            print("Inference on test dataset")
            df_distribution, _, _, _ = calculate_errors_and_distributions(device,
                                                                       model,
                                                                       test_loader,
                                                                       label2class,
                                                                       number_of_classes=number_of_classes,
                                                                       cfg=cfg,
                                                                       df_distribution=df_distribution,
                                                                       type_dataset="TEST",
                                                                       path_save=checkpoint_dir,
                                                                       embedding_centroids=train_embedding_centroids,
                                                                       latent_centroids=train_latent_centroids)
            torch.cuda.empty_cache()
            gc.collect()
        if anomaly_loader is not None:
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")

            print("Inference on anomaly dataset")
            df_distribution, _, _, df_anomaly_distances = calculate_errors_and_distributions(device,
                                                                                             model,
                                                                                             anomaly_loader,
                                                                                             label2class,
                                                                                             number_of_classes=number_of_classes,
                                                                                             cfg=cfg,
                                                                                             df_distribution=df_distribution,
                                                                                             type_dataset="ANOMALY",
                                                                                             path_save=checkpoint_dir,
                                                                                             embedding_centroids=train_embedding_centroids,
                                                                                             latent_centroids=train_latent_centroids,
                                                                                             df_anomaly_distances=df_anomaly_distances)
            torch.cuda.empty_cache()
            gc.collect()
        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")
        analyze_error_distribution(df_distribution, checkpoint_dir)
        analyze_embedding_distances(df_distribution, df_anomaly_distances, checkpoint_dir)

        print("Save the errors and dist distribution dataset at: ", os.path.join(checkpoint_dir, "errors_dist_distribution.csv"))
        df_distribution.to_csv(os.path.join(checkpoint_dir, "errors_dist_distribution.csv"), index=False)
        print("Save the anomaly distance distribution dataset at: ", os.path.join(checkpoint_dir, "anomaly_dist_distribution.csv"))
        df_anomaly_distances.to_csv(os.path.join(checkpoint_dir, "anomaly_dist_distribution.csv"), index=False)

        if aws_bucket is not None and aws_directory is not None:
            print("Final upload on S3")
            multiup(aws_bucket, aws_directory, saving_dir_experiments)

        print("End test")

    wandb.finish()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True