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


from model import TimeAutoencoder, find_last_checkpoint_file


def train_batch(inputs, model, optimizer, criterion):
    model.train()
    target = model.base_model(inputs)
    outputs = model(inputs)
    #print("target.size(): ", target.size())
    #print("outputs.size(): ", outputs.size())
    #print("type(target): ", type(target))
    #print("type(outputs): ", type(outputs))
    loss = criterion(outputs, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


@torch.no_grad()
def val_loss(inputs, model, criterion):
    model.eval()
    target = model.base_model(inputs)
    outputs = model(inputs)
    val_loss = criterion(outputs, target)
    return val_loss.item()


def find_errors(array, array_hat):
    # givent two array with same shape (n, m) i have to find the
    # mse between the first component of the first array and the corresponding fisrt component of the hat array -> thus use axis = 1
    errors = (np.square(array - array_hat)).mean(axis=1)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    return mean_error, max_error


def find_distances(latents, centroid):
    # find the mean and the max distance between latents array and the corresponding centroid
    list_dists = []
    for latent in latents:
        dist = np.linalg.norm(latent - centroid)
        list_dists.append(dist)
    list_dists = np.array(list_dists)
    mean_dist = np.mean(list_dists)
    max_dist = np.max(list_dists)
    return mean_dist, max_dist


def find_errors_and_latents_distribution(device,
                                         model,
                                         dataloader,
                                         number_of_classes,
                                         centroids=None,
                                         path_save=None,
                                         type_dataset=None):

    # centroids is the list of the centroids of each class calculated in the train set
    if centroids is None:
        centroids = []

    class_labels = []
    latents_array = None
    outputs_array = None
    target_array = None
    mean_errors = None
    max_errors = None
    mean_dists = None
    max_dists = None
    model = model.eval()

    with torch.no_grad():
        # cycle on all batches
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            targets = model.base_model(inputs)
            outputs = model(inputs)
            latents = model.encoder(targets)

            #print('latents.size(): ', latents.size())
            #print('outputs.size(): ', outputs.size())
            #print('targets.size(): ', targets.size())

            #print('latents.detach().cpu().numpy().shape: ', latents.detach().cpu().numpy().shape)
            #print('latents.detach().cpu().numpy().shape: ', latents.detach().cpu().numpy().shape)
            #print('latents.detach().cpu().numpy().shape: ', latents.detach().cpu().numpy().shape)

            # stack the results
            if latents_array is None:
                latents_array = latents.detach().cpu().numpy()
            else:
                latents_array = np.vstack((latents_array, latents.detach().cpu().numpy()))

            if target_array is None:
                target_array = targets.detach().cpu().numpy()
            else:
                target_array = np.vstack((target_array, targets.detach().cpu().numpy()))

            if outputs_array is None:
                outputs_array = outputs.detach().cpu().numpy()
            else:
                outputs_array = np.vstack((outputs_array, outputs.detach().cpu().numpy()))

            #latents_array.append(latents.detach().cpu().numpy())
            #target_array.append(targets.detach().cpu().numpy())
            #outputs_array.append(outputs.detach().cpu().numpy())
            class_labels.extend(labels.detach().cpu().numpy().tolist())

        # transform to numpy array
        class_labels = np.array(class_labels)
        #latents_array = np.array(latents_array)
        #outputs_array = np.array(outputs_array)
        #target_array = np.array(target_array)

        print('class_labels.shape: ', class_labels.shape)
        print('latents_array.shape: ', latents_array.shape)
        print('outputs_array.shape: ', outputs_array.shape)
        print('target_array.shape: ', target_array.shape)

        if len(centroids) == 0:
            # calculate the centroid of the latent vectors for each class. Iter over the label of each class
            for idx in range(number_of_classes):
                filter_idxs = np.where(class_labels == idx)[0]
                print("filter_idxs: ", filter_idxs)
                latents_array_filtered = np.take(latents_array, filter_idxs, 0)
                centroid = np.mean(latents_array_filtered, axis=0)
                centroids.append(centroid)
        centroids = np.array(centroids)

        for idx in range(number_of_classes):
            print("Examin class label: ", idx)

            filter_idxs = np.where(class_labels == idx)[0]
            latents_array_filtered = np.take(latents_array, filter_idxs, 0)
            outputs_array_filtered = np.take(outputs_array, filter_idxs, 0)
            target_array_filtered = np.take(target_array, filter_idxs, 0)
            centroid = centroids[idx]

            mean_error, max_error = find_errors(target_array_filtered, outputs_array_filtered)
            mean_dist, max_dist = find_distances(latents_array_filtered, centroid)

            # stack the results of errors and distances
            if mean_errors is None:
                mean_errors = mean_error
            else:
                mean_errors = np.vstack((mean_errors, mean_error))

            if max_errors is None:
                max_errors = max_error
            else:
                max_errors = np.vstack((max_errors, max_error))

            if mean_dists is None:
                mean_dists = mean_dist
            else:
                mean_dists = np.vstack((mean_dists, mean_dist))

            if max_dists is None:
                max_dists = max_dist
            else:
                max_dists = np.vstack((max_dists, max_dist))


            #mean_errors.append(mean_error)
            #max_errors.append(max_error)
            #mean_dists.append(mean_dist)
            #max_dists.append(max_dist)

        if path_save is not None and type_dataset is not None:
            f = open(os.path.join(path_save, type_dataset + "_report.txt"), 'w')
            f.write('Report dataset{}'.format(type_dataset))
            f.write('\n\n')
            f.write('Mean_error_0: {}\n'.format(mean_errors[0]))
            f.write('Mean_error_1: {}\n'.format(mean_errors[1]))
            f.write('Mean_error_2: {}\n'.format(mean_errors[2]))
            f.write('Mean_error_3: {}\n'.format(mean_errors[3]))
            f.write('Mean_error_4: {}\n'.format(mean_errors[4]))
            f.write('Mean_error_5: {}\n'.format(mean_errors[5]))
            f.write('Mean_error_6: {}\n'.format(mean_errors[6]))
            f.write('Mean_error_7: {}\n'.format(mean_errors[7]))
            f.write('Mean_error_8: {}\n'.format(mean_errors[8]))
            f.write('Mean_error_9: {}\n'.format(mean_errors[9]))
            f.write('\n\n')
            f.write('Max_error_0: {}\n'.format(max_errors[0]))
            f.write('Max_error_1: {}\n'.format(max_errors[1]))
            f.write('Max_error_2: {}\n'.format(max_errors[2]))
            f.write('Max_error_3: {}\n'.format(max_errors[3]))
            f.write('Max_error_4: {}\n'.format(max_errors[4]))
            f.write('Max_error_5: {}\n'.format(max_errors[5]))
            f.write('Max_error_6: {}\n'.format(max_errors[6]))
            f.write('Max_error_7: {}\n'.format(max_errors[7]))
            f.write('Max_error_8: {}\n'.format(max_errors[8]))
            f.write('Max_error_9: {}\n'.format(max_errors[9]))
            f.write('\n\n')
            f.write('Mean_dist_0: {}\n'.format(mean_dists[0]))
            f.write('Mean_dist_1: {}\n'.format(mean_dists[1]))
            f.write('Mean_dist_2: {}\n'.format(mean_dists[2]))
            f.write('Mean_dist_3: {}\n'.format(mean_dists[3]))
            f.write('Mean_dist_4: {}\n'.format(mean_dists[4]))
            f.write('Mean_dist_5: {}\n'.format(mean_dists[5]))
            f.write('Mean_dist_6: {}\n'.format(mean_dists[6]))
            f.write('Mean_dist_7: {}\n'.format(mean_dists[7]))
            f.write('Mean_dist_8: {}\n'.format(mean_dists[8]))
            f.write('Mean_dist_9: {}\n'.format(mean_dists[9]))
            f.write('\n\n')
            f.write('Max_dist_0: {}\n'.format(max_dists[0]))
            f.write('Max_dist_1: {}\n'.format(max_dists[1]))
            f.write('Max_dist_2: {}\n'.format(max_dists[2]))
            f.write('Max_dist_3: {}\n'.format(max_dists[3]))
            f.write('Max_dist_4: {}\n'.format(max_dists[4]))
            f.write('Max_dist_5: {}\n'.format(max_dists[5]))
            f.write('Max_dist_6: {}\n'.format(max_dists[6]))
            f.write('Max_dist_7: {}\n'.format(max_dists[7]))
            f.write('Max_dist_8: {}\n'.format(max_dists[8]))
            f.write('Max_dist_9: {}\n'.format(max_dists[9]))

            if type_dataset == "TRAIN":
                f.write('\n\n')
                f.write('centroid_0: {}\n\n'.format(centroids[0]))
                f.write('centroid_1: {}\n\n'.format(centroids[1]))
                f.write('centroid_2: {}\n\n'.format(centroids[2]))
                f.write('centroid_3: {}\n\n'.format(centroids[3]))
                f.write('centroid_4: {}\n\n'.format(centroids[4]))
                f.write('centroid_5: {}\n\n'.format(centroids[5]))
                f.write('centroid_6: {}\n\n'.format(centroids[6]))
                f.write('centroid_7: {}\n\n'.format(centroids[7]))
                f.write('centroid_8: {}\n\n'.format(centroids[8]))
                f.write('centroid_9: {}\n\n'.format(centroids[9]))

            f.close()

        if path_save is not None and type_dataset == "TRAIN":
            tsne = TSNE(2)
            clustered = tsne.fit_transform(latents_array)

            fig = plt.figure(figsize=(12, 10))
            cmap = plt.get_cmap('Spectral', 10)
            plt.scatter(*zip(*clustered), c=class_labels, cmap=cmap)
            plt.colorbar(drawedges=True)
            fig.savefig(os.path.join(path_save, "TSNE_latent_array_train.png"))

        return mean_errors, max_errors, mean_dists, max_dists, centroids


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
    number_of_classes = cfg["data"]["number_of_classes"]

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

        # define empty lists for the values of the loss and the accuracy of train and validation obtained in the batch of the current epoch
        # then at the end I take the average and I get the final values of the whole era
        train_epoch_losses = []
        val_epoch_losses = []

        # cycle on all train batches of the current epoch by executing the train_batch function
        for inputs, _ in tqdm(train_loader, desc=f"epoch {str(epoch)} | train"):
            inputs = inputs.to(device)
            batch_loss = train_batch(inputs, model, optimizer, criterion)
            train_epoch_losses.append(batch_loss)
        train_epoch_loss = np.array(train_epoch_losses).mean()

        # cycle on all batches of val of the current epoch by calculating the accuracy and the loss function
        for inputs, _ in tqdm(val_loader, desc=f"epoch {str(epoch)} | val"):
            inputs = inputs.to(device)
            validation_loss = val_loss(inputs, model, criterion)
            val_epoch_losses.append(validation_loss)
        val_epoch_loss = np.mean(val_epoch_losses)

        # calculate the errors and distances distribution on the train set
        mean_errors_train, max_errors_train, mean_dists_train, max_dists_train, centroids_train = find_errors_and_latents_distribution(device=device,
                                                                                                                                       model=model,
                                                                                                                                       dataloader=train_loader,
                                                                                                                                       number_of_classes=number_of_classes)

        # calculate the errors and distances distribution on the validation set
        mean_errors_val, max_errors_val, mean_dists_val, max_dists_val, _ = find_errors_and_latents_distribution(device=device,
                                                                                                                 model=model,
                                                                                                                 dataloader=val_loader,
                                                                                                                 number_of_classes=number_of_classes,
                                                                                                                 centroids=centroids_train)

        wandb.log({'Learning Rate': optimizer.param_groups[0]['lr'],
                   'Train Loss': train_epoch_loss,
                   'Valid Loss': val_epoch_loss,
                   'Mean_error_train_0': mean_errors_train[0],
                   'Mean_error_train_1': mean_errors_train[1],
                   'Mean_error_train_2': mean_errors_train[2],
                   'Mean_error_train_3': mean_errors_train[3],
                   'Mean_error_train_4': mean_errors_train[4],
                   'Mean_error_train_5': mean_errors_train[5],
                   'Mean_error_train_6': mean_errors_train[6],
                   'Mean_error_train_7': mean_errors_train[7],
                   'Mean_error_train_8': mean_errors_train[8],
                   'Mean_error_train_9': mean_errors_train[9],
                   'Mean_error_val_0': mean_errors_val[0],
                   'Mean_error_val_1': mean_errors_val[1],
                   'Mean_error_val_2': mean_errors_val[2],
                   'Mean_error_val_3': mean_errors_val[3],
                   'Mean_error_val_4': mean_errors_val[4],
                   'Mean_error_val_5': mean_errors_val[5],
                   'Mean_error_val_6': mean_errors_val[6],
                   'Mean_error_val_7': mean_errors_val[7],
                   'Mean_error_val_8': mean_errors_val[8],
                   'Mean_error_val_9': mean_errors_val[9],
                   'Mean_dist_train_0': mean_dists_train[0],
                   'Mean_dist_train_1': mean_dists_train[1],
                   'Mean_dist_train_2': mean_dists_train[2],
                   'Mean_dist_train_3': mean_dists_train[3],
                   'Mean_dist_train_4': mean_dists_train[4],
                   'Mean_dist_train_5': mean_dists_train[5],
                   'Mean_dist_train_6': mean_dists_train[6],
                   'Mean_dist_train_7': mean_dists_train[7],
                   'Mean_dist_train_8': mean_dists_train[8],
                   'Mean_dist_train_9': mean_dists_train[9],
                   'Mean_dist_val_0': mean_dists_val[0],
                   'Mean_dist_val_1': mean_dists_val[1],
                   'Mean_dist_val_2': mean_dists_val[2],
                   'Mean_dist_val_3': mean_dists_val[3],
                   'Mean_dist_val_4': mean_dists_val[4],
                   'Mean_dist_val_5': mean_dists_val[5],
                   'Mean_dist_val_6': mean_dists_val[6],
                   'Mean_dist_val_7': mean_dists_val[7],
                   'Mean_dist_val_8': mean_dists_val[8],
                   'Mean_dist_val_9': mean_dists_val[9]})
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

    # load ans shuffle csv dataset
    df_dataset_train = pd.read_csv(path_dataset_train_csv)
    df_dataset_train = df_dataset_train.sample(frac=1).reset_index(drop=True)

    df_dataset_val = pd.read_csv(path_dataset_val_csv)
    df_dataset_val = df_dataset_val.sample(frac=1).reset_index(drop=True)

    df_dataset_test = None
    if path_dataset_test_csv is not None:
        df_dataset_test = pd.read_csv(path_dataset_test_csv)
        df_dataset_test = df_dataset_test.sample(frac=1).reset_index(drop=True)

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
    train_loader, val_loader, test_loader = create_loaders(df_dataset_train=df_dataset_train,
                                                           df_dataset_val=df_dataset_val,
                                                           df_dataset_test=df_dataset_test,
                                                           data_cfg=cfg["data"],
                                                           dataset_path=dataset_path,
                                                           batch_size=batch_size)

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: {}".format(device))

    # create the model
    print("*****************************************")
    print("create the model")
    model = TimeAutoencoder(cfg["model"])
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
        criterion = nn.MSELoss()

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

        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        # 12 - execute the inferences on the train, val and test set
        print("Inference on train dataset")

        _, _, _, _, centroids_train = find_errors_and_latents_distribution(device=device,
                                                                           model=model,
                                                                           dataloader=train_loader,
                                                                           number_of_classes=cfg['data']['number_of_classes'],
                                                                           path_save=checkpoint_dir,
                                                                           type_dataset="TRAIN")

        print("-------------------------------------------------------------------")
        print("-------------------------------------------------------------------")

        print("Inference on val dataset")

        # calculate the errors and distances distribution on the validation set
        _, _, _, _, _ = find_errors_and_latents_distribution(device=device,
                                                             model=model,
                                                             dataloader=val_loader,
                                                             number_of_classes=cfg['data']['number_of_classes'],
                                                             centroids=centroids_train,
                                                             path_save=checkpoint_dir,
                                                             type_dataset="VAL")

        if test_loader is not None:
            print("-------------------------------------------------------------------")
            print("-------------------------------------------------------------------")

            print("Inference on test dataset")
            _, _, _, _, _ = find_errors_and_latents_distribution(device=device,
                                                                 model=model,
                                                                 dataloader=test_loader,
                                                                 number_of_classes=cfg['data']['number_of_classes'],
                                                                 centroids=centroids_train,
                                                                 path_save=checkpoint_dir,
                                                                 type_dataset="TEST")

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