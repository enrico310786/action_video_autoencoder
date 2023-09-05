import os
import torch
import yaml
import argparse
import torch.nn as nn
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
import pandas as pd

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    Resize
)

from model import TimeAutoencoder, find_last_checkpoint_file

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha_slowfast):
        super().__init__()
        self.alpha_slowfast = alpha_slowfast

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha_slowfast
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def load_video(video_path, permute_color_frame, transform):

    video = EncodedVideo.from_path(video_path)
    start_time = 0
    # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
    clip_duration = int(video.duration)
    end_sec = start_time + clip_duration
    video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
    video_data = transform(video_data)
    video_tensor = video_data["video"]
    if permute_color_frame:
        video_tensor = torch.permute(video_tensor, (1, 0, 2, 3))

    return video_tensor


def evaluate_anomaly_accuracy_v2(model, transform, path_datset, thershold_err, thershold_dist, file_result, embedding_centroids=None, invert_accuracy=False, is_slowfast=False):

    # 4 iter over the anomaly clips
    counter_anomaly_error = 0
    counter_non_anomaly_error = 0
    counter_anomaly_dist = 0
    counter_non_anomaly_dist = 0

    with torch.no_grad():
        if embedding_centroids is None:
            embedding_centroids = []
            # this is the train dataset with wich i have to calcule the centroids
            for subdir, dirs, files in os.walk(path_datset):
                for dir in dirs:
                    print("Evaluate the centroid for class: ", dir)
                    path_subdir = os.path.join(path_datset, dir)
                    embeddings_array = None
                    for file in os.listdir(path_subdir):
                        video_path = os.path.join(path_subdir, file)
                        tensor_video = load_video(video_path, permute_color_frame, transform)

                        if is_slowfast:
                            tensor_video = [i.to(device)[None, ...] for i in tensor_video]
                        else:
                            tensor_video = tensor_video[None].to(device)

                        # calculate the embedding directly from the base model
                        emb = model.base_model(tensor_video)
                        if embeddings_array is None:
                            embeddings_array = emb.detach().cpu().numpy()
                        else:
                            embeddings_array = np.vstack((embeddings_array, emb.detach().cpu().numpy()))
                    print('embeddings_array.shape: ', embeddings_array.shape)
                    centroid = np.mean(embeddings_array, axis=0)
                    embedding_centroids.append(centroid)
            embedding_centroids = np.array(embedding_centroids)
            print("----------------------------------------------------------")

        print("embedding_centroids.shape: ", embedding_centroids.shape)

        for subdir, dirs, files in os.walk(path_datset):
            for dir in dirs:
                path_subdir = os.path.join(path_datset, dir)
                for file in os.listdir(path_subdir):
                    video_path = os.path.join(path_subdir, file)
                    tensor_video = load_video(video_path, permute_color_frame, transform)

                    if is_slowfast:
                        tensor_video = [i.to(device)[None, ...] for i in tensor_video]
                    else:
                        tensor_video = tensor_video[None].to(device)

                    # reconstructed embedding
                    emb, rec_emb, _ = model(tensor_video)
                    #rec_emb = model(tensor_video)
                    # embedding
                    #emb = model.base_model(tensor_video)
                    error = loss(emb, rec_emb).item()

                    # chech the reconstructing error
                    if error >= thershold_err:
                        counter_anomaly_error += 1
                    else:
                        counter_non_anomaly_error += 1

                    # check the distance from the centroid: if for all the centroids is greater then the thershold distance then is an anomaly
                    centroid_emb_distances = []
                    for idx in range(len(embedding_centroids)):
                        centroid_emb_distances.append(np.linalg.norm(emb.detach().cpu().numpy() - embedding_centroids[idx]))
                    if all(i >= thershold_dist for i in centroid_emb_distances):
                        counter_anomaly_dist += 1
                    else:
                        counter_non_anomaly_dist += 1

        file_result.write('\n')
        file_result.write('*************************************************************\n')
        file_result.write("RESULT ANOMALY WITH RECONSTRUCTION ERROR\n")
        file_result.write('*************************************************************\n')
        total_clip = counter_anomaly_error + counter_non_anomaly_error
        anomaly_accuracy = counter_anomaly_error/total_clip

        file_result.write("total_clip: {}\n".format(total_clip))
        file_result.write("counter_anomaly_error: {}\n".format(counter_anomaly_error))

        if invert_accuracy:
            file_result.write("non anomaly_accuracy: {}\n".format(1-anomaly_accuracy))
        else:
            file_result.write("anomaly_accuracy: {}\n".format(anomaly_accuracy))

        file_result.write('\n')

        file_result.write('*************************************************************\n')
        file_result.write("RESULT ANOMALY WITH EMBEDDING DISTANCE\n")
        file_result.write('*************************************************************\n')

        total_clip = counter_anomaly_dist + counter_non_anomaly_dist
        anomaly_accuracy = counter_anomaly_dist/total_clip

        file_result.write("total_clip: {}\n".format(total_clip))
        file_result.write("counter_anomaly_dist: {}\n".format(counter_anomaly_dist))

        if invert_accuracy:
            file_result.write("non anomaly_accuracy: {}\n".format(1-anomaly_accuracy))
        else:
            file_result.write("anomaly_accuracy: {}\n".format(anomaly_accuracy))

        return embedding_centroids, file_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    parser.add_argument('--thershold_error', type=str, help='Reconstructing error above the thershold gives an anomaly clip')
    parser.add_argument('--thershold_dist', type=str, help='Thershold distance between the array and one ov the train centroid')
    parser.add_argument('--name_output_file', type=str, help='Name of the output file with the results')
    opt = parser.parse_args()

    path_config_file = opt.path_config_file
    name_output_file = opt.name_output_file
    thershold_error = float(opt.thershold_error)
    thershold_dist = float(opt.thershold_dist)

    # 1 - load config file
    print('path_config_file: ', path_config_file)
    cfg = load_config(path_config_file)
    model_cfg = cfg['model']
    data_cfg = cfg['data']
    dataset_cfg = cfg['dataset']
    is_slowfast = False
    if model_cfg['name_time_model'] == "3d_slowfast":
        is_slowfast = True
        print("Set is_slowfast to True")

    num_frames_to_sample = data_cfg["num_frames_to_sample"]
    mean = [float(i) for i in data_cfg["mean"]]
    std = [float(i) for i in data_cfg["std"]]
    min_size = data_cfg["min_size"]
    max_size = data_cfg["max_size"]
    resize_to = data_cfg["resize_to"]
    permute_color_frame = data_cfg.get("permute_color_frame", 1.0) > 0.0
    alpha_slowfast = data_cfg.get("alpha_slowfast", None)

    path_dataset = dataset_cfg['dataset_path']
    path_model = os.path.join(model_cfg['saving_dir_experiments'], model_cfg['saving_dir_model'])
    path_checkpoint = os.path.join(path_model, 'best.pth')

    if not is_slowfast:
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((resize_to, resize_to))
                ]
            ),
        )
    else:
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(num_frames_to_sample),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize((resize_to, resize_to)),
                    PackPathway(alpha_slowfast)
                ]
            ),
        )

    # 2 - load model
    model = TimeAutoencoder(model_cfg)
    print("Upload the best checkpoint at the path: ", path_checkpoint)
    checkpoint = torch.load(path_checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model = model.eval()
    model = model.to(device)

    # 3 instantiate the loss function: MSE
    loss = nn.MSELoss(reduction='sum')

    file = open(os.path.join(path_model, name_output_file + ".txt"), 'w')

    with torch.no_grad():
        # 4 iter over the anomaly clips
        file.write('thershold_error: {}\n'.format(thershold_error))
        file.write('thershold_dist: {}\n'.format(thershold_dist))
        print("thershold_error: ", thershold_error)
        print("thershold_dist: ", thershold_dist)
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        file.write("NON ANOMALY ACCURACY - TRAIN SET\n")
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        path_train_dir = os.path.join(path_dataset, 'train')
        embedding_centroids, file = evaluate_anomaly_accuracy_v2(model=model,
                                                                 transform=transform,
                                                                 path_datset=path_train_dir,
                                                                 thershold_err=thershold_error,
                                                                 thershold_dist=thershold_dist,
                                                                 file_result=file,
                                                                 invert_accuracy=True,
                                                                 is_slowfast=is_slowfast)
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        file.write("NON ANOMALY ACCURACY - VAL SET\n")
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        path_val_dir = os.path.join(path_dataset, 'val')
        _, file = evaluate_anomaly_accuracy_v2(model=model,
                                               transform=transform,
                                               path_datset=path_val_dir,
                                               thershold_err=thershold_error,
                                               thershold_dist=thershold_dist,
                                               embedding_centroids=embedding_centroids,
                                               file_result=file,
                                               invert_accuracy=True,
                                               is_slowfast=is_slowfast)
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        file.write("NON ANOMALY ACCURACY - TEST SET\n")
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        path_test_dir = os.path.join(path_dataset, 'test')
        _, file = evaluate_anomaly_accuracy_v2(model=model,
                                               transform=transform,
                                               path_datset=path_test_dir,
                                               thershold_err=thershold_error,
                                               thershold_dist=thershold_dist,
                                               embedding_centroids=embedding_centroids,
                                               file_result=file,
                                               invert_accuracy=True,
                                               is_slowfast=is_slowfast)
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        file.write("ANOMALY ACCURACY\n")
        file.write("------------------------------------------------\n")
        file.write("------------------------------------------------\n")
        path_anomaly_dir = os.path.join(path_dataset, 'anomaly')
        _, file = evaluate_anomaly_accuracy_v2(model=model,
                                               transform=transform,
                                               path_datset=path_anomaly_dir,
                                               thershold_err=thershold_error,
                                               thershold_dist=thershold_dist,
                                               file_result=file,
                                               embedding_centroids=embedding_centroids,
                                               is_slowfast=is_slowfast)

        file.close()