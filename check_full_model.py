import argparse
import numpy as np
import av
import yaml
import torch

from prettytable import PrettyTable
from transformers import AutoImageProcessor
from model import TimeAutoencoder
import torch.nn as nn


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize
)

from pytorchvideo.data.encoded_video import EncodedVideo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


np.random.seed(0)



def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_config_file', type=str, help='Path of the config file to use')
    opt = parser.parse_args()

    # 2 - load config file
    path_config_file = opt.path_config_file
    print('path_config_file: ', path_config_file)
    cfg = load_config(path_config_file)
    model_cfg = cfg['model']

    # 3 - load model
    model = TimeAutoencoder(model_cfg)
    model.to(device)

    print("")
    print(model)
    print("")

    print("Check layers properties")
    for i, properties in enumerate(model.named_parameters()):
        print("Model layer: {} -  name: {} - requires_grad: {} ".format(i, properties[0], properties[1].requires_grad))
    print("")

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("pytorch_total_params: ", pytorch_total_params)
    print("pytorch_total_trainable_params: ", pytorch_total_trainable_params)

    print("")

    count_parameters(model)

    print("************************************************************")
    print("************************************************************")

    '''
    image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]

    resize_to = (height, width)
    print("resize_to: ", resize_to)
    num_frames_to_sample = model.base_model.base_model.config.num_frames
    print("num_frames_to_sample: ", num_frames_to_sample)

    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(num_frames_to_sample),
                Lambda(lambda x: x / 255.0),
                Normalize(mean, std),
                RandomShortSideScale(min_size=256, max_size=320),
                RandomCrop(resize_to),
            ]
        ),
    )

    file_path = "resources/6326-9_70170.avi"
    video = EncodedVideo.from_path(file_path)
    start_time = 0
    # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
    clip_duration = int(video.duration)
    # print("clip_duration: ", clip_duration)
    end_sec = start_time + clip_duration
    video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
    video_data = transform(video_data)
    inputs = video_data["video"]
    inputs = inputs[None].to(device)

    print("Before - inputs.size(): ", inputs.size())
    # The TimeSformer model has to permute the color channel with the num frame channel
    inputs = torch.permute(inputs, (0, 2, 1, 3, 4))
    print("After - inputs.size(): ", inputs.size())

    # forward pass
    outputs = model(inputs)
    print("outputs.size():", outputs.size())
    print("output: {}".format(outputs))

    target = model.base_model(inputs)
    print("target.size():", target.size())

    loss = nn.MSELoss()
    err = loss(outputs, target)
    print("err", err)

    latent = model.encoder(target)
    print("latent.size():", latent.size())

    '''