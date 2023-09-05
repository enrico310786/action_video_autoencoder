import torch
import os

from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

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

# for slow_fast check
# https://pytorchvideo.org/docs/tutorial_torchhub_inference
alpha = 4


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df_dataset, data_cfg, dataset_path, is_train=False, is_slowfast=False) -> None:
        super().__init__()

        self.df_dataset = df_dataset
        self.dataset_path = dataset_path
        self.data_cfg = data_cfg
        self.num_frames_to_sample = self.data_cfg["num_frames_to_sample"]
        self.mean = [float(i) for i in self.data_cfg["mean"]]
        self.std = [float(i) for i in self.data_cfg["std"]]
        self.min_size = self.data_cfg["min_size"]
        self.max_size = self.data_cfg["max_size"]
        self.resize_to = self.data_cfg["resize_to"]
        self.permute_color_frame = self.data_cfg.get("permute_color_frame", 1.0) > 0.0
        self.is_train = is_train
        self.is_slowfast = is_slowfast

        if not self.is_slowfast:
            if is_train:
                self.transform = ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            RandomShortSideScale(min_size=self.min_size, max_size=self.max_size),
                            RandomCrop(self.resize_to),
                            #RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                )
            else:
                self.transform = ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            Resize((self.resize_to, self.resize_to))
                        ]
                    ),
                )
        else:
            if is_train:
                self.transform = ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            RandomShortSideScale(min_size=self.min_size, max_size=self.max_size),
                            RandomCrop(self.resize_to),
                            PackPathway()
                        ]
                    ),
                )
            else:
                self.transform = ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            Resize((self.resize_to, self.resize_to)),
                            PackPathway()
                        ]
                    ),
                )

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dataset_path, self.df_dataset.iloc[idx]["PATH"])
        label = self.df_dataset.iloc[idx]["LABEL"]
        class_name = self.df_dataset.iloc[idx]["CLASS"]

        video = EncodedVideo.from_path(video_path)
        start_time = 0
        # follow this post for clip duration https://towardsdatascience.com/using-pytorchvideo-for-efficient-video-understanding-24d3cd99bc3c
        clip_duration = int(video.duration)
        end_sec = start_time + clip_duration
        video_data = video.get_clip(start_sec=start_time, end_sec=end_sec)
        video_data = self.transform(video_data)
        video_tensor = video_data["video"]
        if self.permute_color_frame:
            video_tensor = torch.permute(video_tensor, (1, 0, 2, 3))

        return video_tensor, label, class_name


def create_loaders(df_dataset_train, df_dataset_val, df_dataset_test, df_dataset_anomaly, data_cfg, dataset_path, batch_size, is_slowfast=False):

    # 1 - istanzio la classe dataset di train, val e test
    classification_dataset_train = ClassificationDataset(df_dataset=df_dataset_train,
                                                         data_cfg=data_cfg,
                                                         dataset_path=dataset_path,
                                                         is_train=True,
                                                         is_slowfast=is_slowfast)
    classification_dataset_val = ClassificationDataset(df_dataset=df_dataset_val,
                                                       data_cfg=data_cfg,
                                                       dataset_path=dataset_path,
                                                       is_slowfast=is_slowfast)
    classification_dataset_test = None
    if df_dataset_test is not None:
        classification_dataset_test = ClassificationDataset(df_dataset=df_dataset_test,
                                                            data_cfg=data_cfg,
                                                            dataset_path=dataset_path,
                                                            is_slowfast=is_slowfast)

    classification_dataset_anomaly = None
    if df_dataset_test is not None:
        classification_dataset_anomaly = ClassificationDataset(df_dataset=df_dataset_anomaly,
                                                               data_cfg=data_cfg,
                                                               dataset_path=dataset_path,
                                                               is_slowfast=is_slowfast)


    # 2 - istanzio i dataloader
    classification_dataloader_train = DataLoader(dataset=classification_dataset_train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 drop_last=False)

    classification_dataloader_val = DataLoader(dataset=classification_dataset_val,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               drop_last=False)

    classification_dataloader_test = None
    if classification_dataset_test is not None:
        classification_dataloader_test = DataLoader(dataset=classification_dataset_test,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0,
                                                    drop_last=False)

    classification_dataloader_anomaly = None
    if classification_dataset_anomaly is not None:
        classification_dataloader_anomaly = DataLoader(dataset=classification_dataset_anomaly,
                                                       batch_size=batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       drop_last=False)

    return classification_dataloader_train, classification_dataloader_val, classification_dataloader_test, classification_dataloader_anomaly