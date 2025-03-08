import torch
import os
from torchvision.transforms import v2
from torch.utils.data import Dataset, Sampler
import pandas as pd
from PIL import Image
import random


class MiniImagenet(Dataset):
    def __init__(self, root: str, label: str) -> None:
        super().__init__()
        self.root = root
        self.data = pd.read_csv(label)
        self.class_names = self.data.label.unique().tolist()
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224)),
        ])
        self.labels = {class_name: id for id, class_name in enumerate(self.class_names)}
        self.class_indices = {}
        for idx, class_name in enumerate(self.data.label.tolist()):
            if class_name not in self.class_indices.keys():
                self.class_indices[class_name] = [idx]
            else:
                self.class_indices[class_name].append(idx)

    def __len__(self):
        # count = 0
        # for class_name in self.data:
        #     count += len(self.data[class_name])
        # return count
        return len(self.data)

    def __getitem__(self, idx):
        filename, class_name = self.data.filename[idx], self.data.label[idx]
        label = self.labels[class_name]
        img = self.transform(
            Image.open(os.path.join(self.root, filename)).convert("RGB")
        )
        return img, label


class Fewshot_sampler(Sampler):
    def __init__(self, n_ways: int, k_shot: int, q_query: int, dataset: MiniImagenet):
        super().__init__()
        self.n_ways = n_ways
        self.k_shot = k_shot
        self.q_query = q_query
        self.class_indices = dataset.class_indices
        # self.episodes = 5
        self.episodes = len(dataset)

    def __iter__(self):
        # selected_classes = random.sample(list(self.class_indices.keys()), self.n_ways)
        # support_set = []
        # query_set = []
        # for class_name in selected_classes:
        #     samples = random.sample(
        #         self.class_indices[class_name], self.k_shot + self.q_query
        #     )
        #     support = samples[: self.k_shot]
        #     query = samples[self.k_shot :]
        #     support_set.extend(support)
        #     query_set.extend(query)
        # yield support_set + query_set
        for _ in range(self.episodes):
            selected_classes = random.sample(
                list(self.class_indices.keys()), self.n_ways
            )
            support_set = []
            query_set = []
            for class_name in selected_classes:
                samples = random.sample(
                    self.class_indices[class_name], self.k_shot + self.q_query
                )
                support = samples[: self.k_shot]
                query = samples[self.k_shot :]
                support_set.extend(support)
                query_set.extend(query)
            yield support_set + query_set

    def __len__(self):
        return self.episodes
