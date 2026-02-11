import os
import random
import os.path as osp
import tarfile
import zipfile
from collections import defaultdict
import gdown
import json
import torch
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np


def read_json(fpath):
    """Read json file from a path."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Writes to a json file."""
    if not osp.exists(osp.dirname(fpath)):
        os.makedirs(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_image(path):
    """Read image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    if not osp.exists(path):
        raise IOError('No file exists at {}'.format(path))

    while True:
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print(
                'Cannot read image from {}, '
                'probably due to heavy IO. Will re-try'.format(path)
            )


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith('.') and 'sh' not in f]
    if sort:
        items.sort()
    return items


class Datum:
    """Data instance which defines the basic attributes.

    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath='', label=0, domain=-1, classname=''):
        assert isinstance(impath, str)
        assert isinstance(label, int)
        assert isinstance(domain, int)
        assert isinstance(classname, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    """A unified dataset class for
    1) domain adaptation
    2) domain generalization
    3) semi-supervised learning
    """
    dataset_dir = '' # the directory where the dataset is stored
    domains = [] # string names of all domains

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x # labeled training data
        self._train_u = train_u # unlabeled training data (optional)
        self._val = val # validation data (optional)
        self._test = test # test data

        self._num_classes = self.get_num_classes(train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    def get_num_classes(self, data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return max(label_set) + 1

    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def check_input_domains(self, source_domains, target_domains):
        self.is_input_domain_valid(source_domains)
        self.is_input_domain_valid(target_domains)

    def is_input_domain_valid(self, input_domains):
        for domain in input_domains:
            if domain not in self.domains:
                raise ValueError(
                    'Input domain must belong to {}, '
                    'but got [{}]'.format(self.domains, domain)
                )

    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        if from_gdrive:
            gdown.download(url, dst, quiet=False)
        else:
            raise NotImplementedError

        print('Extracting file ...')

        try:
            tar = tarfile.open(dst)
            tar.extractall(path=osp.dirname(dst))
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(dst, 'r')
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        print('File extracted to {}'.format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output


class DatasetWrapper(TorchDataset):
    def __init__(self, data_source, input_size, transform=None, is_train=False,
                 return_img0=False, k_tfm=1):
        self.data_source = data_source
        self.transform = transform # accept list (tuple) as input
        self.is_train = is_train
        # Augmenting an image K>1 times is only allowed during training
        self.k_tfm = k_tfm if is_train else 1
        self.return_img0 = return_img0

        if self.k_tfm > 1 and transform is None:
            raise ValueError(
                'Cannot augment the image {} times '
                'because transform is None'.format(self.k_tfm)
            )

        # Build transform that doesn't apply any data augmentation
        interp_mode = T.InterpolationMode.BICUBIC
        to_tensor = []
        to_tensor += [T.Resize(input_size, interpolation=interp_mode)]
        to_tensor += [T.ToTensor()]
        normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        )
        to_tensor += [normalize]
        self.to_tensor = T.Compose(to_tensor)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        output = {
            'label': item.label,
            'domain': item.domain,
            'impath': item.impath
        }

        img0 = read_image(item.impath)

        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                for i, tfm in enumerate(self.transform):
                    img = self._transform_image(tfm, img0)
                    keyname = 'img'
                    if (i + 1) > 1:
                        keyname += str(i + 1)
                    output[keyname] = img
            else:
                img = self._transform_image(self.transform, img0)
                output['img'] = img

        if self.return_img0:
            output['img0'] = self.to_tensor(img0)

        return output['img'], output['label']

    def _transform_image(self, tfm, img0):
        img_list = []

        for k in range(self.k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img


def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, input_size=input_size, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=8,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available())
    )
    assert len(data_loader) > 0

    return data_loader


import torch.nn.functional as F
# @torch.no_grad()
# def select_shots_herding(data_source, model, transform, num_shots, device='cuda', batch_size=64):
#     """
#     Selects the N best samples per class.
#     - If num_shots == 1: Selects the sample closest to the global class mean.
#     - If num_shots > 1: Uses Spherical K-Means to select N diverse prototypes.
#     """
#     print(f"\n[Herding] Selecting best {num_shots} shots from {len(data_source)} total samples...")
#     model.eval()
#     model.to(device)
    
#     # 1. Create loader
#     temp_loader = torch.utils.data.DataLoader(
#         DatasetWrapper(data_source, 224, transform=transform, is_train=False),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         drop_last=False
#     )
    
#     all_features = []
#     all_labels = []
    
#     # 2. Extract features
#     print("[Herding] Extracting features...")
#     for images, labels in temp_loader:
#         images = images.to(device)
#         with torch.cuda.amp.autocast():
#             output = model(images)
            
#         # CLS token + Normalize
#         global_features = output #[:, 0, :]
#         global_features = F.normalize(global_features, dim=-1)
        
#         all_features.append(global_features.cpu())
#         all_labels.append(labels)
        
#     all_features = torch.cat(all_features, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    
#     selected_indices = []
#     unique_classes = torch.unique(all_labels).sort().values
    
#     print(f"[Herding] Running selection (Mode: {'K-Means' if num_shots > 2 else 'Mean-Centroid'})...")
    
#     for cls_idx in unique_classes:
#         # Get indices/features for this class
#         cls_mask = (all_labels == cls_idx)
#         global_indices = torch.nonzero(cls_mask).squeeze(1)
#         cls_feats = all_features[cls_mask].to(device) # Move to GPU for K-Means
        
#         num_samples = len(global_indices)
#         k = min(num_shots, num_samples)
        
#         if k == num_samples:
#             # Take all if not enough samples
#             selected_indices.extend(global_indices.tolist())
#             continue

#         # --- STRATEGY SWITCH ---
        
#         if k <= 2:
#             # Strategy: Pick k samples closest to the Global Class Mean
#             prototype = cls_feats.mean(dim=0, keepdim=True)
#             prototype = F.normalize(prototype, dim=-1)
            
#             # Cosine similarity
#             sims = torch.mm(cls_feats, prototype.t()).squeeze()
            
#             # FIX: Use topk instead of argmax to handle k=2
#             _, best_rel_indices = torch.topk(sims, k)
            
#             # Add all selected indices
#             for rel_idx in best_rel_indices:
#                 selected_indices.append(global_indices[rel_idx].item())
            
#         else:
#             # >1 Shot: Spherical K-Means for Diversity
#             # We want k clusters.
            
#             # 1. Initialize Centroids (Randomly pick k samples)
#             # We use a fixed seed per class for reproducibility within the run
#             g_cpu = torch.Generator()
#             g_cpu.manual_seed(int(cls_idx)) 
#             init_perm = torch.randperm(num_samples, generator=g_cpu)[:k]
#             centroids = cls_feats[init_perm].clone()
            
#             # 2. Run K-Means (Simple PyTorch implementation)
#             # Max iterations 20 is usually plenty for feature convergence
#             for _ in range(20):
#                 # Assign points to closest centroid (Cosine Similarity)
#                 # sims: [N_samples, K_clusters]
#                 sims = torch.mm(cls_feats, centroids.t()) 
#                 labels = torch.argmax(sims, dim=1)
                
#                 # Update centroids
#                 new_centroids = []
#                 for i in range(k):
#                     mask = (labels == i)
#                     if mask.sum() > 0:
#                         cluster_mean = cls_feats[mask].mean(dim=0)
#                         new_centroids.append(cluster_mean)
#                     else:
#                         # Handle empty cluster (rare): keep old centroid
#                         new_centroids.append(centroids[i])
                
#                 centroids = torch.stack(new_centroids)
#                 centroids = F.normalize(centroids, dim=-1) # Spherical constraint
            
#             # 3. Select Representative for each Cluster
#             # For each centroid, find the actual sample closest to it
#             sims = torch.mm(cls_feats, centroids.t()) # [N, K]
            
#             for i in range(k):
#                 # We only consider points assigned to this cluster to ensure partition
#                 # (Optional: can also just pick globally closest, but this is safer)
#                 mask = (labels == i)
                
#                 if mask.sum() > 0:
#                     # Find max sim within this cluster
#                     cluster_sims = sims[:, i].clone()
#                     cluster_sims[~mask] = -1.0 # Ignore other points
#                     best_rel_idx = torch.argmax(cluster_sims).item()
#                 else:
#                     # Fallback if cluster empty: just pick globally closest
#                     best_rel_idx = torch.argmax(sims[:, i]).item()
                
#                 selected_indices.append(global_indices[best_rel_idx].item())

#     # Filter and Return
#     # Sort indices to maintain some order (optional)
#     selected_indices.sort()
#     herded_data_source = [data_source[i] for i in selected_indices]
    
#     print(f"[Herding] Selection complete. Reduced from {len(data_source)} to {len(herded_data_source)} samples.")
#     return herded_data_source



# @torch.no_grad()
# def select_shots_herding(data_source, input_size, model, transform, num_shots, device='cuda', batch_size=8):
#     """
#     Selects the N best samples per class.
#     - If num_shots == 1: Selects the sample closest to the global class mean.
#     - If num_shots > 1: Uses Spherical K-Means to select N diverse prototypes.
#     """
#     print(f"\n[Herding] Selecting best {num_shots} shots from {len(data_source)} total samples...")
#     model.eval()
#     model.to(device)
    
#     # 1. Create loader
#     temp_loader = torch.utils.data.DataLoader(
#         DatasetWrapper(data_source, input_size, transform=transform, is_train=False),
#         batch_size=8,
#         shuffle=False,
#         num_workers=4,
#         drop_last=False
#     )
    
#     all_features = []
#     all_labels = []
    
#     # 2. Extract features
#     print("[Herding] Extracting features...")
#     for images, labels in temp_loader:
#         images = images.to(device)
#         with torch.cuda.amp.autocast():
#             output = model.encode_image(images)
#         # CLS token + Normalize
#         global_features = output #[:, 0, :]
#         global_features = F.normalize(global_features, dim=-1)
        
#         all_features.append(global_features.cpu())
#         all_labels.append(labels)
        
#     all_features = torch.cat(all_features, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    
#     selected_indices = []
#     unique_classes = torch.unique(all_labels).sort().values
    
#     print(f"[Herding] Running selection (Mode: {'K-Means' if num_shots > 2 else 'Mean-Centroid'})...")
    
#     for cls_idx in unique_classes:
#         # Get indices/features for this class
#         cls_mask = (all_labels == cls_idx)
#         global_indices = torch.nonzero(cls_mask).squeeze(1)
#         cls_feats = all_features[cls_mask].to(device) # Move to GPU for K-Means
        
#         num_samples = len(global_indices)
#         k = min(num_shots, num_samples)
        
#         if k == num_samples:
#             # Take all if not enough samples
#             selected_indices.extend(global_indices.tolist())
#             continue

#         # --- STRATEGY SWITCH ---
        
#         if k <= 2:
#             # Strategy: Pick k samples closest to the Global Class Mean
#             prototype = cls_feats.mean(dim=0, keepdim=True)
#             prototype = F.normalize(prototype, dim=-1)
            
#             # Cosine similarity
#             sims = torch.mm(cls_feats, prototype.t()).squeeze()
            
#             # FIX: Use topk instead of argmax to handle k=2
#             _, best_rel_indices = torch.topk(sims, k)
            
#             # Add all selected indices
#             for rel_idx in best_rel_indices:
#                 selected_indices.append(global_indices[rel_idx].item())
            
#         else:
#             # >1 Shot: Spherical K-Means for Diversity
#             # We want k clusters.
            
#             # 1. Initialize Centroids (Randomly pick k samples)
#             # We use a fixed seed per class for reproducibility within the run
#             g_cpu = torch.Generator()
#             g_cpu.manual_seed(int(cls_idx)) 
#             init_perm = torch.randperm(num_samples, generator=g_cpu)[:k]
#             centroids = cls_feats[init_perm].clone()
            
#             # 2. Run K-Means (Simple PyTorch implementation)
#             # Max iterations 20 is usually plenty for feature convergence
#             for _ in range(20):
#                 # Assign points to closest centroid (Cosine Similarity)
#                 # sims: [N_samples, K_clusters]
#                 sims = torch.mm(cls_feats, centroids.t()) 
#                 labels = torch.argmax(sims, dim=1)
                
#                 # Update centroids
#                 new_centroids = []
#                 for i in range(k):
#                     mask = (labels == i)
#                     if mask.sum() > 0:
#                         cluster_mean = cls_feats[mask].mean(dim=0)
#                         new_centroids.append(cluster_mean)
#                     else:
#                         # Handle empty cluster (rare): keep old centroid
#                         new_centroids.append(centroids[i])
                
#                 centroids = torch.stack(new_centroids)
#                 centroids = F.normalize(centroids, dim=-1) # Spherical constraint
            
#             # 3. Select Representative for each Cluster
#             # For each centroid, find the actual sample closest to it
#             sims = torch.mm(cls_feats, centroids.t()) # [N, K]
            
#             for i in range(k):
#                 # We only consider points assigned to this cluster to ensure partition
#                 # (Optional: can also just pick globally closest, but this is safer)
#                 mask = (labels == i)
                
#                 if mask.sum() > 0:
#                     # Find max sim within this cluster
#                     cluster_sims = sims[:, i].clone()
#                     cluster_sims[~mask] = -1.0 # Ignore other points
#                     best_rel_idx = torch.argmax(cluster_sims).item()
#                 else:
#                     # Fallback if cluster empty: just pick globally closest
#                     best_rel_idx = torch.argmax(sims[:, i]).item()
                
#                 selected_indices.append(global_indices[best_rel_idx].item())

#     # Filter and Return
#     # Sort indices to maintain some order (optional)
#     selected_indices.sort()
#     herded_data_source = [data_source[i] for i in selected_indices]
    
#     print(f"[Herding] Selection complete. Reduced from {len(data_source)} to {len(herded_data_source)} samples.")
#     return herded_data_source





# @torch.no_grad()
# def select_shots_herding(data_source, input_size, model, transform, num_shots, device='cuda', batch_size=64):
#     """
#     Selects N shots per class using Herding (Nearest to Mean) or Spherical K-Means.
#     """
#     print(f"\n[Herding] Selecting best {num_shots} shots from {len(data_source)} total samples...")
    
#     model.eval()
#     model.to(device)
    
#     # 1. Feature Extraction
#     # We use a simple local wrapper to ensure we don't rely on the complex global DatasetWrapper
#     # which might have training-specific augmentations (k_tfm>1).
#     class FeatureDataset(torch.utils.data.Dataset):
#         def __init__(self, data, tfm):
#             self.data = data
#             self.tfm = tfm
#         def __len__(self): return len(self.data)
#         def __getitem__(self, idx):
#             item = self.data[idx]
#             from PIL import Image
#             img = Image.open(item.impath).convert('RGB')
#             if self.tfm: img = self.tfm(img)
#             return img, item.label

#     dataset = FeatureDataset(data_source, transform)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    
#     all_features = []
#     all_labels = []
    
#     print("[Herding] Extracting features...")
#     for images, labels in tqdm(loader):
#         images = images.to(device)
        
#         # Robust forward pass handling
#         if hasattr(model, 'attnpool'): # ResNet
#             x = model(images)
#             # Handle tuple return from our previous monkey patches
#             global_features = x[0] if isinstance(x, tuple) else x
#         else: # ViT
#             x = model(images)
#             global_features = x[0] if isinstance(x, tuple) else x
            
#         global_features = F.normalize(global_features, dim=-1)
#         all_features.append(global_features.cpu())
#         all_labels.append(labels)
        
#     all_features = torch.cat(all_features, dim=0)
#     all_labels = torch.cat(all_labels, dim=0)
    
#     selected_indices = []
#     unique_classes = torch.unique(all_labels).sort().values
    
#     print(f"[Herding] Running selection (Mode: {'K-Means' if num_shots > 1 else 'Mean-Centroid'})...")
    
#     for cls_idx in tqdm(unique_classes):
#         cls_mask = (all_labels == cls_idx)
#         # These are indices in the 'all_features' array, which map 1:1 to 'data_source'
#         global_indices = torch.nonzero(cls_mask).squeeze(1)
        
#         cls_feats = all_features[cls_mask].to(device)
#         num_samples = len(global_indices)
#         k = min(num_shots, num_samples)
        
#         if k == num_samples:
#             selected_indices.extend(global_indices.tolist())
#             continue

#         # Strategy A: 1-Shot (Mean Centroid)
#         if k == 1:
#             prototype = cls_feats.mean(dim=0, keepdim=True)
#             prototype = F.normalize(prototype, dim=-1)
#             # Find sample closest to mean
#             sims = torch.mm(cls_feats, prototype.t()).squeeze()
#             best_idx = torch.argmax(sims).item()
#             selected_indices.append(global_indices[best_idx].item())
            
#         # Strategy B: K-Means (Diversity)
#         else:
#             # 1. K-Means++ Initialization (Better than random)
#             centroids = torch.zeros(k, cls_feats.shape[1], device=device)
#             # Pick first centroid randomly
#             g_cpu = torch.Generator()
#             g_cpu.manual_seed(int(cls_idx))
#             first_idx = torch.randint(0, num_samples, (1,), generator=g_cpu).item()
#             centroids[0] = cls_feats[first_idx]
            
#             # Pick remaining centroids based on distance
#             if k > 1:
#                 # Compute distances from current centroids
#                 dists = torch.full((num_samples,), float('inf'), device=device)
#                 for i in range(1, k):
#                     # Distance is 1 - Cosine Similarity
#                     last_centroid = centroids[i-1].unsqueeze(0)
#                     # [N]
#                     sim_to_last = torch.mm(cls_feats, last_centroid.t()).squeeze()
#                     dist_to_last = 1 - sim_to_last
#                     dists = torch.minimum(dists, dist_to_last)
                    
#                     # Pick point with max distance (deterministic for reproducibility)
#                     next_idx = torch.argmax(dists).item()
#                     centroids[i] = cls_feats[next_idx]

#             centroids = F.normalize(centroids, dim=-1)

#             # 2. Run Spherical K-Means
#             for _ in range(20):
#                 sims = torch.mm(cls_feats, centroids.t())
#                 labels = torch.argmax(sims, dim=1)
                
#                 new_centroids = []
#                 for i in range(k):
#                     mask = (labels == i)
#                     if mask.sum() > 0:
#                         cluster_mean = cls_feats[mask].mean(dim=0)
#                         new_centroids.append(cluster_mean)
#                     else:
#                         # Re-init empty cluster to random point (rare fallback)
#                         rand_idx = torch.randint(0, num_samples, (1,)).item()
#                         new_centroids.append(cls_feats[rand_idx])
                
#                 centroids = torch.stack(new_centroids)
#                 centroids = F.normalize(centroids, dim=-1)

#             # 3. Select Prototypes (Samples closest to final centroids)
#             sims = torch.mm(cls_feats, centroids.t()) # [N, K]
#             used_indices_in_class = set()
            
#             # For each cluster, find best representative
#             # We iterate by cluster ID
#             for i in range(k):
#                 # Get samples belonging to this cluster
#                 cluster_mask = (labels == i)
                
#                 if cluster_mask.sum() > 0:
#                     # Filter sims to only points in this cluster
#                     cluster_sims = sims[:, i].clone()
#                     cluster_sims[~cluster_mask] = -1.0
                    
#                     # Sort to handle duplicates (if multiple centroids converge to same point)
#                     sorted_vals, sorted_idxs = torch.sort(cluster_sims, descending=True)
                    
#                     found = False
#                     for idx in sorted_idxs:
#                         idx = idx.item()
#                         if idx not in used_indices_in_class:
#                             selected_indices.append(global_indices[idx].item())
#                             used_indices_in_class.add(idx)
#                             found = True
#                             break
                    
#                     # Fallback: if all points in cluster used (unlikely), pick any unused global point
#                     if not found:
#                          for idx in range(num_samples):
#                             if idx not in used_indices_in_class:
#                                 selected_indices.append(global_indices[idx].item())
#                                 used_indices_in_class.add(idx)
#                                 break
#                 else:
#                     # Empty cluster: just pick the best unused point globally
#                     sim_to_centroid = sims[:, i]
#                     sorted_idxs = torch.argsort(sim_to_centroid, descending=True)
#                     for idx in sorted_idxs:
#                         idx = idx.item()
#                         if idx not in used_indices_in_class:
#                             selected_indices.append(global_indices[idx].item())
#                             used_indices_in_class.add(idx)
#                             break

#     selected_indices.sort()
#     herded_data_source = [data_source[i] for i in selected_indices]
    
#     print(f"[Herding] Selection complete. Reduced from {len(data_source)} to {len(herded_data_source)} samples.")
#     return herded_data_source


@torch.no_grad()
def select_shots_herding(data_source, input_size, model, transform, num_shots, device='cuda', batch_size=64):
    """
    Selects N shots per class using Standard Herding (Welling's Mean-Shift).
    
    Algorithm:
    1. Calculate the global class mean (mu).
    2. Iteratively select sample x_i such that the average of selected 
       samples (including x_i) gets closest to mu.
    """
    print(f"\n[Herding] Selecting best {num_shots} shots from {len(data_source)} total samples...")
    
    model.eval()
    model.to(device)
    
    # --- 1. Robust Feature Extraction ---
    class FeatureDataset(torch.utils.data.Dataset):
        def __init__(self, data, tfm):
            self.data = data
            self.tfm = tfm
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            from PIL import Image
            img = Image.open(item.impath).convert('RGB')
            if self.tfm: img = self.tfm(img)
            return img, item.label

    dataset = FeatureDataset(data_source, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=8, pin_memory=True, drop_last=False)
    
    all_features = []
    all_labels = []
    
    print("[Herding] Extracting features...")
    for images, labels in tqdm(loader):
        images = images.to(device)
        
        # Handle the patched model (Tuple return vs Tensor return)
        if hasattr(model, 'encode_image'): 
            # If using standard OpenCLIP or patched method
            features = model.encode_image(images)
        else:
            features = model(images)

        # If features is a tuple (Global, Spatial), take Global [0]
        if isinstance(features, tuple):
            features = features[0]
            
        # L2 Normalize features (Critical for CLIP)
        features = F.normalize(features, dim=-1)
        
        all_features.append(features.cpu())
        all_labels.append(labels)
        
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    selected_indices = []
    unique_classes = torch.unique(all_labels).sort().values
    
    print(f"[Herding] Running Iterative Mean-Shift Selection...")
    
    for cls_idx in tqdm(unique_classes):
        cls_mask = (all_labels == cls_idx)
        
        # Get Global indices (to map back to data_source) and Features
        global_indices = torch.nonzero(cls_mask).squeeze(1)
        cls_feats = all_features[cls_mask] # Keep on CPU to save GPU memory, or move to GPU if small
        
        num_samples = len(global_indices)
        k = min(num_shots, num_samples)
        
        # Case: Not enough samples, take all
        if k == num_samples:
            selected_indices.extend(global_indices.tolist())
            continue

        # --- HERDING ALGORITHM ---
        
        # 1. Calculate the Global Class Mean (Target)
        class_mean = cls_feats.mean(dim=0)
        class_mean = F.normalize(class_mean, dim=0) # Re-normalize mean
        
        selected_in_class = []
        current_sum = torch.zeros_like(class_mean)
        
        # Mask to ensure we don't pick the same sample twice
        remaining_mask = torch.ones(num_samples, dtype=torch.bool)
        
        for i in range(k):
            # We want to find x* that minimizes: || (current_sum + x*) / (i+1) - class_mean ||
            # This is equivalent to minimizing: || (current_sum + x*) - (i+1) * class_mean ||
            
            target = (i + 1) * class_mean - current_sum
            
            # Since inputs are normalized, minimizing Euclidean distance 
            # is equivalent to maximizing Dot Product (Cosine Sim) 
            # between x* and the residual target.
            
            # Extract only remaining features
            # Note: Doing this on CPU is fine, move to .cuda() if slow
            candidates = cls_feats
            
            # Compute dot product
            sims = torch.matmul(candidates, target)
            
            # Mask out already selected
            sims[~remaining_mask] = -float('inf')
            
            # Select best
            best_rel_idx = torch.argmax(sims).item()
            
            # Update state
            selected_in_class.append(global_indices[best_rel_idx].item())
            current_sum += candidates[best_rel_idx]
            remaining_mask[best_rel_idx] = False
            
        selected_indices.extend(selected_in_class)

    # Sort to maintain original dataset order (optional but good for debugging)
    selected_indices.sort()
    
    herded_data_source = [data_source[i] for i in selected_indices]
    
    print(f"[Herding] Selection complete. Reduced from {len(data_source)} to {len(herded_data_source)} samples.")
    return herded_data_source