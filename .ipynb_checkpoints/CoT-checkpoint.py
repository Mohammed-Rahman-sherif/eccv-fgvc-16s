import os
import random
import argparse
import yaml
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

import open_clip
import math

from datasets import build_dataset
from datasets.utils import build_data_loader
from utils import *


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--backbone', type=str, default='RN50', help='OpenCLIP backbone (e.g., ViT-B-16, RN50)')
    parser.add_argument('--pretrained', type=str, default='openai', help='OpenCLIP pretrained tag (e.g., openai, laion2b_s34b_b88k)')
    args = parser.parse_args()
    return args


# =============================================================================
# 1. UNIVERSAL PATCH FOR SPATIAL FEATURES (Hybrid Strategy)
# =============================================================================

class SpatialFeatureHook:
    """ Context manager to capture outputs from ViT layers """
    def __init__(self, module):
        self.module = module
        self.hook_handle = None
        self.captured_output = None

    def hook_fn(self, module, input, output):
        self.captured_output = output

    def enable(self):
        self.hook_handle = self.module.register_forward_hook(self.hook_fn)
    
    def disable(self):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def get_data(self):
        return self.captured_output




def patch_open_clip_model(model):
    """
    Applies the appropriate patch based on architecture using Forward Hooks.
    Updated to upsample ResNet spatial features to 14x14.
    """
    original_visual = model.visual
    original_encode_image = model.encode_image

    # ------------------------------------------------------------------
    # CASE A: Vision Transformer (ViT) - (Unchanged)
    # ------------------------------------------------------------------
    if hasattr(original_visual, 'transformer'):
        target_layer = original_visual.ln_post
        hook = SpatialFeatureHook(target_layer)
        hook.enable() 

        def new_encode_image_vit(self, image):
            global_feat = original_encode_image(image)
            full_sequence = hook.get_data() # (N, L, D)
            
            if hasattr(original_visual, 'proj') and original_visual.proj is not None:
                full_sequence = full_sequence @ original_visual.proj
            
            spatial_feat = full_sequence[:, 1:, :] 
            return global_feat, spatial_feat

        import types
        model.encode_image = types.MethodType(new_encode_image_vit, model)

    # ------------------------------------------------------------------
    # CASE B: ResNet (RN50) -> Updated to Upsample to 14x14
    # ------------------------------------------------------------------
    elif hasattr(original_visual, 'attnpool'):
        # Hook the last ResNet layer
        target_layer = original_visual.layer4
        hook = SpatialFeatureHook(target_layer)
        hook.enable()

        def new_encode_image_resnet(self, image):
            # 1. Run standard forward to trigger hook and get global features
            global_feat = original_encode_image(image)

            # 2. Retrieve spatial feature map: (N, 2048, 7, 7)
            x_raw = hook.get_data() 
            
            # 3. Apply Attention Pooling logic to transform features to embedding space
            attnpool = original_visual.attnpool

            # Flatten: (N, C, H, W) -> (HW, N, C) -> (49, N, 2048)
            x = x_raw.reshape(x_raw.shape[0], x_raw.shape[1], -1).permute(2, 0, 1)
            
            # Add mean token
            x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)

            # Add Positional Embedding (standard CLIP RN50 pos_embed is for 7x7)
            pos_embed = attnpool.positional_embedding[:, None, :].to(x.dtype)
            if x.shape[0] != pos_embed.shape[0]:
                pos_embed = pos_embed[:x.shape[0]]
            x = x + pos_embed

            # Run Multi-Head Attention
            x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=attnpool.num_heads,
                q_proj_weight=attnpool.q_proj.weight,
                k_proj_weight=attnpool.k_proj.weight,
                v_proj_weight=attnpool.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([attnpool.q_proj.bias, attnpool.k_proj.bias, attnpool.v_proj.bias]),
                bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0,
                out_proj_weight=attnpool.c_proj.weight,
                out_proj_bias=attnpool.c_proj.bias,
                use_separate_proj_weight=True,
                training=attnpool.training,
                need_weights=False
            )

            # -------------------------------------------------------
            # UP-SAMPLING LOGIC
            # -------------------------------------------------------
            # x is currently (50, N, D) -> [Global, Spatial...]
            spatial_tokens = x[1:] # (49, N, D)
            
            # Permute to (N, D, 49)
            spatial_tokens = spatial_tokens.permute(1, 2, 0)
            
            # Reshape to grid (N, D, 7, 7)
            side = int(math.sqrt(spatial_tokens.shape[-1]))
            feature_map = spatial_tokens.view(spatial_tokens.shape[0], spatial_tokens.shape[1], side, side)
            
            # INTERPOLATE: 7x7 -> 14x14
            # This simulates creating patches at a higher resolution
            feature_map = F.interpolate(feature_map, size=(14, 14), mode='bicubic', align_corners=False)
            
            # Flatten back: (N, D, 14, 14) -> (N, D, 196) -> (N, 196, D)
            spatial_feat = feature_map.flatten(2).permute(0, 2, 1)

            return global_feat, spatial_feat

        import types
        model.encode_image = types.MethodType(new_encode_image_resnet, model)

    else:
        raise ValueError("Unknown OpenCLIP architecture.")

    return model

# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================

def pre_load_features(cfg, split, model, loader):
    global_features, spatial_features, labels = [], [], []
    print(f"Extracting Global + Spatial features for {split}...")
    
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            
            global_feat, spatial_feat = model.encode_image(images)
            
            global_feat /= global_feat.norm(dim=-1, keepdim=True)
            spatial_feat /= spatial_feat.norm(dim=-1, keepdim=True)

            global_features.append(global_feat)
            spatial_features.append(spatial_feat)
            labels.append(target)

    global_features = torch.cat(global_features)
    spatial_features = torch.cat(spatial_features)
    labels = torch.cat(labels)
    
    return global_features, spatial_features, labels


# =============================================================================
# 3. TEXT-GUIDED SPATIAL CHAIN OF THOUGHT (Logic Unchanged)
# =============================================================================

def whitening_loss(adapter_weight):
    """ Forces adapter keys to be orthogonal (Feature Decorrelation) """
    w = adapter_weight / (adapter_weight.norm(dim=1, keepdim=True) + 1e-8)
    gram = torch.mm(w, w.t())
    I = torch.eye(gram.shape[0]).to(gram.device)
    loss = torch.norm(gram - I, p='fro')
    return loss / (gram.shape[0])


def inference_cot(global_feats, spatial_feats, cache_keys, cache_values, clip_weights, beta, alpha, 
                  adapter=None, adapter_proj=None, cot_scale=0.0, global_scale=1.0, training_mode=False):
    
    # --- Step 1: Global Glance ---
    if adapter:
        base_affinity = adapter(global_feats)
    else:
        base_affinity = global_feats @ cache_keys
        
    base_cache_logits = ((-1) * (beta - beta * base_affinity)).exp() @ cache_values
    clip_logits = 100. * global_feats @ clip_weights
    base_tip_logits = clip_logits + base_cache_logits * alpha

    # Handle float/tensor conversion for scales
    c_scale_val = cot_scale.item() if isinstance(cot_scale, torch.Tensor) else cot_scale
    g_scale_val = global_scale.item() if isinstance(global_scale, torch.Tensor) else global_scale
    
    # If CoT scale is effectively zero and we are not training, just return base
    if abs(c_scale_val) < 1e-5 and abs(g_scale_val - 1.0) < 1e-5 and not training_mode:
        return base_tip_logits

    # --- Step 2: Text-Guided Discovery ---
    pred_indices = base_tip_logits.argmax(dim=1) 
    
    target_text_emb = clip_weights.t()[pred_indices] 
    target_text_emb = target_text_emb / target_text_emb.norm(dim=-1, keepdim=True)
    
    sim_map = torch.bmm(spatial_feats, target_text_emb.unsqueeze(-1)).squeeze(-1)
    
    attn_weights = F.softmax(sim_map * 10, dim=1).unsqueeze(-1) 
    
    # --- Step 3: Feature Refinement ---
    focal_feat = (spatial_feats * attn_weights).sum(dim=1)
    focal_feat = focal_feat / (focal_feat.norm(dim=-1, keepdim=True) + 1e-8)
    
    if adapter_proj:
        focal_feat = adapter_proj(focal_feat)
        focal_feat = focal_feat / (focal_feat.norm(dim=-1, keepdim=True) + 1e-8)
    
    # --- Step 4: Final Prediction (Updated with Global Scale) ---
    # Apply global_scale to global_feats and cot_scale to focal_feat
    refined_query = (global_feats * g_scale_val) + (cot_scale * focal_feat)
    
    # Normalize to keep on hypersphere
    refined_query = refined_query / (refined_query.norm(dim=-1, keepdim=True) + 1e-8)
    
    if adapter:
        cot_affinity = adapter(refined_query)
    else:
        cot_affinity = refined_query @ cache_keys
        
    cot_cache_logits = ((-1) * (beta - beta * cot_affinity)).exp() @ cache_values
    cot_tip_logits = clip_logits + cot_cache_logits * alpha

    if training_mode:
        return cot_tip_logits, base_tip_logits
    else:
        return cot_tip_logits



# =============================================================================
# 4. HP SEARCH STRATEGIES
# =============================================================================

def search_hp_original(cfg, cache_keys, cache_values, global_feats, labels, clip_weights, adapter=None):
    if cfg['search_hp'] == True:
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        
        print(f"Searching HP (Original Tip-Adapter Mode)...")

        with torch.no_grad():
            if adapter:
                affinity = adapter(global_feats)
            else:
                affinity = global_feats @ cache_keys
            
            clip_logits = 100. * global_feats @ clip_weights

        for beta in beta_list:
            for alpha in alpha_list:
                
                cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                tip_logits = clip_logits + cache_logits * alpha
                
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("Best (Original): beta: {:.2f}, alpha: {:.2f}; Acc: {:.2f}".format(best_beta, best_alpha, best_acc))
        return best_beta, best_alpha
    
    return cfg['init_beta'], cfg['init_alpha']


def search_hp_cot(cfg, cache_keys, cache_values, global_feats, spatial_feats, labels, clip_weights, adapter=None, adapter_proj=None, cot_scale=0.0, global_scale=1.0):
    if cfg['search_hp'] == True:
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]

        best_acc = 0
        best_beta, best_alpha = 0, 0
        
        print(f"Searching HP (CoT Mode)...")
        
        # Pre-calculate the refined affinity ONCE since scales are fixed during HP search
        with torch.no_grad():
            # 1. Base Logic for finding targets
            if adapter:
                base_affinity = adapter(global_feats)
            else:
                base_affinity = global_feats @ cache_keys
            
            base_cache_logits = ((-1) * (1.0 - 1.0 * base_affinity)).exp() @ cache_values
            clip_logits = 100. * global_feats @ clip_weights
            base_tip_logits = clip_logits + base_cache_logits

            # Handle scale values
            c_scale_val = cot_scale.item() if isinstance(cot_scale, torch.Tensor) else cot_scale
            g_scale_val = global_scale.item() if isinstance(global_scale, torch.Tensor) else global_scale
            
            # 2. Compute Refined Query
            if abs(c_scale_val) < 1e-5:
                # If cot_scale is 0, just use global (scaled or not)
                refined_query = global_feats * g_scale_val
                refined_query = refined_query / (refined_query.norm(dim=-1, keepdim=True) + 1e-8)
            else:
                pred_indices = base_tip_logits.argmax(dim=1)
                target_text_emb = clip_weights.t()[pred_indices]
                target_text_emb = target_text_emb / target_text_emb.norm(dim=-1, keepdim=True)
                
                sim_map = torch.bmm(spatial_feats, target_text_emb.unsqueeze(-1)).squeeze(-1)
                attn_weights = F.softmax(sim_map * 10, dim=1).unsqueeze(-1)
                
                focal_feat = (spatial_feats * attn_weights).sum(dim=1)
                focal_feat = focal_feat / (focal_feat.norm(dim=-1, keepdim=True) + 1e-8)
                
                if adapter_proj:
                    focal_feat = adapter_proj(focal_feat)
                    focal_feat = focal_feat / (focal_feat.norm(dim=-1, keepdim=True) + 1e-8)
                
                # Apply Both Scales
                refined_query = (global_feats * g_scale_val) + (cot_scale * focal_feat)
                refined_query = refined_query / (refined_query.norm(dim=-1, keepdim=True) + 1e-8)

            # 3. Compute Final Affinity
            if adapter:
                final_affinity = adapter(refined_query)
            else:
                final_affinity = refined_query @ cache_keys
            
            final_clip_logits = 100. * global_feats @ clip_weights

        # 4. Search Beta/Alpha
        for beta in beta_list:
            for alpha in alpha_list:
                cache_logits = ((-1) * (beta - beta * final_affinity)).exp() @ cache_values
                tip_logits = final_clip_logits + cache_logits * alpha
                
                acc = cls_acc(tip_logits, labels)
            
                if acc > best_acc:
                    best_acc = acc
                    best_beta = beta
                    best_alpha = alpha

        print("Best (CoT): beta: {:.2f}, alpha: {:.2f}; Acc: {:.2f}".format(best_beta, best_alpha, best_acc))
        return best_beta, best_alpha
    
    return cfg['init_beta'], cfg['init_alpha']


# =============================================================================
# 5. TRAINING LOOPS
# =============================================================================

def run_tip_adapter(cfg, cache_keys, cache_values, val_global, val_labels, test_global, test_spatial, test_labels, clip_weights):
    print("\n-------- Searching hyperparameters on the val set (Training-Free). --------")
    best_beta, best_alpha = search_hp_original(cfg, cache_keys, cache_values, val_global, val_labels, clip_weights)

    print("\n-------- Evaluating on the test set (Standard Inference). --------")
    tip_logits = inference_cot(test_global, test_spatial, cache_keys, cache_values, clip_weights, best_beta, best_alpha, cot_scale=0.0)
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter (Standard)'s test accuracy: {:.2f}. ****\n".format(acc))


def run_tip_adapter_F(cfg, cache_keys, cache_values, val_global, val_spatial, val_labels, test_global, test_spatial, test_labels, clip_weights, clip_model, train_loader_F):
    
    print("\n-------- Starting Fine-tuning (Tip-Adapter-F) with Projected Spatial CoT + Global Scale --------")
    
    # Safely get dtype
    model_dtype = next(clip_model.parameters()).dtype

    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(model_dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    feat_dim = cache_keys.shape[0]
    adapter_proj = nn.Linear(feat_dim, feat_dim, bias=True).to(model_dtype).cuda()
    nn.init.eye_(adapter_proj.weight)
    nn.init.zeros_(adapter_proj.bias)
    
    # Initialize Scales
    # cot_scale: 0.1 (small start)
    # global_scale: 1.0 (standard start)
    cot_scale = nn.Parameter(torch.tensor(0.1).cuda())
    global_scale = nn.Parameter(torch.tensor(1.0).cuda())
    
    optimizer = torch.optim.AdamW([
        {'params': adapter.parameters(), 'lr': cfg['lr']},
        {'params': adapter_proj.parameters(), 'lr': cfg['lr']}, 
        {'params': cot_scale, 'lr': cfg['lr'] * 10},
        {'params': global_scale, 'lr': cfg['lr'] * 10} # Add global_scale to optimizer
    ], lr=cfg['lr'], eps=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0
    
    orth_lambda = 0.05 
    improvement_lambda = 1.0

    for train_idx in range(cfg['train_epoch']):
        adapter.train()
        adapter_proj.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        
        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                global_feat, spatial_feat = clip_model.encode_image(images)
                global_feat /= global_feat.norm(dim=-1, keepdim=True)
                spatial_feat /= spatial_feat.norm(dim=-1, keepdim=True)

            cot_logits, base_logits = inference_cot(
                global_feat, spatial_feat, cache_keys, cache_values, clip_weights, 
                beta, alpha, adapter=adapter, adapter_proj=adapter_proj, 
                cot_scale=cot_scale, global_scale=global_scale, # Pass global_scale
                training_mode=True
            )

            loss_cls = F.cross_entropy(cot_logits, target)
            loss_base = F.cross_entropy(base_logits, target).detach()
            
            loss_imp = F.relu(loss_cls - loss_base)
            loss_orth = whitening_loss(adapter.weight)
            
            loss = loss_cls + orth_lambda * loss_orth + improvement_lambda * loss_imp

            acc = cls_acc(cot_logits, target)
            correct_samples += acc / 100 * len(cot_logits)
            all_samples += len(cot_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clamp scales to be non-negative to avoid vector flipping
            cot_scale.data.clamp_(min=0.0)
            global_scale.data.clamp_(min=0.01) 
            
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('Epoch: {:}/{:}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}, CoT Scale: {:.4f}, Global Scale: {:.4f}'.format(
            train_idx, cfg['train_epoch'], current_lr, correct_samples / all_samples, 
            sum(loss_list)/len(loss_list), cot_scale.item(), global_scale.item()))

        adapter.eval()
        adapter_proj.eval()
        with torch.no_grad():
            tip_logits = inference_cot(test_global, test_spatial, cache_keys, cache_values, clip_weights, 
                                       beta, alpha, adapter=adapter, adapter_proj=adapter_proj, 
                                       cot_scale=cot_scale, global_scale=global_scale)
            acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F (CoT)'s test accuracy: {:.2f}. ****".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            # Save all parameters including global_scale
            state = {
                'adapter': adapter.weight, 
                'adapter_proj': adapter_proj.state_dict(), 
                'cot_scale': cot_scale,
                'global_scale': global_scale
            }
            torch.save(state, cfg['cache_dir'] + "/best_F_" + "RN" + str(cfg['shots']) + "shots.pt")
    
    # Reload best model
    state = torch.load(cfg['cache_dir'] + "/best_F_" + "RN" + str(cfg['shots']) + "shots.pt")
    adapter.weight = nn.Parameter(state['adapter'])
    adapter_proj.load_state_dict(state['adapter_proj'])
    cot_scale = state['cot_scale']
    global_scale = state['global_scale']
    
    print(f"\n**** After fine-tuning, Best Acc: {best_acc:.2f} at epoch {best_epoch}. ****")

    print("\n-------- Searching hyperparameters on the val set (Tip-Adapter-F). --------")
    
    # Search for Original (Adapter only, no CoT scales)
    best_beta_orig, best_alpha_orig = search_hp_original(
        cfg, cache_keys, cache_values, val_global, val_labels, 
        clip_weights, adapter=adapter
    )
    
    # Search for CoT (Uses adapter + proj + both scales)
    best_beta_cot, best_alpha_cot = search_hp_cot(
        cfg, cache_keys, cache_values, val_global, val_spatial, val_labels, 
        clip_weights, adapter=adapter, adapter_proj=adapter_proj, 
        cot_scale=cot_scale, global_scale=global_scale
    )

    print("\n-------- Evaluating on the test set. --------")
    
    # Test Original
    logits_orig = inference_cot(test_global, test_spatial, cache_keys, cache_values, clip_weights, 
                                best_beta_orig, best_alpha_orig, adapter=adapter, 
                                adapter_proj=None, cot_scale=0.0, global_scale=1.0) # Explicit defaults
    acc_orig = cls_acc(logits_orig, test_labels)
    
    # Test CoT
    logits_cot = inference_cot(test_global, test_spatial, cache_keys, cache_values, clip_weights, 
                               best_beta_cot, best_alpha_cot, adapter=adapter, adapter_proj=adapter_proj, 
                               cot_scale=cot_scale, global_scale=global_scale)
    acc_cot = cls_acc(logits_cot, test_labels)
    
    print("Test Accuracy (using Original HP): {:.2f}".format(acc_orig))
    print("Test Accuracy (using CoT HP):      {:.2f}".format(acc_cot))
    print("**** Tip-Adapter-F (CoT) Best Test Accuracy: {:.2f}. ****\n".format(max(best_acc, acc_orig, acc_cot)))


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    args = get_arguments()
    assert (os.path.exists(args.config))
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cache_dir = os.path.join('./caches', cfg['dataset'])
    os.makedirs(cache_dir, exist_ok=True)
    cfg['cache_dir'] = cache_dir

    print("\nRunning configs.")
    print(cfg, "\n")

    print(f"Loading OpenCLIP Model: {args.backbone} | Pretrained: {args.pretrained}")
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.backbone, 
        pretrained=args.pretrained
    )
    clip_model.cuda()
    clip_model.eval()

    # Apply the Universal Patch to enable Spatial Feature Extraction
    clip_model = patch_open_clip_model(clip_model)

    random.seed(1)
    torch.manual_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Preparing dataset.")
    # dataset = build_dataset(cfg['dataset'], cfg['root_path'], cfg['shots'])
    dataset = build_dataset(cfg['dataset'], cfg['root_path'], shots=-1)
    
    val_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    from datasets.utils import select_shots_herding
    dataset._train_x = select_shots_herding(
        data_source=dataset.train_x,
        input_size=224,
        model=clip_model.visual,
        transform=val_transform,
        num_shots=cfg['shots'],  # Use the actual shots from config here
        device=device
    )

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64, tfm=preprocess, is_train=False, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64, tfm=preprocess, is_train=False, shuffle=False)

    mean = getattr(preprocess.transforms[-1], 'mean', (0.48145466, 0.4578275, 0.40821073))
    std = getattr(preprocess.transforms[-1], 'std', (0.26862954, 0.26130258, 0.27577711))
    
    train_tranform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_loader_cache = build_data_loader(data_source=dataset._train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=False)
    train_loader_F = build_data_loader(data_source=dataset._train_x, batch_size=256, tfm=train_tranform, is_train=True, shuffle=True)

    # --------------------------------------------------------------------------
    # Textual Features with Prompt Ensembling
    # --------------------------------------------------------------------------
    print("\nGetting textual features as CLIP's classifier.")
    with torch.no_grad():
        tokenizer = open_clip.get_tokenizer(args.backbone)
        
        templates = dataset.template
        if isinstance(templates, str):
            templates = [templates]
            
        all_class_embeddings = []
        
        print(f"Processing {len(dataset.classnames)} classes with {len(templates)} templates each...")
        
        for classname in tqdm(dataset.classnames):
            name = classname.replace("_", " ") 
            texts = [t.format(name) for t in templates]
            text_tokens = tokenizer(texts).cuda() 
            class_emb = clip_model.encode_text(text_tokens)
            class_emb /= class_emb.norm(dim=-1, keepdim=True)
            mean_emb = class_emb.mean(dim=0)
            mean_emb /= mean_emb.norm()
            all_class_embeddings.append(mean_emb)
            
        clip_weights = torch.stack(all_class_embeddings, dim=1).cuda()

    print("\nConstructing cache model by few-shot visual features and labels.")
    
    augment_epoch = cfg['augment_epoch']
    print(f"Augment Epochs: {augment_epoch}")
    
    accumulated_features = None
    all_labels = None

    with torch.no_grad():
        for k in range(augment_epoch):
            print(f"Augment Epoch: {k+1}/{augment_epoch}")
            epoch_features = []
            epoch_labels = []
            for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                images = images.cuda()
                target = target.cuda()
                
                # Use our patched encode_image, but we only need global for the cache
                global_feat, _ = clip_model.encode_image(images) 
                global_feat /= global_feat.norm(dim=-1, keepdim=True)
                
                epoch_features.append(global_feat)
                epoch_labels.append(target)
            
            epoch_features = torch.cat(epoch_features)
            epoch_labels = torch.cat(epoch_labels)
            
            if k == 0:
                accumulated_features = epoch_features
                all_labels = epoch_labels
            else:
                accumulated_features += epoch_features

    accumulated_features /= augment_epoch
    accumulated_features /= accumulated_features.norm(dim=-1, keepdim=True)

    cache_keys = accumulated_features.permute(1, 0)
    cache_values = F.one_hot(all_labels).to(cache_keys.dtype)
    
    print(f"Cache Keys Shape: {cache_keys.shape}")   
    print(f"Cache Values Shape: {cache_values.shape}") 

    print("\nLoading visual features and labels from val set.")
    val_global, val_spatial, val_labels = pre_load_features(cfg, "val", clip_model, val_loader)

    print("\nLoading visual features and labels from test set.")
    test_global, test_spatial, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)

    run_tip_adapter(cfg, cache_keys, cache_values, val_global, val_labels, test_global, test_spatial, test_labels, clip_weights)
    
    run_tip_adapter_F(cfg, cache_keys, cache_values, val_global, val_spatial, val_labels, 
                      test_global, test_spatial, test_labels, clip_weights, clip_model, train_loader_F)
            

if __name__ == '__main__':
    main()