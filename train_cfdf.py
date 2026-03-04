import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 引入我们系统的 CF-DF 模型
from backend.models.cf_df import CFDFModel
from backend.config import CLIP_FEATURE_DIM

# ================= 1. 数据集定义 =================
class MAMIFeatureDataset(Dataset):
    def __init__(self, pkl_path):
        print(f"Loading features from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)
        self.keys = list(self.data_dict.keys())
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        sample = self.data_dict[key]
        
        # 提取并展平特征 (从 [1, 512] -> [512])
        text_feat = sample['text_features_clip'].squeeze(0).float()
        image_feat = sample['image_features_clip'].squeeze(0).float()
        label = sample['label'].squeeze(0).float()
        
        return text_feat, image_feat, label

# ================= 2. 训练配置 =================
TRAIN_PKL = "/home/munan/experiment/Multimodal/dataset/MAMI/precomputed_features/train_path_features.pkl"
TEST_PKL = "/home/munan/experiment/Multimodal/dataset/MAMI/precomputed_features/test_path_features.pkl"
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
DIFFUSION_WEIGHT = 0.1  # 扩散模型损失的权重

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 3. 主训练循环 =================
def main():
    # 数据加载
    train_dataset = MAMIFeatureDataset(TRAIN_PKL)
    test_dataset = MAMIFeatureDataset(TEST_PKL)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 模型初始化
    model = CFDFModel(feature_dim=CLIP_FEATURE_DIM).to(device)
    
    # 优化器与损失函数
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion_cls = nn.BCEWithLogitsLoss()
    criterion_diff = nn.MSELoss()
    
    best_f1 = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_cls_loss, total_diff_loss = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for text_feat, image_feat, labels in pbar:
            text_feat, image_feat, labels = text_feat.to(device), image_feat.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # --- 手动前向传播以进行训练 ---
            # 1. 特征投影
            f_t = model.text_projection(text_feat)
            f_v = model.image_projection(image_feat)
            
            # 2. 构建反事实特征
            F_o, F_no_text, F_no_image = model._counterfactual_features(f_t, f_v)
            
            # 3. 计算分类损失 (三个世界均受监督)
            z_o = model.classifier(F_o).squeeze(-1)
            z_no_t = model.classifier(F_no_text).squeeze(-1)
            z_no_v = model.classifier(F_no_image).squeeze(-1)
            
            loss_cls = (criterion_cls(z_o, labels) + 
                        criterion_cls(z_no_t, labels) + 
                        criterion_cls(z_no_v, labels)) / 3.0
            
            # 4. 计算扩散去噪损失
            batch_size = F_o.size(0)
            t = torch.randint(0, model.alphas_cumprod.size(0), (batch_size,), device=device)
            
            def get_diffusion_loss(feat):
                noise = torch.randn_like(feat)
                sqrt_alpha = model.sqrt_alphas_cumprod[t].unsqueeze(-1)
                sqrt_one_minus_alpha = model.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
                
                # 前向加噪
                x_t = sqrt_alpha * feat + sqrt_one_minus_alpha * noise
                # 去噪网络预测噪声
                eps_pred = model.denoiser(x_t, t.float() / model.alphas_cumprod.size(0))
                return criterion_diff(eps_pred, noise)
            
            # 共享去噪网络，同时学习净化三种特征
            loss_diff = (get_diffusion_loss(F_o) + 
                         get_diffusion_loss(F_no_text) + 
                         get_diffusion_loss(F_no_image)) / 3.0
            
            # 5. 总损失与反向传播
            loss = loss_cls + DIFFUSION_WEIGHT * loss_diff
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_diff_loss += loss_diff.item()
            
            pbar.set_postfix({
                "Loss": total_loss/len(pbar), 
                "Cls": total_cls_loss/len(pbar), 
                "Diff": total_diff_loss/len(pbar)
            })
            
        # --- 验证阶段 ---
        model.eval()
        all_preds, all_labels = [], []
        
        print(f"Evaluating Epoch {epoch}...")
        with torch.no_grad():
            for text_feat, image_feat, labels in test_loader:
                text_feat, image_feat = text_feat.to(device), image_feat.to(device)
                
                # 验证时调用模型原本的 forward 机制 (执行真实的 Tweedie 去噪净化并聚合因果效应)
                # t_star=50, num_samples=10 这是原模型代码的默认超参数
                final_prob, _ = model(text_feat, image_feat, t_star=50, num_samples=10)
                
                preds = (final_prob >= 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch {epoch} Result: Acc: {acc:.4f} | F1: {f1:.4f} | Precision: {p:.4f} | Recall: {r:.4f}")
        
        # 保存表现最好的模型
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(SAVE_DIR, "CFDF_MAMI_best.pt")
            torch.save(model.state_dict(), save_path)
            print(f"🚀 New best model saved! (F1: {best_f1:.4f}) -> {save_path}")

if __name__ == "__main__":
    main()
