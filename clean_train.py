import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
import numpy as np

try:
    from model_rwkv import DehazeRWKV_Real
except ImportError:
    print("❌ 错误：找不到 model_rwkv.py")
    exit()

class DehazeDataset(Dataset):
    def __init__(self, data_root):
        self.hazy_dir = os.path.join(data_root, "archive(1)", "hazy")
        self.clear_dir = os.path.join(data_root, "archive(1)", "clear")
        self.hazy_paths = sorted(glob.glob(os.path.join(self.hazy_dir, "*.jpg"))) + \
                          sorted(glob.glob(os.path.join(self.hazy_dir, "*.png")))
        self.clear_paths = sorted(glob.glob(os.path.join(self.clear_dir, "*.jpg"))) + \
                           sorted(glob.glob(os.path.join(self.clear_dir, "*.png")))
        
    def __len__(self):
        return len(self.hazy_paths)

    def __getitem__(self, idx):
        try:
            hazy_path = self.hazy_paths[idx]
            filename = os.path.basename(hazy_path)
            clear_path = os.path.join(self.clear_dir, filename)
            if not os.path.exists(clear_path):
                clear_path = self.clear_paths[idx % len(self.clear_paths)]
            hazy_img = Image.open(hazy_path).convert('RGB').resize((128, 128))
            clear_img = Image.open(clear_path).convert('RGB').resize((128, 128))
            return torch.tensor(np.array(hazy_img).transpose(2,0,1)/255.0, dtype=torch.float32), \
                   torch.tensor(np.array(clear_img).transpose(2,0,1)/255.0, dtype=torch.float32)
        except:
            return self.__getitem__(0)

def main():
    # 1. 自动获取所有可用显卡
    device_ids = list(range(torch.cuda.device_count()))
    print(f"🖥️ 检测到可用显卡数量: {len(device_ids)}")
    device = torch.device('cuda:0')
    
    # 2. 实例化模型
    model = DehazeRWKV_Real(hidden_dim=32, num_blocks=2)
    
    # 🌟 关键点：如果有多张卡，直接开启 DataParallel
    if len(device_ids) > 1:
        print(f"🚀 正在使用 DataParallel 在 {len(device_ids)} 张卡上并行训练...")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    
    # 3. 增大 Batch Size (现在有 8 张卡，可以放心开大)
    BATCH_SIZE = 128 
    dataset = DehazeDataset("/mnt/sdb1/leijh/EnergySnake1/robot/Restore-RWKV/data/dehaze_data")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(100):
        for i, (hazy, clear) in enumerate(loader):
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad()
            out = model(hazy)
            loss = criterion(out, clear)
            loss.backward()
            optimizer.step()
            
            if i % 20 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")
        
        if epoch % 5 == 0:
            # 🌟 保存时注意：如果是 DataParallel，需要取 model.module
            save_obj = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            os.makedirs("./checkpoints/rwkv", exist_ok=True)
            torch.save(save_obj, f"./checkpoints/rwkv/rwkv_dehaze_epoch_{epoch}.pth")
    
if __name__ == '__main__':
    main()
