"""纯卷积基线推理，输出目录与 RWKV 版一致，便于消融对比。"""
import os
import glob
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from model_cnn_baseline import DehazeCNN

CHECKPOINT_PATH = "./checkpoints/cnn/cnn_baseline_epoch_99.pth"
INPUT_DIR = "./test_images"
RESULT_DIR = "./outputs/cnn_ablation/result"
COMPARE_DIR = "./outputs/cnn_ablation/compare"


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if not os.path.isfile(CHECKPOINT_PATH):
    print(f"❌ 找不到权重: {CHECKPOINT_PATH}")
    return

  os.makedirs(RESULT_DIR, exist_ok=True)
  os.makedirs(COMPARE_DIR, exist_ok=True)

  image_paths = sorted(
    p
    for p in glob.glob(os.path.join(INPUT_DIR, "*.*"))
    if p.lower().endswith((".png", ".jpg", ".jpeg"))
  )
  if not image_paths:
    print(f"⚠️ 在 {INPUT_DIR} 里没有测试图片。")
    return

  print(f"🎉 使用纯卷积基线权重: {CHECKPOINT_PATH}")
  model = DehazeCNN().to(device)
  model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
  model.eval()

  for img_path in image_paths:
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    img_np = np.array(img, dtype=np.float32) / 255.0
    # 与训练时一致：RGB -> BGR
    img_np = img_np[:, :, ::-1].copy()
    img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
      output = model(img_tensor)

    output = torch.clamp(output, 0, 1)
    # BGR -> RGB 再保存
    output_rgb = output[:, [2, 1, 0], :, :]
    comparison = torch.cat((img_tensor[:, [2, 1, 0], :, :], output_rgb), dim=3)

    save_image(output_rgb, os.path.join(RESULT_DIR, img_name))
    save_image(comparison, os.path.join(COMPARE_DIR, img_name))
    print(f"✅ {img_name}")

  print(f"📁 去雾结果: {RESULT_DIR}")
  print(f"📁 对比图:   {COMPARE_DIR}")


if __name__ == "__main__":
  main()
