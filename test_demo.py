import os
import glob
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from model_rwkv import DehazeRWKV_Real

CHECKPOINT_PATH = "./checkpoints/rwkv/rwkv_dehaze_epoch_50.pth"
INPUT_DIR = "./test_images"
RESULT_DIR = "./outputs/rwkv/result"
COMPARE_DIR = "./outputs/rwkv/compare"


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
    print(f"⚠️ 在 {INPUT_DIR} 里没有测试图片，请先放入 png/jpg。")
    return

  print(f"🎉 使用权重: {CHECKPOINT_PATH}")
  print("🏗️ 加载 Vision-RWKV (hidden_dim=32, num_blocks=2)...")
  model = DehazeRWKV_Real(hidden_dim=32, num_blocks=2).to(device)

  state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
  if list(state_dict.keys())[0].startswith("module."):
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
  model.load_state_dict(state_dict)
  model.eval()

  print(f"🔎 处理 {len(image_paths)} 张图 → result/ 与 compare/")

  for img_path in image_paths:
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB").resize((256, 256))
    img_np = np.transpose(np.array(img, dtype=np.float32) / 255.0, (2, 0, 1))
    img_tensor = torch.tensor(img_np).unsqueeze(0).to(device)

    with torch.no_grad():
      with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        output = model(img_tensor)

    output = torch.clamp(output, 0, 1)
    comparison = torch.cat((img_tensor, output), dim=3)

    save_image(output, os.path.join(RESULT_DIR, img_name))
    save_image(comparison, os.path.join(COMPARE_DIR, img_name))
    print(f"✅ {img_name}")

  print(f"📁 去雾结果: {RESULT_DIR}")
  print(f"📁 对比图:   {COMPARE_DIR}")


if __name__ == "__main__":
  main()
