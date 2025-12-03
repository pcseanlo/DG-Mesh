from PIL import Image
import sys
import os
import numpy as np
from tqdm import tqdm

masks = os.listdir(sys.argv[1])
for _, mask_file in tqdm(enumerate(masks)):
    if not mask_file.endswith('.png'):
        continue
    mask_path = os.path.join(sys.argv[1], mask_file)
    try:
      mask_img = Image.open(mask_path)  # Convert to grayscale
      mask_data = np.array(mask_img)
      mask_data = mask_data[:, :, :3] * mask_data[:, :, 3:4]
      mask_data_valid = mask_data.sum(axis=2) > 0
      mask_data = mask_data_valid.astype(np.uint8) * 255
      mask_img = Image.fromarray(mask_data.astype(np.uint8))
      save_path = os.path.join(sys.argv[1], f'{mask_file}')
      mask_img.save(save_path)
    except Exception as e:
      print(f"Error processing {mask_file}: {e}")