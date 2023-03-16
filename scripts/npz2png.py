import os, json, sys, numpy as np
from torchvision.transforms.functional import to_tensor, resize
from torchvision.utils import save_image

results = np.load(sys.argv[1])['arr_0']
out_path = os.path.splitext(sys.argv[1])[0]
dst_wh = None
if len(sys.argv) > 2:
	dst_wh = json.loads(sys.argv[2])

for idx, img in enumerate(results):
	tensor = to_tensor(img)
	if dst_wh:
		tensor = resize(tensor, [dst_wh[1], dst_wh[0]])
	save_image(tensor, f'{out_path}_{idx:02d}.png')