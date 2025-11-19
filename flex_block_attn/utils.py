import matplotlib.pyplot as plt
import numpy as np

def visualize_attn_mask(mask, title="Attention Mask", cmap='gray', save_path='attention_mask.png'):
    if isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        mask_np = mask.cpu().numpy()

    if mask_np.dtype == bool:
        mask_np = mask_np.astype(float)
    elif mask_np.max() > 1 or mask_np.min() < 0:
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min())

    plt.figure(figsize=(8, 8))
    plt.imshow(mask_np, cmap=cmap, interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    #plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()
