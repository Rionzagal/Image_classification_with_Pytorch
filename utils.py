import numpy as np
from matplotlib import pyplot as plt
import torch

def view_classify(img, ps):
    labels = np.array(['plane','car','bird','cat','deer','dog','frog','horse','ship','truck'])
    ps = ps.data.numpy().squeeze()
    img = img.numpy().transpose(1,2,0)
    img = img * np.array([0.4915, 0.4823, 0.4468]) + np.array([0.4915, 0.4823, 0.4468])
    img = img.clip(0,1)

    fig, (ax0, ax1) = plt.subplots(figsize=(6,9), ncols=2)
    ax0.imshow(img)
    ax0.axis('off')
    ax1.barh(labels, ps)
    ax1.set_aspect(0.1)
    ax1.set_yticks(labels)
    ax1.set_yticklabels(labels)
    ax1.set_title('Class probability')
    ax1.set_xlim(0,1)

    plt.tight_layout()
    plt.show()
    return None