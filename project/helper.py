import numpy as np
# import matplotlib.pyplot as plt

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))

    mean = np.array([0.311, 0.340, 0.299])
    std = np.array([0.167, 0.144, 0.138])

    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(3)


def cal_mean_and_std(train_loader):
    todiv = 0
    mean = [0, 0, 0]
    std = [0, 0, 0]

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.numpy()

        per_image_mean = np.mean(data, axis=(2,3))
        std += np.std(data, axis=(0,2,3))

        mean += np.mean(per_image_mean, axis=0)
        print(mean, std)
        todiv = batch_idx + 1

    mean = mean/todiv
    std = std/todiv

    return mean,std