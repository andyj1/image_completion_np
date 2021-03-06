import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_latent(tsne, context_mu, batch_label, ax, params):
    batch_size = params.batch_size
    r_dim = params.r_dim
    z_dim = params.z_dim
    epoch = params.epoch
    iter = params.iter
    
    x_train = tsne.fit_transform(context_mu.cpu().detach().numpy())
    x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
    ax.scatter(x_train[:,0], x_train[:,1], s=10, c=batch_label.numpy())
    ax.set_xlabel('t-sne axis 1')
    ax.set_ylabel('t-sne axis 2')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_title(params.save_path[:-4])
    # plt.axis("off")
    # plt.legend(loc='best', markerscale=0.5, scatterpoints=1, fontsize=10)
    plt.savefig(params.save_path)

    ax.clear()

# def plot_completion(mnist_np, img, x, y, n_pixels, epoch, batch):
#     npoints = [10, 50, 250]
    
#     # Normalize img to [0..1]
#     img = (img - img.min())
#     img = img / img.max()
    
#     fig, axes = plt.subplots(3, len(npoints), squeeze=False)
    
#     for i, n in enumerate(npoints):
#         mask = torch.zeros(n_pixels, dtype=torch.bool)
#         mask[torch.randperm(n_pixels)[:n]] = 1

#         x_ctx = x[mask]
#         y_ctx = y[mask]

#         loss, kld, mus, sigmas, logits = mnist_np(x_ctx, y_ctx, x)
#         mu = mus['target']
#         sigma = sigmas['target']
#         mask = mask.numpy().astype(np.bool)
        
#         # Show observed points
#         masked_img = np.tile(img.clone().numpy()[:, :, np.newaxis], (1, 1, 3))
#         masked_img[~mask] = [0.0, 0.0, 0.5]
#         axes[0][i].imshow(masked_img.reshape(28, 28, 3))
        
#         # Show the mean
#         mean_img = np.clip(np.tile(mu.detach().numpy(), (1, 1, 3)), 0, 1)
#         axes[1][i].imshow(mean_img.reshape(28, 28, 3))
        
#         # Show the standard deviation
#         std_img = np.clip(np.tile(sigma.detach().numpy(), (1, 1, 3)), 0, 1)
#         axes[2][i].imshow(std_img.reshape(28, 28, 3))
        
#         axes[0][i].set_title("{} observed points".format(n))
#         for j in range(3):
#             axes[j][i].axis("off")
#     plt.savefig(f'./fig/Epoch_{epoch+1}_Batch_{batch+1}.png')
#     # plt.show()
#     # plt.close()
