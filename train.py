from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from data_generator import eigenmic
from model_cov_matrix_2FDC_k3 import cov_upsam
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import math
import os

def plot_cov_matrices(test_dataset, label_flat, pred_flat, indices_real, indices_img, ch, epoch, outdir="cov_plots", title_prefix=""):
    label_real, label_img = test_dataset.reconstruct_covariance_matrix_from_concat(label_flat, indices_real, indices_img, ch)
    pred_real, pred_img = test_dataset.reconstruct_covariance_matrix_from_concat(pred_flat, indices_real, indices_img, ch)
    label_real_avg = label_real.mean(dim=(2, 3)).cpu().numpy()
    label_img_avg = label_img.mean(dim=(2, 3)).cpu().numpy()
    pred_real_avg = pred_real.mean(dim=(2, 3)).cpu().numpy()
    pred_img_avg = pred_img.mean(dim=(2, 3)).cpu().numpy()

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Use log scale for color mapping
    eps = 1e-8
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    im0 = axs[0, 0].imshow(np.log10(np.abs(label_real_avg) + eps))
    axs[0, 0].set_title(f"{title_prefix}Label Real (log10 abs avg)")
    fig.colorbar(im0, ax=axs[0, 0])
    im1 = axs[0, 1].imshow(np.log10(np.abs(pred_real_avg) + eps))
    axs[0, 1].set_title(f"{title_prefix}Pred Real (log10 abs avg)")
    fig.colorbar(im1, ax=axs[0, 1])
    im2 = axs[1, 0].imshow(np.log10(np.abs(label_img_avg) + eps))
    axs[1, 0].set_title(f"{title_prefix}Label Imag (log10 abs avg)")
    fig.colorbar(im2, ax=axs[1, 0])
    im3 = axs[1, 1].imshow(np.log10(np.abs(pred_img_avg) + eps))
    axs[1, 1].set_title(f"{title_prefix}Pred Imag (log10 abs avg)")
    fig.colorbar(im3, ax=axs[1, 1])
    plt.tight_layout()
    fname = os.path.join(outdir, f"{title_prefix}cov_epoch_{epoch+1}_log.png")
    #plt.savefig(fname)
    plt.close(fig)
    print(f"Saved covariance plot (log scale): {fname}")

def main():
    train_path = "../dataset/eigen_dev_train_splits"
    test_path = "../dataset/eigen_dev_test_splits"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataset = eigenmic(train_path, device)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_dataset = eigenmic(test_path, device)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = cov_upsam().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss().to(device)
    epochs = 100

    tr_history = []
    te_history = []
    all_train_losses = []
    all_test_losses = []

    start_epoch=0
    if pre_trained is not None:
        ckpt = torch.load(pre_trained)
        model.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['opt'])
        start_epoch = ckpt['epoch'] + 1

    for epoch in range(start_epoch, epochs):
        # Training
        model.train()
        train_epoch_loss = 0
        for i, (train_4ch_cov, train_32ch_cov, audio_filename) in enumerate(train_loader):
            print(i)
            train_4ch_cov, train_32ch_cov = train_4ch_cov.to(torch.float32).to(device), train_32ch_cov.to(torch.float32).to(device)
            pred_32ch_cov = model(train_4ch_cov)
            loss_train = loss_fn(pred_32ch_cov, train_32ch_cov)
            optimizer.zero_grad()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, norm_type=2.0, error_if_nonfinite=False, foreach=None)
            optimizer.step()
            train_epoch_loss += loss_train.item()
            all_train_losses.append(loss_train.item())
        avg_train_loss = train_epoch_loss / len(train_loader)
        tr_history.append(avg_train_loss)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f}")
    
        # Testing 
        if epoch % 10 == 0 or (epoch+1) == epochs:
            model.eval()
            test_epoch_loss = 0
            te_loss = np.zeros(epochs)
            with torch.no_grad():
                for i, (test_4ch_cov, test_32ch_cov, audio_filename) in enumerate(test_loader):
                    print("Test epoch is", epoch,'\n')
                    test_4ch_cov = test_4ch_cov.to(torch.float32).to(device)
                    test_32ch_cov = test_32ch_cov.to(torch.float32).to(device)
                    pred_32ch_cov = model(test_4ch_cov)
                    loss_test = loss_fn(pred_32ch_cov, test_32ch_cov)
                    test_epoch_loss += loss_test.item()
                    all_test_losses.append(loss_test.item())
                    if i == 0:
                        print("the audio file name of the cov plot is:", audio_filename[0],'\n')
                        indices_real = test_dataset.indices_32_real
                        indices_img = test_dataset.indices_32_img
                        ch = 32
                        plot_cov_matrices(
                            test_dataset, test_32ch_cov[0], pred_32ch_cov[0],
                            indices_real, indices_img, ch, epoch, title_prefix="32ch_"
                        )
            avg_test_loss = test_epoch_loss / len(test_loader)
            te_history.append(avg_test_loss)
            print(f"Epoch {epoch+1} Test Loss: {avg_test_loss:.6f}")
            
            # printhing the number of epochs and the average test loss value
            print(f"Epoch [{epoch+1}/{epochs}], Avg test Loss: {test_epoch_loss/(i+1):.4f}")
            te_loss[epoch] = test_epoch_loss/(i+1)
            #print('test avg loss', te_loss[epoch],'\n')
    
            
            net_save = {'net': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
            
            # This line activates for model_cov_matrix_2FDC_k3
            torch.save(net_save, "best_model_2FDC_k3.pth")

        else:
            try:
                te_history.append(avg_test_loss) # dummy append to simply not change the graph code
            except UnboundLocalError:
                pass

    # Plot train and test loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs+1), tr_history, label='Train Loss')
    plt.plot(range(1, epochs+1), te_history, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()
    plt.grid(True)
    # Save the combined plot
    plt.savefig("train_vs_test_loss_2FDC_k3.jpeg", bbox_inches="tight")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    pre_trained = None # add the path to your saved model here. example: pre_trained='best_model_2FDC_k3.pth"
    main()






