from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from data_generator import eigenmic
from model_cov_matrix_norm_1FDC_expanded_k3 import cov_upsam
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
from collections import defaultdict
import os
import numpy as np
import pandas as pd

# Paths to folders
test_path = "../dataset/eigen_dev_test_splits"

# Device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset and loader
test_dataset = eigenmic(test_path, device)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# it pushes the model to the device (cpu, gpu)
model = cov_upsam().to(device)

# If your model is on CPU
model.load_state_dict(torch.load("best_model_1FDC_beyond_k3.pth", map_location=device)["net"])

# prepares the model for predictions
model.eval()

# Loss
test_loss_fn = nn.MSELoss().to(device)
test_loss = 0
rmse__dir = "RMSE_test_loss"
if not os.path.exists(rmse__dir):
    os.makedirs(rmse__dir)

################################ ~ Cx plotting ~ #####################################################################################################
def plot_cov_matrices(audiofile, Cx4, test_dataset, label_flat, pred_flat, indices_real, indices_img, ch, outdir="cov_seg_plots", title_prefix=""):
    label_real, label_img = test_dataset.reconstruct_covariance_matrix_from_concat(label_flat, indices_real, indices_img, ch)
    pred_real, pred_img = test_dataset.reconstruct_covariance_matrix_from_concat(pred_flat, indices_real, indices_img, ch)
    Cx32_ref = torch.complex(label_real, label_img).cpu().numpy()
    Cx32_pred = torch.complex(pred_real,pred_img).cpu().numpy()
    label_real_avg = label_real.mean(dim=(2, 3)).cpu().numpy()
    label_img_avg = label_img.mean(dim=(2, 3)).cpu().numpy()
    pred_real_avg = pred_real.mean(dim=(2, 3)).cpu().numpy()
    pred_img_avg = pred_img.mean(dim=(2, 3)).cpu().numpy()

    print("Is Cx32 predicted a complex matrix?", np.iscomplexobj(Cx32_pred))

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    comp_dir = 'FDCsqr_plots'
    if not os.path.exists(comp_dir):
        os.makedirs(comp_dir)

    Cx_path = 'Covariance_matrices'
    if not os.path.exists(Cx_path):
        os.makedirs(Cx_path)
    # Full file path
    save_path = os.path.join(Cx_path, f"Cx32_pred_{audiofile}.npy")

    # If your data is a GPU tensor
    np.save(save_path, Cx32_pred)

    print(f"Saved 32 channel predicted matrix to: {save_path}")
    print("The prediction of 32 is", Cx32_pred.shape,'\n')
    # Use log scale for color mapping
    eps = 1e-8
    # Compute 3 subplot of the reversed triangulated Cx matrices
    matrices = [Cx4, Cx32_ref, Cx32_pred] # List of matrices to plot

    # Compute log10(abs) and reduce 4D -> 2D if needed
    matrices_log = [
        np.log10(np.abs(m[:, :, 50, 12]) + eps) if m.ndim == 4 else np.log10(np.abs(m) + eps)
        for m in matrices
    ]

    # Shared color scale
    vmin = min(m.min() for m in matrices_log)
    vmax = max(m.max() for m in matrices_log)
  
    # Titles for each subplot
    titles = [f"Cx4",
            f"Reference Cx32",
            f"Predicted Cx32"]

    # Create figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = np.atleast_1d(axs)

    # Plot all matrices
    ims = [ax.imshow(mat_log, vmin=vmin, vmax=vmax)
        for ax, mat_log in zip(axs, matrices_log)]

    for ax, title in zip(axs, titles):
        ax.set_title(f"{title}", fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

    # Make common colorbar outside the plots (to the right)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=14)

    fname = os.path.join(comp_dir, f"{title_prefix}cov_seg_log_{audiofile}.png")
    print("fname is: ", fname, '\n')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
    plt.savefig(fname)
    plt.close(fig)
    
    ####################### ~ End of the Cx plotting ~ ###########################################################################################

# Dictionary: group -> list of losses
#group_train_losses = defaultdict(list)
# containers
group_train_docs = defaultdict(list)
file_train_losses = {}
group_test_losses = defaultdict(list)
group_test_docs = defaultdict(list)
file_test_losses = {}
all_test_rmse_losses = []
predtest = []
groundtest = []
all_pred_gt = []
test_rmse_loss = torch.zeros(1024)

with torch.no_grad():
    for j, (test_cov_4ch, test_cov_32ch, audiofile, Cx4) in enumerate(test_loader):
        # move to device
        test_cov_4ch, test_cov_32ch = test_cov_4ch.to(device).float(), test_cov_32ch.to(device).float()
        test_pred_cov = model(test_cov_4ch)
        loss = test_loss_fn(test_pred_cov, test_cov_32ch).item()
        # Extract per-channel predicted and GT values 
        pred_mean = torch.mean(test_pred_cov, dim=(0, 2, 3)).cpu().numpy()   # shape [32]
        gt_mean   = torch.mean(test_cov_32ch, dim=(0, 2, 3)).cpu().numpy()    # shape [32]

        all_pred_gt.append(
            np.concatenate([pred_mean, gt_mean]).tolist()
        )
        rmse_per_column = torch.sqrt(torch.mean((test_pred_cov - test_cov_32ch) ** 2, dim=(0, 2, 3)))  # shape [32]
        # Accumulate RMSE per column
        test_rmse_loss += rmse_per_column.cpu()   # works because both are tensors of shape [32]

        # Convert to list to save per batch
        rmse_list = rmse_per_column.cpu().detach().numpy().tolist()
        all_test_rmse_losses.append(rmse_list)

        ####### ~ Cx plotting ~ ####################################
        audio_name = os.path.basename(audiofile[0]) if isinstance(audiofile, (list, tuple)) else os.path.basename(audiofile)
        audio_key = "_".join(audio_name.split("_")[-4:])   # the last part of the name
        audio_key = os.path.splitext(audio_key)[0] #it removes the extension eg ".wav"
        print("the audio key is", audio_key,'\n')
        indices_real = test_dataset.indices_32_real
        indices_img = test_dataset.indices_32_img
        ch = 32
        Cx4 = Cx4.detach().cpu().numpy()
        Cx4 = Cx4.squeeze(0)
        plot_cov_matrices(
            audio_key, Cx4, test_dataset, test_cov_32ch[0], test_pred_cov[0],
            indices_real, indices_img, ch, title_prefix="32ch_test_"
        )
        ######### ~ End plotting ~ ###################################################
        predtest.append(test_pred_cov.detach().cpu().float().mean().item())
        groundtest.append(test_cov_32ch.detach().cpu().float().mean().item())

        # extract group key from filename (e.g., "folder1_file1" from "folder1_file1_doc3")
        fname = os.path.basename(audiofile[0]) if isinstance(audiofile, (list, tuple)) else os.path.basename(audiofile)
        key = "_".join(fname.split("_")[:-1])   # everything except the last part (_docX)

        # accumulate loss for this group
        file_test_losses[fname] = loss
        group_test_losses[key].append(loss)
        group_test_docs[key].append(fname)

    # Compute averages per group
    group_avg_test_losses = {key: sum(vals)/len(vals) for key, vals in group_test_losses.items()}

    # Print results
    for key, avg_loss in group_avg_test_losses.items():
        print('test loss:', f"{key}: {avg_loss:.4f}")

    # Find top 10 highest losses
    top_n = 10
    test_highest = sorted(group_avg_test_losses.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    print("\nTop 10 groups with highest test loss:")
    for key, val in test_highest:
        print(f"{key}: {val:.4f}")

        for doc_name in group_test_docs[key]:
            doc_test_loss = file_test_losses[doc_name]
            print(f"   - Document: {doc_name}, Test Loss: {doc_test_loss:.4f}")

    # Stack the columns horizontally into a 2D array
    '''data = np.column_stack((all_test_rmse_losses, groundtest, predtest))
    filename = "RMSEloss.csv"
    filepath = os.path.join(rmse__dir, filename)
    # Save as CSV
    np.savetxt(filepath, data, delimiter=",", fmt="%.6f", header="rmseloss_test, ground truth, predicted", comments="")'''

    df = pd.DataFrame(all_test_rmse_losses, columns=[f'col_{i+1}' for i in range(len(test_rmse_loss))])
    df.to_csv('RMSE_test_loss/rmse_per_column.csv', index=False)

    columns = [f'pred_ch{i+1}' for i in range(len(test_rmse_loss))] + \
            [f'gt_ch{i+1}'   for i in range(len(test_rmse_loss))]
    df_pg = pd.DataFrame(all_pred_gt, columns=columns)
    df_pg.to_csv("RMSE_test_loss/gtn'pred_per_column.csv", index=False)
    # Plot
    x = list(group_avg_test_losses.keys())
    y = list(group_avg_test_losses.values())
    x_numeric = range(len(x))

    plt.scatter(x_numeric, y, color='green', marker='*')

    # Annotate the top-N highest losses
    for key, val in test_highest:
        idx = x.index(key)   # find its x-position
        plt.text(idx, val, f"{key}\n{val:.3f}", 
                ha='center', va='bottom', fontsize=8, color="red")

    plt.xticks(x_numeric, x, rotation=45, ha='right')
    plt.xlabel("File Group")
    plt.ylabel("Average Test Loss")
    plt.title("Average Test Loss per File Group")
    plt.tight_layout()
    plt.savefig("Average Test Loss Groups.jpeg", bbox_inches="tight")
    plt.show()
    plt.close()

    #RMSE loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(all_test_rmse_losses) +1), all_test_rmse_losses, label='RMSE Loss')
    plt.xlabel(f'{audio_key}')
    plt.ylabel('RMSE Loss')
    plt.title('RMSE Loss 4 to 32 ch')
    plt.legend()
    plt.grid(True)
    # Save the combined plot
    plt.savefig("RMSE_loss.jpeg", bbox_inches="tight")
    plt.show()
    plt.clf()

