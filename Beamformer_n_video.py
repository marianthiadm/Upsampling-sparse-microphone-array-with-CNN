import torch
import math
from data_generator import eigenmic

#Beamforming
def Beam(Cx, src_dirs, H_mic):
    ch, ch, f, t = Cx.shape

    if ch == 4:
        m = [5,9,25,21]
        H_mic = H_mic[:,m,:].clone()

    P = torch.empty(len(src_dirs), f , t, dtype=torch.complex64, device=device)

    H_mic = H_mic.detach().clone()
    H_right= H_mic.permute(1,2,0).clone() # [ch, d, f]
    H_left = H_right.conj().clone()
    Cxn = Cx.permute(2,0,1,3).clone() # [f, ch, ch, t]
    Cxn = Cxn.to(torch.complex128)
    
    for times in range(t):
        print("times is", times, '\n')
        cxs = Cxn[:,:,:,times] # dir x f x ch x ch
        aL = einops.rearrange(H_left, 'ch d f -> d f 1 ch').clone() 
        aR = einops.rearrange(H_right, 'ch d f -> d f ch 1')
        temp1 = torch.matmul(aL, cxs).clone()
        temp2 = torch.matmul(temp1, aR).clone()
        P[:,:,times] = temp2.squeeze().clone() # [d, f, t]

    return P, ch, f, t


if __name__=='__main__':
    import torch
    import numpy as np
    import os 
    import glob
    import einops
    import re
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.io as sio
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import subprocess
    import shutil
    import subprocess
    from overlay import data_for_overlay

    # Device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    BASE = os.path.dirname(os.path.abspath(__file__))
    video_path  = os.path.join(BASE, "..", "dataset", "Cx_videos")
    audio_path  = os.path.join(BASE, "..", "dataset", "Cx_data")
    # Validate:
    if not os.path.isdir(video_path):
        raise FileNotFoundError(f"Video folder does not exist: {video_path}")

    if not os.path.isdir(audio_path):
        raise FileNotFoundError(f"Audio folder does not exist: {audio_path}")
       
    # Create 1D grids like MATLAB linspace
    phi_grid = torch.linspace(-math.pi, math.pi, steps=141)
    theta_grid = torch.linspace(-math.pi / 2, math.pi / 2, steps=51)
    phi_deg = phi_grid * 180 / np.pi
    theta_deg = theta_grid * 180 / np.pi

    # Create meshgrid 
    theta, phi = torch.meshgrid(theta_deg, phi_deg, indexing='ij') 

    # Flatten the 2D grids into 1D column vectors
    src_dirs_phi = phi.reshape(-1, 1)
    src_dirs_theta = theta.reshape(-1, 1)

    # Concatenate into a 2D tensor with two columns
    src_dirs = torch.cat((src_dirs_phi, src_dirs_theta), dim=1)
    d = len(src_dirs)

    matt = sio.loadmat('transferFunc32.mat')
    H_mic_np = matt['H_mic']  # numpy array, shape preserved
    H_mic = torch.tensor(H_mic_np).to(device) 
    #print("The H_mic is", H_mic.shape, '\n')
    
    # Load NumPy array
    Cx_path = "Covariance_matrices"
    
    # Get all .npy files in the folder
    file = [f for f in os.listdir(Cx_path) if f.endswith(".npy")]
    # Sort by the number in the filename (so matrix2 comes before matrix10)
    file.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    #print("the file name is", file,'\n')
    Cx = {}

    for fname in file:
        path = os.path.join(Cx_path, fname)
        arr = np.load(path)
        tensor = torch.from_numpy(arr).to(device)  # move to device
        Cx[fname] = tensor
        match = re.search(r'(\d+)(?=\.npy$)', fname)
        if not match:
            continue  # skip files with no numbers in name
        num = int(match.group())
      
        # Create variables for current file
        Cx_name = fname #Cx_name contains the file name (audiofile and num of channels)
        Cx_number = num #Cx_number contains the num of the split to know the segment

        # Load matrix
        path = os.path.join(Cx_path, fname)
        arr = np.load(path)
        Cx_tensor = torch.from_numpy(arr).to(device)

        filename = "_".join(Cx_name.split("_")[-3:]) 
        print("the file name is", filename,'\n')
        audio_name = os.path.splitext(filename)[0]

        # video
        video_base = re.sub(r"_part\d+$", "", audio_name)
        video_file = os.path.join(video_path, f"{video_base}.mp4")
        print("Absolute video path:", os.path.abspath(video_file))
        print("the audio path is", audio_path, '\n')
        print("audio name with the split is", audio_name,'\n')

        # Print current info
        print(f"Loaded: {Cx_name} | number: {Cx_number} | shape: {Cx_tensor.shape}\n")

        P, ch, f, t = Beam(Cx_tensor, src_dirs, H_mic)

        #Plotting beamformed map
        # Choose f index to plot if wanted
        #f_idx = 150
        va_avg_f = P.mean(dim=(1)) #averaged across frequencies
        va_mag = va_avg_f.abs()
        va_mag_np = va_mag.cpu().numpy() 
        # Global min and max across all directions and times
        vmin = np.min(va_mag_np)
        vmax = np.max(va_mag_np)
       
        # Extract the slice for frequency f_idx and time t_idx
        n_theta = 51  # as in your theta_grid
        n_phi = 141   # as in your phi_grid
        t0 = 12
        # we averaged only over frequency (still have time axis)
        # To plot for a specific time index t0:
        va_plot = va_mag_np[:, t0].reshape(n_phi, n_theta).T  # 2D grid for contour

        output = data_for_overlay(Cx_name, va_mag_np, audio_name, n_phi, n_theta, Cx_number, 
                                  video_file, audio_name, plot_fps=5, overlay_position="topright")

        '''plt.figure(figsize=(10, 5))
        cp = plt.contourf(phi, theta, va_plot, levels=50, cmap='jet')  # phi, theta are 2D grids
        plt.xlabel('Phi (deg)')
        plt.ylabel('Theta (deg)')
        plt.title(f'Beamforming averaged over frequency {Cx_name}')
        plt.colorbar(cp, label='P')
        plt.gca().invert_xaxis()  # reverses the x-axis
        plt.show()'''

        # Convert degrees to radians
        '''phi_rad = phi * np.pi / 180
        theta_rad = theta * np.pi / 180
        # Convert spherical to Cartesian coordinates for plotting
        R = 1  # radius of the sphere
        X = R * np.cos(theta_rad) * np.cos(phi_rad)
        Y = R * np.cos(theta_rad) * np.sin(phi_rad)
        Z = R * np.sin(theta_rad)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Normalize VA for facecolors
        norm = (va_plot - va_plot.min()) / (va_plot.max() - va_plot.min())
        colors = plt.cm.viridis(norm)
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1)

        mappable = ScalarMappable(cmap='jet', norm=Normalize(vmin=va_plot.min(), vmax=va_plot.max()))
        mappable.set_array([])  # ScalarMappable needs an array, but can be empty

        # Attach the colorbar explicitly to the Axes
        fig.colorbar(mappable, ax=ax, shrink=0.7, label='P')
        ax.set_title(f'2nd plot for {Cx_name}, P is averaged over frequency at t={t0}')
        plt.show()

        plt.figure(figsize=(10, 5))
        ax = plt.subplot(111, projection='mollweide')
        phi_rad = phi * np.pi / 180
        theta_rad = theta * np.pi / 180
        cp = ax.pcolormesh(phi_rad, theta_rad, va_plot, shading='auto', cmap='jet')
        plt.colorbar(cp, orientation='horizontal', label='P')
        ax.set_title(f'3rd plot for {Cx_name}, P is averaged over frequency at t={t0} (Mollweide)')
        plt.grid(True)
        plt.show()

        t = 0
        for t in range(va_mag_np.shape[1]):
            va_plot = va_mag_np[:, t].reshape(n_phi, n_theta).T
            plt.figure(figsize=(10, 5))
            ax = plt.subplot(111, projection='mollweide')
            cp = ax.pcolormesh(phi, theta, va_plot, shading='auto', cmap='jet')
            plt.colorbar(cp, orientation='horizontal', label='P')
            ax.set_title(f'4th plot for {Cx_name} per Time frame {t}')
            plt.grid(True)
            plt.show()'''

