from torch.utils.data import Dataset
import os
import torch
import glob
import torchaudio
#import soundfile as sf
import math
import numpy as np

class eigenmic(Dataset):
    def __init__(self, audio_dir, device):
        super().__init__()
        self.device = device
        fs = 24e3
        self.audio_filenames = glob.glob(os.path.join(audio_dir, '*.wav'))
        
        self.frame_length = round(0.02*fs)
        self.fft_length =  2 ** math.ceil(math.log2(self.frame_length))
        self.hop_length = int(round(self.fft_length/2))
        self.window = torch.hann_window(self.fft_length).to(device)
        self.time_frame = self.hop_length / fs
        self.maps_per_sec = 5  # Number of covariance matrices per second

        # Precompute indices for flattening/reconstruction
        self.indices_32_real = self.get_upper_triangle_indices(32, offset=0)
        self.indices_32_img = self.get_upper_triangle_indices(32, offset=1)
        self.indices_4_real = self.get_upper_triangle_indices(4, offset=0)
        self.indices_4_img = self.get_upper_triangle_indices(4, offset=1)

    def __len__(self):
        return len(self.audio_filenames)
    
    def __getitem__(self, idx):
        audio_filename = self.audio_filenames[idx]

        waveform, fs = torchaudio.load(audio_filename, frame_offset=0, num_frames=int(5 * 24000))
        data32 = waveform.to(self.device)

         # load the 32ch audio
        #data32, fs = sf.read(audio_filename)
        #data32 = torch.from_numpy(data32.T).to(self.device)

        data32_stft = torch.stft(input=data32, n_fft=self.fft_length, hop_length=self.hop_length, win_length=self.fft_length, window=self.window,
                                  center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex=True)

        # select the stft of the tetrahedron 4ch
        data4_stft = data32_stft[[5, 9, 25, 21], :, :].clone()
        
        Cx32_real, Cx32_img, Cx32_comp = self.calculate_covariance_matrix(data32_stft) # ch x ch x f x t
        Cx4_real, Cx4_img, Cx4_comp = self.calculate_covariance_matrix(data4_stft)    # ch x ch x f x t
        
        audio_name = os.path.basename(audio_filename[0]) if isinstance(audio_filename, (list, tuple)) else os.path.basename(audio_filename)
        audio_key = "_".join(audio_name.split("_")[-4:])   # the last part of the name
        audio_key = os.path.splitext(audio_key)[0] #it removes the extension eg ".wav"
        Cx_path = 'Covariance_matrices'
        
        if not os.path.exists(Cx_path):
            os.makedirs(Cx_path)
        # Full file path
        save_path4 = os.path.join(Cx_path, f"Cx4_ref_{audio_key}.npy")

        # If your data is a GPU tensor
        np.save(save_path4, Cx4_comp.cpu().numpy())
        print(f"Saved 4 channel matrix to: {save_path4} Is Cx_avg complex?", Cx4_comp.is_complex(),'\n')

        # 32 channel reference Cx
        save_path32 = os.path.join(Cx_path, f"Cx32_ref_{audio_key}.npy")

        # If your data is a GPU tensor
        np.save(save_path32, Cx32_comp.cpu().numpy())
        print(f"Saved 32 channel matrix to: {save_path32} Is Cx_avg complex?", Cx32_comp.is_complex(),'\n')

        # Flatten using correct indices
        flat_real_4 = self.flatten_covariance_matrix(Cx4_real, self.indices_4_real)
        flat_img_4 = self.flatten_covariance_matrix(Cx4_img, self.indices_4_img)
        flat_real_32 = self.flatten_covariance_matrix(Cx32_real, self.indices_32_real)
        flat_img_32 = self.flatten_covariance_matrix(Cx32_img, self.indices_32_img)

        # Merge real and imag parts along the first dimension
        Cx4_step = torch.cat((flat_real_4, flat_img_4), dim=0)    # (num_unique_real+num_unique_img, f, t)
        Cx32_step = torch.cat((flat_real_32, flat_img_32), dim=0) # (num_unique_real+num_unique_img, f, t)
        
        Cx4_comp = Cx4_comp / (Cx4_step[0] + torch.finfo(Cx4_comp.dtype).eps) #normalizing the original reference Cx4
        # TO DO: Optional normalization 
        Cx32_norm = Cx32_step/(Cx4_step[0] + torch.finfo().eps)
        Cx4_norm = Cx4_step/(Cx4_step[0] + torch.finfo().eps)
        return Cx4_norm, Cx32_norm, audio_filename, Cx4_comp

    def calculate_covariance_matrix(self, stft_data):
        data_stft_ch_last = stft_data.permute(2, 1, 0)  # t x f x ch
        temp1 = data_stft_ch_last.unsqueeze(-1)  # t x f x ch x 1
        temp2 = data_stft_ch_last.unsqueeze(-2)  # t x f x 1 x ch
        Cx = torch.matmul(temp1, torch.conj(temp2))  # t x f x ch x ch
        Cx = Cx.permute(2, 3, 1, 0)  # ch x ch x f x t

        # Time averaging
        ch, ch2, f, t = Cx.shape
        step = round(1 / (self.maps_per_sec * self.time_frame))
        num_chunks = math.floor(t / step)
        if num_chunks == 0:
            raise ValueError("Not enough time frames for the requested maps_per_sec.")
        Cx = Cx[:, :, :, :step * num_chunks].clone()
        Cx_reshaped = Cx.view(ch, ch2, f, num_chunks, step)
        Cx_avg = Cx_reshaped.mean(dim=-1).clone()  # ch x ch x f x num_chunks

        Cx_real = Cx_avg.real
        Cx_img = Cx_avg.imag
        #print("Is Cx_avg complex?", Cx_avg.is_complex())
        return Cx_real, Cx_img, Cx_avg
    
    def flatten_covariance_matrix(self, Cx, indices):
        """
        Flattens the upper triangle (with or without diagonal) using precomputed indices.
        Args:
            Cx: (ch, ch, f, t)
            indices: (row_idx, col_idx)
        Returns:
            flat: (num_unique, f, t)
        """
        row_idx, col_idx = indices
        flat = Cx[row_idx, col_idx, :, :]
        return flat

    def reconstruct_covariance_matrix_from_concat(self, flat, indices_real, indices_img, ch):
        """
        Reconstructs the full (Hermitian) matrix from concatenated flat input (real+imag).
        Args:
            flat: (num_unique_real + num_unique_img, f, t) - concatenated real and imag parts
            indices_real: (row_idx, col_idx) for real part (with diagonal)
            indices_img: (row_idx, col_idx) for imag part (without diagonal)
            ch: number of channels (4 or 32)
        Returns:
            Cx_real, Cx_img: (ch, ch, f, t)
        """
        f = flat.shape[1]
        t = flat.shape[2]
        num_real = indices_real[0].shape[0]
        num_img = indices_img[0].shape[0]
        flat_real = flat[:num_real, :, :]
        flat_img = flat[num_real:num_real+num_img, :, :]

        # Real part
        Cx_real = torch.zeros((ch, ch, f, t), device=flat.device)
        row_idx_r, col_idx_r = indices_real
        Cx_real[row_idx_r, col_idx_r, :, :] = flat_real
        Cx_real[col_idx_r, row_idx_r, :, :] = flat_real

        # Imaginary part
        Cx_img = torch.zeros((ch, ch, f, t), device=flat.device)
        row_idx_i, col_idx_i = indices_img
        Cx_img[row_idx_i, col_idx_i, :, :] = flat_img
        Cx_img[col_idx_i, row_idx_i, :, :] = -flat_img

        return Cx_real, Cx_img

    @staticmethod
    def get_upper_triangle_indices(ch, offset=0):
        """
        Returns the upper triangle indices (row_idx, col_idx) for a ch x ch matrix.
        offset=0: includes diagonal (for real part)
        offset=1: excludes diagonal (for imaginary part)
        """
        return torch.triu_indices(ch, ch, offset=offset)
            

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    #train_path = "32ch_audios"
    train_path = "../dataset/eigen_dev_train_splits"
    #device = torch.device("cuda:0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = eigenmic(train_path, device)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    for i, (cx4, cx32, audio_file) in enumerate(train_loader):
        print(cx4.shape)
        print(cx32.shape)
        print('the audio files are:', audio_file,'\n')
        break
