import torch
import torch.nn as nn
import einops

class FreqDependentConv1D(nn.Module):
    """ Applies a 1D convolution across time different for every frequency.
    
    Same parameters as torch.nn.Conv1D (though I haven't tested it using different stride,
    padding, and dilation) with the addition of `freqs`, which should be equal to the
    number of frequencies (i.e, the number of independent 1D convolutions).
    
    Shape:
        - Input: [B, Cin, Tin, F]
        - Output: [B, Cout, Tout, F], where Tout is determined by the kernel size, padding, and so on.
    """
    def __init__(self, in_channels, out_channels, kernel_size,freqs=257 , stride=1, padding=0, dilation=1, 
                 bias=True, padding_mode='zeros', device=None, dtype=None):
        super().__init__()
        self.freqs = freqs
        # kernel_size=(C, kernel_size): Here C might be 1, might be in_channels
        self.grouped_conv = nn.Conv2d(in_channels=freqs, out_channels=freqs*out_channels, 
                                      kernel_size=(in_channels, kernel_size), groups=freqs,
                                      stride=(1, stride), padding=(0, padding), dilation=(1, dilation), 
                                      bias=bias, padding_mode=padding_mode, device=device, dtype=dtype)
    
    def forward(self, input):
        x = einops.rearrange(input, 'B C F T -> B F C T')
        y = self.grouped_conv(x)
        output = einops.rearrange(y, 'B (F C) 1 T -> B C F T', F=self.freqs)
        return output

class ChannelLayerNorm(nn.Module):
    """
    Applies LayerNorm over the channel dimension (C) for each time-frequency bin.
    Input shape: [B, C, T, F]
    Output shape: [B, C, T, F]
    """
    def __init__(self, C, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(C, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        # Rearrange so that channels are the last dimension
        x = einops.rearrange(x, 'B C F T -> B T F C')
        x = self.layer_norm(x)
        # Restore original shape
        x = einops.rearrange(x, 'B T F C -> B C F T')
        return x
    
# creating the class of the model
class cov_upsam(nn.Module):
    def __init__(self):
        super(cov_upsam, self).__init__()
        # adding the layers of the model
        # Feature extraction
        
        self.up0 = FreqDependentConv1D(in_channels=16, out_channels=32,kernel_size=3,stride=1,padding=1)
        self.norm0 = ChannelLayerNorm(32)
        self.up1 = nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3,stride=1,padding=1)
        self.norm1 = ChannelLayerNorm(64)
        # First upsampling (×2)
        self.up2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,stride=1,padding=1) # batch, channel,f, t
        self.norm2 = ChannelLayerNorm(128)
        # Second upsampling (×2 again, total ×4)
        self.up3 = nn.Conv2d(in_channels=128, out_channels=256,kernel_size=3,stride=1,padding=1)
        self.norm3 = ChannelLayerNorm(256)
        # upsampling (×2)
        self.up4 = nn.Conv2d(in_channels=256, out_channels=512,kernel_size=3,stride=1,padding=1)
        self.norm4 = ChannelLayerNorm(512)
        # upsampling (×2)
        self.up5 = nn.Conv2d(in_channels=512, out_channels=1024,kernel_size=3,stride=1,padding=1)
        

        self.relu = nn.ReLU(inplace=True)
        # Dropouts at deeper layers
        self.drop2 = nn.Dropout2d(p=0.2)
        self.drop3 = nn.Dropout2d(p=0.3)


    def forward(self, x):
        #print("The size of x:", x.shape, '\n')
        x = self.relu(self.norm0(self.up0(x)))
        
        x = self.relu(self.norm1(self.up1(x)))

        x = self.relu(self.norm2(self.up2(x)))
        x = self.drop2(x)  # optional dropout
        
        x = self.relu(self.norm3(self.up3(x)))
        x = self.drop3(x)
        
        x = self.relu(self.norm4(self.up4(x)))
        x = self.drop3(x)

        x = self.up5(x)

        return x
        
        #print("Input shape:", x.shape)  # [B, C, H, W]

# It checks, on dummy data, if the model works
if __name__ == '__main__':
    dummy_model = cov_upsam()
    dummy_input = torch.rand((8, 16, 257, 24))  # low res input image batch x channels x time x freq
    dummy_pred = dummy_model(dummy_input)
    print(dummy_pred.size())   # 8, 1024, 257, 24
    # Check parameter count
    total_params = sum(p.numel() for p in dummy_model.parameters())
    print(f"Total parameters: {total_params}")