### U-Net implementation in PyTorch

![U-Net architecture](unet.png "UNet architecture")




The original paper can be found here: https://arxiv.org/abs/1505.04597

#### Initialize the U-Net network

For a 3-channel input and 1-channel output, initialize the network as follows:

```
model = UNet(in_channels=3, out_channels=1)
```


#### Example script

```
import torch

x = torch.rand((1,3,572,572), dtype=torch.float32)
out = model(x)
print(out.size())

```
