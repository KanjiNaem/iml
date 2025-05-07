from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

"""
README FIRST

The below code is a template for the solution. You can change the code according
to your preferences, but the test_model function has to save the output of your
model on the test data as it does in this template. This output must be submitted.

Replace the dummy code with your own code in the TODO sections.

We also encourage you to use tensorboard or wandb to log the training process
and the performance of your model. This will help you to debug your model and
to understand how it is performing. But the template does not include this
functionality.
Link for wandb:
https://docs.wandb.ai/quickstart/
Link for tensorboard:
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")

# If you have a Mac consult the following link:
# https://pytorch.org/docs/stable/notes/mps.html

# It is important that your model and all data are on the same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")

# else:
#     device=torch.device("mps")

def _make_center_mask(tensor: torch.Tensor, invert: bool = False) -> torch.Tensor:

    mask = torch.ones_like(tensor)
    mask[:, :, 10:18, 10:18] = 0
    if invert:
        mask = 1.0 - mask
    return mask


def get_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    where N_train_samples is the number of training samples, N_test_samples is
    the number of test samples, C is the number of channels (1 for grayscale),
    H is the height of the image, and W is the width of the image.
    """
    # Load the training data
    train_data = np.load("train_data.npz")["data"]

    # Make the training data a tensor
    train_data = torch.tensor(train_data, dtype=torch.float32)

    # Load the test data
    test_data_input = np.load("test_data.npz")["data"]

    # Make the test data a tensor
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    ########################################
    # TODO: Given the original training images, create the input images and the
    # label images to train your model.
    # Replace the two placholder lines below (which currently just copy the
    # training data) with your own implementation.
    train_data_label = train_data.clone()
    train_data_input = train_data.clone()
    #set the center 8x8 pixels to black
    input_mask= _make_center_mask(train_data_input)
    train_data_input=train_data_input *input_mask


    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze(), cmap="gray")
            plt.title("Training Input")
            plt.colorbar(label="Pixel Value")
            plt.axis('on')

            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")
            plt.colorbar(label="Pixel Value")
            plt.axis('on')

            plt.tight_layout()
            plt.savefig(f"train_image_output/image_{i}.png", dpi=150)
            plt.close()

    return train_data_input, train_data_label, test_data_input


def train_model(train_data_input, train_data_label, **kwargs):
    """
    Train the model. Fill in the details of the data loader, the loss function,
    the optimizer, and the training loop.

    Args:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - model: torch.nn.Module
    """
    model = Model()
    model.train()
    model.to(device)

    # Create a custom loss function that only considers the center 8x8 patch
    def center_patch_loss(output, target):
        # Get only the center 8x8 patch (positions 10:18, 10:18)
        output_center = output[:, :, 10:18, 10:18]
        target_center = target[:, :, 10:18, 10:18]
        # Calculate MSE loss on only the center patch
        return F.mse_loss(output_center, target_center)

    criterion = center_patch_loss
    # TODO: Dummy optimizer - change this to a more suitable optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TODO: Correctly setup the dataloader - the below is just a placeholder
    # Also consider that you might not want to use the entire dataset for
    # training alone
    # (batch_size needs to be changed)
    batch_size = 128
    dataset = TensorDataset(train_data_input, train_data_label)
    # Consider the shuffle parameter and other parameters of the DataLoader
    # class (see
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    # TODO: Modify the training loop in case you need to

    # TODO: The value of n_epochs is just a placeholder and likely needs to be
    # changed
    n_epochs = 10

    for epoch in range(n_epochs):
        for x, y in tqdm(
            data_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            mask = _make_center_mask(x)          # (B,1,28,28)
            x_cat = torch.cat([x, mask], dim=1)  # (B,2,28,28)
            optimizer.zero_grad()
            output = model(x_cat)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} loss: {loss.item()}")

    return model


# TODO: define a model. Here, a basic MLP model is defined. You can completely
# change this model - and are encouraged to do so.
class Model(nn.Module):
    """
    Implement your model here.
    """

    def __init__(self):
            super().__init__()
            #encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(2, 64,   3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64,  3, stride=2, padding=1), nn.ReLU())  # 28→14
            self.enc2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128,128, 3, stride=2, padding=1), nn.ReLU())  # 14→7

            #Bottleneck with dilation
            self.dilated = nn.Sequential(
                nn.Conv2d(128,128,3, padding=2, dilation=2), nn.ReLU(),
                nn.Conv2d(128,128,3, padding=4, dilation=4), nn.ReLU())

            #Decoder
            self.up1 = nn.ConvTranspose2d(128,128,2,stride=2)   # 7→14
            self.dec1 = nn.Sequential(
                nn.Conv2d(128+128,64,3,padding=1), nn.ReLU())

            self.up2 = nn.ConvTranspose2d(64,64,2,stride=2)     # 14→28
            self.dec2 = nn.Sequential(
                nn.Conv2d(64+64,32,3,padding=1), nn.ReLU(),
                nn.Conv2d(32,1,1))  # predict full image

    def forward(self, x):
        e1 = self.enc1(x)          # (B, 64, 14, 14)   ← skip‑1
        e2 = self.enc2(e1)         # (B,128,  7,  7)   ← skip‑2

        b  = self.dilated(e2)      # (B,128,  7,  7)

        d1 = self.up1(b)           # (B,128, 14, 14)

        skip2 = F.interpolate(e2, size=d1.shape[-2:], mode="nearest")  # 7→14
        d1 = self.dec1(torch.cat([d1, skip2], dim=1))  # (B,256,14,14)

        d2 = self.up2(d1)          # (B, 64, 28, 28)
        skip1 = F.interpolate(e1, size=d2.shape[-2:], mode="nearest")  # 14→28
        out  = self.dec2(torch.cat([d2, skip1], dim=1))  # (B,1,28,28)
        return out


def test_model(model, test_data_input):
    """
    Uses your model to predict the ouputs for the test data. Saves the outputs
    as a binary file. This file needs to be submitted. This function does not
    need to be modified except for setting the batch_size value. If you choose
    to modify it otherwise, please ensure that the generating and saving of the
    output data is not modified.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise to avoid memory issues
        test_data_output = []
        # TODO: You can increase or decrease this batch size depending on your
        # memory requirements of your computer / model
        # This will not affect the performance of the model and your score
        batch_size = 128
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            batch = test_data_input[i:i+batch_size]
            mask  = _make_center_mask(batch)
            batch_cat = torch.cat([batch, mask], dim=1).to(device)
            output = model(batch_cat)
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            plt.figure(figsize=(18, 5))  # Wider figure for three subplots

            # Show the input image
            plt.subplot(1, 3, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.colorbar(label="Pixel Value")
            plt.axis('on')

            # Show the full predicted output
            plt.subplot(1, 3, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output (Full)")
            plt.colorbar(label="Pixel Value")
            plt.axis('on')

            # Create a third subplot showing only the center patch in original position
            plt.subplot(1, 3, 3)
            # Create a black image with just the center patch visible
            center_only = np.zeros_like(test_data_output[i].squeeze())
            center_only[10:18, 10:18] = test_data_output[i, 0, 10:18, 10:18]
            plt.imshow(center_only, cmap="gray")
            plt.title("Center Patch")
            plt.colorbar(label="Pixel Value")
            plt.axis('on')

            plt.tight_layout()  # Improve spacing between subplots
            plt.savefig(f"test_image_output/image_{i}.png", dpi=150)
            plt.close()


def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Load the data
    train_data_input, train_data_label, test_data_input = get_data()
    # Train the model
    model = train_model(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    test_model(model, test_data_input)

    return None


if __name__ == "__main__":
    main()