from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
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

# torch.set_num_threads(4)
# torch.set_num_interop_threads(8)
# print(torch.get_num_threads(), torch.get_num_interop_threads())

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
    
    for img in train_data_input:
        img.data[0, 10:18, 10:18] = 0
    
    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze(), cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")

            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input

def stitch_output(input, output):
    return output.clone().clamp(0, 255)
    combined = input.clone()
    combined[:, :, 10:18, 10:18] = output.clone()
    return combined.clamp(0, 255)

def make_center_mask(tensor: torch.Tensor, invert: bool = False) -> torch.Tensor:
    mask = torch.ones_like(tensor)
    mask[:, :, 10:18, 10:18] = 0
    if invert:
        mask = 1.0 - mask
    return mask

def center_patch_loss(output, target):
        # Get only the center 8x8 patch (positions 10:18, 10:18)
        output_center = output[:, :, 10:18, 10:18]
        target_center = target[:, :, 10:18, 10:18]
        # Calculate MSE loss on only the center patch
        return F.mse_loss(output_center, target_center)

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
    print(model)
    model.train()
    model.to(device)

    # TODO: Dummy criterion - change this to the correct loss function
    # https://pytorch.org/docs/stable/nn.html#loss-functions
    criterion = center_patch_loss
    # TODO: Dummy optimizer - change this to a more suitable optimizer
    optimizer = torch.optim.Adam(model.parameters())

    # TODO: Correctly setup the dataloader - the below is just a placeholder
    # Also consider that you might not want to use the entire dataset for
    # training alone
    # (batch_size needs to be changed)
    batch_size = 1000
    dataset = TensorDataset(train_data_input, train_data_label)
    # Consider the shuffle parameter and other parameters of the DataLoader
    # class (see
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    # TODO: Modify the training loop in case you need to

    # TODO: The value of n_epochs is just a placeholder and likely needs to be
    # changed
    n_epochs = 30

    for epoch in range(n_epochs):
        for x, y in tqdm(
            data_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            # mask = make_center_mask(x)
            # x = torch.cat([x, mask], dim=1)
            optimizer.zero_grad()
            output = stitch_output(x, model(x))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} loss: {loss.item()}") # type: ignore

    return model

def cross_validate(dataset, n_splits, n_epochs):
    """
    Perform cross-validation on the model using the given dataset.

    Args:
    - model: torch.nn.Module
    - dataset: TensorDataset
    - n_splits: int, number of splits for cross-validation

    Returns:
    - None
    """

    kf = KFold(n_splits=n_splits)

    for i, (train_index, val_index) in enumerate(kf.split(dataset)):
        train_dataset = Subset(dataset, train_index.tolist())
        val_dataset = Subset(dataset, val_index.tolist())

        # Create data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

        model = Model()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()

        # Train the model on the training set
        model.train()
        for epoch in range(n_epochs):
            for x, y in tqdm(train_loader, desc=f"Training {epoch}", leave=False):
                x, y = x.to(device), y.to(device)
                # mask = make_center_mask(x)
                # x = torch.cat([x, mask], dim=1)
                optimizer.zero_grad()
                output = stitch_output(x, model(x))
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
        # Validate the model on the validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Validating", leave=False):
                x, y = x.to(device), y.to(device)
                # mask = make_center_mask(x)
                # x = torch.cat([x, mask], dim=1)
                output = stitch_output(x, model(x))
                loss = criterion(output, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Fold {i} validation loss: {val_loss}")

# TODO: define a model. Here, a basic MLP model is defined. You can completely
# change this model - and are encouraged to do so.
class Model(nn.Module):
    """
    Implement your model here.
    """

    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()

        # self.norm1 = nn.BatchNorm2d(1)
        # 1*28*28
        # self.conv1 = nn.Conv2d(
        #     in_channels=1,
        #     out_channels=8,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )
        # # 8*14*14
        # self.pool1 = nn.MaxPool2d(
        #     kernel_size=2,
        #     stride=2
        # )
        # # self.norm2 = nn.BatchNorm2d(8)
        # # 8*7*7
        # self.conv2 = nn.Conv2d(
        #     in_channels=8,
        #     out_channels=64,
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )
        # # 64*4*4
        # self.pool2 = nn.MaxPool2d(
        #     kernel_size=4,
        #     stride=4
        # )

        # n0 = 784
        # n1 = 500
        # n2 = 300
        # n3 = 100
        # n4 = 200
        # n5 = 784

        # self.norm1 = nn.BatchNorm1d(n0)
        # self.fc1 = nn.Linear(n0, n1)

        # self.norm2 = nn.BatchNorm1d(n1)
        # self.fc2 = nn.Linear(n1, n2)

        # self.norm3 = nn.BatchNorm1d(n2)
        # self.fc3 = nn.Linear(n2, n3)

        # self.norm4 = nn.BatchNorm1d(n3)
        # self.fc4 = nn.Linear(n3, n4)

        # self.norm5 = nn.BatchNorm1d(n4)
        # self.fc5 = nn.Linear(n4, n5)

        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            nn.Linear(128 * 7 * 7, 784),
            nn.ReLU(),
        )

        # self.dilate = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=4, dilation=4),
        #     nn.ReLU(),
        # )

        # self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        # self.dec1 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        # self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # self.dec2 = nn.Sequential(
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, kernel_size=1),
        # )

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # Flatten the image in the last two dimensions
        # x = x.view(x.shape[0], 1, 28, 28)

        # # x = self.norm1(x)
        # x = self.conv1(x)
        # x = F.leaky_relu(x)
        # x = self.pool1(x)

        # # x = self.norm2(x)
        # x = self.conv2(x)
        # x = F.leaky_relu(x)
        # x = self.pool2(x)

        # x = x.reshape(x.shape[0], -1)

        # x = self.norm1(x)
        # x = self.fc1(x)
        # x = F.leaky_relu(x)

        # x = self.norm2(x)
        # x = self.fc2(x)
        # x = F.leaky_relu(x)

        # x = self.norm3(x)
        # x = self.fc3(x)
        # x = F.leaky_relu(x)

        # x = self.norm4(x)
        # x = self.fc4(x)
        # x = F.leaky_relu(x)

        # x = self.norm5(x)
        # x = self.fc5(x)
        # x = F.leaky_relu(x)

        x = self.enc(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dec(x)

        # x = self.dilate(x)
        # x = self.up1(x)

        # # s1 = F.interpolate(e, size=(14, 14), mode="nearest")
        # x = self.dec1(x)

        # x = self.up2(x)
        # # s2 = F.interpolate(d2, size=(28, 28), mode="nearest")
        # x = self.dec2(x)

        # Reshape the image to the original shape
        x = x.view(x.shape[0], 1, 28, 28)

        return x


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
        batch_size = 64
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = stitch_output(test_data_input[i : i + batch_size], model(test_data_input[i : i + batch_size]))
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
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()

def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # You don't need to change the code below
    # Load the data
    train_data_input, train_data_label, test_data_input = get_data()
    # Train the model
    model = train_model(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    # The name of the submission file is submit_this_test_data_output.npz
    test_model(model, test_data_input)

    # kfold cross-validation
    cross_validate(TensorDataset(train_data_input, train_data_label), n_splits=2, n_epochs=30)

    return None


if __name__ == "__main__":
    main()
