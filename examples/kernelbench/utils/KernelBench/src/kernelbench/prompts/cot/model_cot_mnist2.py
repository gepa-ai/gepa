"""
Let us think about how to optimize the code step by step.
"""

# Step 1: Let us break down the PyTorch module into step-by-step instructions.
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # First convolutional layer
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Second convolutional layer
        self.fc1 = nn.Linear(320, 50)                # First fully connected layer
        self.fc2 = nn.Linear(50, 10)                 # Second fully connected layer

    def forward(self, x):
        """
        Perform the following steps:

        1. Apply the first convolutional layer (`conv1`) to the input.
        2. Apply ReLU activation to the output of `conv1`.
        3. Perform max pooling with a kernel size of 2.
        4. Apply the second convolutional layer (`conv2`) to the result.
        5. Apply ReLU activation to the output of `conv2`.
        6. Perform max pooling with a kernel size of 2 again.
        7. Flatten the 2D feature map into a 1D tensor.
        8. Apply the first fully connected layer (`fc1`) to the flattened tensor.
        9. Apply ReLU activation to the output of `fc1`.
        10. Apply the second fully connected layer (`fc2`).
        11. Compute the log-softmax of the output from `fc2`.

        Returns the log probabilities over 10 classes.
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Steps 1-3
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Steps 4-6
        x = x.view(-1, 320)                        # Step 7
        x = F.relu(self.fc1(x))                    # Steps 8-9
        x = self.fc2(x)                            # Step 10
        return F.log_softmax(x, dim=1)             # Step 11

# Step 2: Let us describe how each step could be implemented inside of a CUDA kernel.
"""
1. Load the input tensor values: `float x = inp[i]`.

2. Perform the first convolution:
   - Use a sliding 5x5 window over the input tensor.
   - Multiply the window elements by the weights of the first convolutional layer.
   - Sum the results and add the bias term.

3. Apply ReLU activation: `out[i] = max(0.0f, conv1_result)`.

4. Perform max pooling:
   - Take a 2x2 window from the convolution output.
   - Compute the maximum value in this window.

5. Repeat steps 2-4 for the second convolutional layer, using the output of the first pooling layer.

6. Flatten the output feature map into a 1D array:
   - Map the 2D indices of the feature map to a 1D memory location.

7. Perform matrix-vector multiplication for the first fully connected layer:
   - Multiply the flattened tensor by the weights of `fc1`.
   - Add the bias term.

8. Apply ReLU activation: `fc1_out = max(0.0f, result_from_fc1)`.

9. Perform matrix-vector multiplication for the second fully connected layer:
   - Multiply the output of `fc1` by the weights of `fc2`.
   - Add the bias term.

10. Compute log-softmax:
   - Normalize the output values by subtracting the maximum for numerical stability.
   - Compute the exponentials and their sum.
   - Take the logarithm of the normalized probabilities.
"""

# Step 3. Let us put all of the steps together into CUDA kernel code.

