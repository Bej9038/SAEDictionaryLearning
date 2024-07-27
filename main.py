import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, lambda_sparsity):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.lambda_sparsity = lambda_sparsity

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_function(self, x, decoded, encoded):
        mse_loss = nn.MSELoss()(decoded, x)
        sparsity_loss = self.lambda_sparsity * torch.sum(torch.norm(encoded, p=2, dim=1))
        return mse_loss + sparsity_loss


def train(model, data, optimizer, batch_size, num_epochs):
    model.train()
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in dataloader:
            inputs = batch[0]
            optimizer.zero_grad()
            encoded, decoded = model(inputs)
            loss = model.loss_function(inputs, decoded, encoded)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

def main():
    # Load latent vectors (example)
    # Replace with actual loading of latent vectors
    latent_vectors = np.load("latent_vectors.npy")
    latent_vectors = torch.tensor(latent_vectors, dtype=torch.float32)

    # Define model parameters
    input_dim = latent_vectors.shape[1]  # Size of the latent vectors
    hidden_dim = 256  # Number of hidden units (can be adjusted)
    lambda_sparsity = 5.0  # Sparsity regularization parameter
    learning_rate = 5e-5
    batch_size = 256
    num_epochs = 50

    # Initialize model, optimizer, and loss function
    model = SparseAutoencoder(input_dim, hidden_dim, lambda_sparsity)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, latent_vectors, optimizer, batch_size, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), "sparse_autoencoder.pth")


if __name__ == "__main__":
    main()
