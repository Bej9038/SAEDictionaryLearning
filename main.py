import torch
import torch.nn as nn
import torch.optim as optim
import utils
from agc import AGC
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, features, lambda_sparsity):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(in_features=input_dim, out_features=features, bias=True)
        self.decoder = nn.Linear(in_features=features, out_features=input_dim, bias=True)
        self.lambda_sparsity = lambda_sparsity
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def compute_loss(self, x, x_hat, fx):
        l2 = self.mse_loss(x, x_hat)
        # l1 = 0
        # for i in range(len(fx)):
        #     l1 += self.lambda_sparsity * torch.abs(fx[i]) * torch.norm(self.decoder[:, i])
        #
        l1 = self.lambda_sparsity * torch.sum(torch.abs(fx) * torch.norm(self.decoder.weight, dim=1))
        return l1 + l2


class Trainer:
    def __init__(self):
        self.device = "cpu"
        self.latent_dim = 256
        self.features = 2 ** 17
        self.lambda_sparsity = 5

        # create LR schedule
        self.lr = 5e-5
        self.batch_size = 128
        self.grad_norm = 1
        self.steps = 200000

        self.encoder = AGC.from_pretrained("Audiogen/agc-discrete").to(self.device)
        self.sae = SparseAutoencoder(self.latent_dim, self.features, self.lambda_sparsity).to(self.device)
        print(f"SAE Parameters: {self.latent_dim * self.features * 2}")
        self.optimizer = optim.Adam(self.sae.parameters(), lr=self.lr, betas=(0.9, 0.999), fused=True)
        self.dataloader = utils.get_dataloader(self.batch_size)

    def train(self):
        self.sae.train()
        step = 0

        with tqdm(total=self.steps) as pbar:
            while step < self.steps:
                for batch in self.dataloader:
                    print(f"batch size: {batch.shape}")
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()
                    inputs = self.encoder.encode(batch)
                    print(inputs.shape)
                    encoded, decoded = self.sae(inputs)
                    loss = self.sae.compute_loss(inputs, decoded, encoded)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)

                    self.optimizer.step()
                    epoch_loss += loss.item()

                    step += 1
                    pbar.update(1)

                    if step % 100 == 0:
                        pbar.set_postfix(loss=epoch_loss/100)
                        epoch_loss = 0

                    if step >= self.steps:
                        break


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
