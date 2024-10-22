import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import utils
from agc import AGC
from tqdm import tqdm
import einops
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import dac
import torchaudio


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, features, lambda_sparsity):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(in_features=input_dim, out_features=features, bias=True)
        self.decoder = nn.Linear(in_features=features, out_features=input_dim, bias=True)
        self.lambda_sparsity = lambda_sparsity
        self.mse_loss = nn.MSELoss()

    def _initialize_weights(self):
        # Initialize biases to zeros
        init.zeros_(self.decoder.bias)
        init.zeros_(self.encoder.bias)

        # Initialize W_d
        W_d = torch.randn(self.decoder.weight.size())
        W_d_norm = torch.norm(W_d, dim=0)
        W_d = W_d / W_d_norm * torch.rand(1).item() * 0.9 + 0.1  # Random L2 norm between 0.1 and 1

        self.decoder.weight.data = W_d

        # Initialize W_e to W_d.T
        self.encoder.weight.data = W_d.T

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return encoded, decoded

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, x):
        return self.decoder(x)

    def compute_loss(self, x, x_hat, fx):
        l2 = self.mse_loss(x, x_hat)
        l1 = self.lambda_sparsity * torch.sum(torch.abs(fx) * torch.norm(self.decoder.weight, dim=0))
        return l1 + l2


class Trainer:
    def __init__(self, rank):
        self.device = "cpu"
        self.latent_dim = 12000
        self.features = 2 ** 10
        self.lambda_sparsity = 5

        # create LR schedule
        self.lr = 1e-6
        self.batch_size = 1
        self.grad_norm = 1
        self.steps = 2**12
        self.writer = SummaryWriter()

        self.encoder = AGC.from_pretrained("Audiogen/agc-discrete").to(self.device)
        # self.dac = dac.DAC.load(dac.utils.download(model_type="44khz"))
        self.sae = SparseAutoencoder(self.latent_dim, self.features, self.lambda_sparsity).to(self.device)
        # self.model = DDP(self.sae, device_ids=[self.device])

        self.optimizer = optim.Adam(self.sae.parameters(), lr=self.lr, betas=(0.9, 0.999), fused=True)
        self.dataloader = utils.get_dataloader(self.batch_size)
        self.scheduler = utils.CustomLRScheduler(self.optimizer, self.steps)
        if self.device == 0: print(f"SAE Parameters: {self.latent_dim * self.features * 2}")

    def train(self):
        self.sae.train()
        step = 0
        with tqdm(total=self.steps) as pbar:
            while step < self.steps:
                for batch in self.dataloader:
                    batch = batch.to(self.device)
                    self.optimizer.zero_grad()

                    # encode with neural compression
                    inputs = self.encoder.encode(batch).to(torch.float32)
                    inputs = einops.rearrange(inputs, 'b h w -> b (h w)')

                    # batch = einops.rearrange(batch, 'b c s -> (b c) 1 s')
                    # z, codes, latents, _, _ = self.dac.encode(batch)

                    # sae
                    encoded, decoded = self.sae(inputs)

                    loss = self.sae.compute_loss(inputs, decoded, encoded)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), 1.0)

                    self.writer.add_scalar('loss', loss.item(), step)
                    self.writer.add_scalar('lr', self.scheduler.get_lr()[0], step)

                    self.optimizer.step()
                    self.scheduler.step()

                    step += 1
                    pbar.update(1)
                    if step % 10 == 0:
                        torch.save(self.sae.state_dict(), 'ckpts/sae.pth')

                    if step >= self.steps:
                        break
        self.writer.close()

    def test(self):
        ckpt = torch.load("ckpts/sae.pth", map_location=torch.device('cpu'))  # or 'cuda' for GPU
        self.sae.load_state_dict(ckpt)
        self.sae.eval()
        with torch.no_grad():
            for batch in self.dataloader:
                batch = batch.to(self.device)

                # grab original encoder latents
                latents = self.encoder.encode(batch).to(torch.float32)
                b, h, w = latents.shape
                latents = einops.rearrange(latents, 'b h w -> b (h w)')

                # grab sae latents
                encoded = self.sae.encode(latents)

                # edits

                decoded = self.sae.decode(encoded)

                new_latents = einops.rearrange(decoded, 'b (h w) -> b h w', h=h, w=w).to(torch.int)
                # recover audio
                output = self.encoder.decode(new_latents)
                torchaudio.save("test.wav", output[0], 44100)
                break


def main(rank, world_size):
    # utils.ddp_setup(rank, world_size)
    trainer = Trainer(rank)
    # trainer.train()
    trainer.test()
    # destroy_process_group()


if __name__ == "__main__":
    main(0, 1)
    # world_size = torch.cuda.device_count()
    # mp.spawn(main, args=(world_size), nprocs=world_size)
