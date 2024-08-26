import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
from math import ceil
import csv
import os

class ImageFeatureFolder(dset.ImageFolder):
    def __init__(self, image_root, landmark_file, transform):
        super(ImageFeatureFolder, self).__init__(root=image_root, transform=transform)
        with open(landmark_file, 'r') as f:
            data = f.read()
            data = data.strip().split('\n')
            self.attrs = torch.FloatTensor([list(map(float, line.split(",")[1:])) for line in data[1:57585]])

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return img, self.attrs[index]

now = datetime.now()
today = now.date()
hour = f"{now.hour}-{now.minute}"
config = {
'num_feature': 4096,
"noise_size": 100,
"batch_size": 128,
"num_epoch": 30,
"learning_rate": 0.0002,
"num_channel": 3,
"beta": (0.0, 0.99),
"result_dir": "./result_av_bild_18_utan_bild_18",
"eval_dir": "./eval_av_bild_18_utan_bild_18",
"loss_filename": "loss_feature_18_utan_bild_18.csv"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for key, val in config.items():
    print(key, "=", val)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
        nn.ConvTranspose2d(config["noise_size"] + config["num_feature"], 512, 4, 1, 0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
        nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.ConvTranspose2d(64, config["num_channel"], 4, 2, 1, bias=False),
        nn.Tanh(),
        )

    def forward(self, x, attr):
        attr = attr.view(-1, config["num_feature"], 1, 1)
        x = torch.cat([x, attr], 1)
        return self.main(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_input = nn.Linear(config["num_feature"], 64 * 64)
        self.main = nn.Sequential(
        nn.Conv2d(config["num_channel"] + 1, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, attr):
        attr = self.feature_input(attr).view(-1, 1, 64, 64)
        x = torch.cat([x, attr], 1)
        return self.main(x).view(-1, 1)

    
class Trainer:
    def __init__(self, dataset_size):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.loss = nn.MSELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=config["learning_rate"],
        betas=config["beta"])
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=config["learning_rate"],
        betas=config["beta"])

        self.generator.to(device)
        self.discriminator.to(device)
        self.loss.to(device)
        self.total_batches = ceil(dataset_size / config["batch_size"])

        self.d_loss = []
        self.g_loss = []

        self.eval_attr = torch.FloatTensor(config["num_feature"], config["num_feature"]).fill_(0)

    def save_loss(self, filename):
        with open(filename, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["d_loss"] + self.d_loss)
            writer.writerow(["g_loss"] + self.g_loss)

    def evaluate(self, epoch):
        noise = torch.FloatTensor(config["batch_size"], config["noise_size"], 1, 1).to(device)
        with torch.no_grad():
            self.eval_attr = self.eval_attr.to(device)
            noise.data.normal_(0, 1)
            fake = self.generator(noise, self.eval_attr)
            vutils.save_image(fake.data, '{}/eval_epoch_{:03d}.png'.format(config["eval_dir"], epoch),
            normalize=True)

    def train(self, dataloader):
        noise = torch.FloatTensor(config["batch_size"], config["noise_size"], 1, 1).to(device)
        label_real = torch.FloatTensor(config["batch_size"], 1).fill_(1).to(device)
        label_fake = torch.FloatTensor(config["batch_size"], 1).fill_(0).to(device)
        for epoch in range(config["num_epoch"]):
            print(f"Epoch {epoch + 1}/{config['num_epoch']}")

        for i, (data, attr) in enumerate(dataloader, 0):
            LINE_CLEAR = '\x1b[2K'
            print(end=LINE_CLEAR)
            print(f"Batch: {i + 1}/{self.total_batches}", end="\r")

            # Handle potentially smaller last batch
            current_batch_size = data.size(0)
            if current_batch_size != config["batch_size"]:
                break
    
        # train discriminator
        self.discriminator.zero_grad()

        batch_size = data.size(0)
        label_real.data.resize(batch_size, 1).fill_(1)
        label_fake.data.resize(batch_size, 1).fill_(0)
        noise.data.normal_(0, 1)

        attr = attr.to(device)
        real = data.to(device)
        d_real = self.discriminator(real, attr)
        #print("d_real size:", d_real.size())

        fake = self.generator(noise, attr)
        d_fake = self.discriminator(fake.detach(), attr) # not update generator
        #print("d_fake size:", d_fake.size())

        d_loss = self.loss(d_real, label_real) + self.loss(d_fake, label_fake) # real label
        d_loss.backward()
        self.optimizer_d.step()

        # train generator
        self.generator.zero_grad()
        d_fake = self.discriminator(fake, attr)
        g_loss = self.loss(d_fake, label_real) # trick the fake into being real
        g_loss.backward()
        self.optimizer_g.step()

        self.g_loss.append(g_loss.item())
        self.d_loss.append(d_loss.item())
        print("epoch{:03d} d_real: {}, d_fake: {}".format(epoch, d_real.mean(), d_fake.mean()))
        print(f"End of epoch {epoch + 1} - Discriminator Loss: {d_loss.item()}, Generator Loss:
        {g_loss.item()}")
        vutils.save_image(fake.data, '{}/result_epoch_{:03d}.png'.format(config["result_dir"], epoch),
        normalize=True)
        self.evaluate(epoch)

transform = transforms.Compose([
transforms.CenterCrop(178),
transforms.Resize(64),
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data = ImageFeatureFolder('./classfolder', "feature_vector.txt", transform)

fig, axs = plt.subplots(1, 5, figsize=(20, 4))

for i, img in enumerate(data.imgs[:5]):
    image = mpimg.imread(img[0])
    axs[i].imshow(image)
    axs[i].axis('off') # Turn off axis labels

plt.show()
print(config)
print(config['noise_size'])

#create result and eval directories, these are not used with the supercomputer Berzelius
#os.mkdir(config["result_dir"])
#os.mkdir(config["eval_dir"])

dataset_size = len(data)
dataloader = torch.utils.data.DataLoader(data, batch_size=config["batch_size"], shuffle=True)
trainer = Trainer(dataset_size)
feature = torch.tensor(#...featurevector to be reconstructed here
 )

attr_tensor = torch.FloatTensor(config["batch_size"], config["num_feature"]).fill_(0)
for i, row in enumerate(attr_tensor):
    attr_tensor[i] = feature

trainer.eval_attr = attr_tensor
trainer.train(dataloader)
trainer.save_loss(config["loss_filename"])