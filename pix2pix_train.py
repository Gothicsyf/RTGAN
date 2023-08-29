import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder, DataAugmentation
from model import Generator, Discriminator
import utils1
from utils1 import plot_test_result, make_gif, plot_loss
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"
from tqdm import tqdm
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/dataset/', help='Parent folder of the dataset')
parser.add_argument('--logs_dir', default='/logs/',
                    help='Parent folder for logs')
parser.add_argument('--dataset', default='edges2shoes', help='input dataset')
parser.add_argument('--direction', default='AtoB', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=14, help='train batch size')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=256, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=195, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args([])
print(params)

# Directories for loading data and saving results
data_dir = os.path.join(params.root_dir, params.dataset)
save_dir = os.path.join(params.logs_dir, params.dataset) + '_results/'
model_dir = os.path.join(params.logs_dir, params.dataset) + '_model/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')

img_dir = os.path.join(params.logs_dir, 'result/') + 'test_result_{}'.format(now_time)
if not os.path.exists(img_dir):
    os.makedirs(img_dir)
# Set the logger
log_dir = save_dir + 'logs/log_{}'.format(now_time)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Data pre-processing
transform = transforms.Compose([transforms.Resize(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

nw = min([os.cpu_count(), params.batch_size if params.batch_size > 1 else 0, 56])
# Train data
DataAugmentation(data_dir, subfolder='train', Data_augmentation=False)
train_data = DatasetFromFolder(data_dir, subfolder='train', direction=params.direction, transform=transform,
                               resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                pin_memory=True,
                                                shuffle=True,
                                                num_workers=nw,)

# Test data
test_data = DatasetFromFolder(data_dir, subfolder='test', direction=params.direction, transform=transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)
test_input, test_target = test_data_loader.__iter__().__next__()

G = Generator(3, params.ngf, params.img_size, 3)
D = Discriminator(6, params.ndf, 1)

# Models

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    G = torch.nn.DataParallel(G, device_ids=[0,1])
    G.cuda()
    D = torch.nn.DataParallel(D, device_ids=[0,1])
    D.cuda()
else:
    G = G.cuda()
    D = D.cuda()


# Loss function
BCE_loss = torch.nn.BCELoss()
L1_loss = torch.nn.L1Loss()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_avg_losses = []
G_avg_losses = []

step = 0

for epoch in tqdm(range(params.num_epochs)):
    D_losses = []
    G_losses = []
    for i, (input, target) in enumerate(train_data_loader):
        x_ = input.cuda()
        y_ = target.cuda()

        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = params.lamb * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.data)
        G_losses.append(G_loss.data)


        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch + 1, params.num_epochs, i + 1, len(train_data_loader), D_loss.data, G_loss.data))

        with open(os.path.join(log_dir, 'train_log.txt'), mode='a') as f:
            f.write('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f\n'
                    % (epoch + 1, params.num_epochs, i + 1, len(train_data_loader), D_loss.data, G_loss.data))
        if epoch % 5 == 0:
            torch.save(G.state_dict(), model_dir + 'generator_param_{}.pkl'.format(epoch))
            torch.save(D.state_dict(), model_dir + 'discriminator_param_{}.pkl'.format(epoch))
        # iterator.update(1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    # Show result for test image
    gen_image = G(Variable(test_input.cuda()))
    gen_image = gen_image.cpu().data
    plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=img_dir)

# Plot average losses
plot_loss(D_avg_losses, G_avg_losses, params.num_epochs, show=True,
          save_dir='/save_dir/')

# Save trained parameters of model
torch.save(G.state_dict(), model_dir + 'generator_param.pth')
torch.save(D.state_dict(), model_dir + 'discriminator_param.pth')

