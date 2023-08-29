import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator
import utils1
import utils
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='/dataset/', help='Parent folder of the dataset')
parser.add_argument('--logs_dir', default='/logs/', help='Parent folder for logs')
parser.add_argument('--dataset', required=False, default='edges2shoes', help='input dataset')
parser.add_argument('--direction', required=False, default='AtoB', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=1, help='test batch size')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
params = parser.parse_args()
print(params)

# Directories for loading data and saving results
data_dir = os.path.join(params.root_dir, params.dataset)
save_dir = os.path.join(params.logs_dir, params.dataset) + '_test_results/'
model_dir = os.path.join(params.logs_dir, params.dataset) + '_model/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Data pre-processing
test_transform = transforms.Compose([transforms.Resize(params.input_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

# Test data
test_data = DatasetFromFolder(data_dir, subfolder='test', direction=params.direction, transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)

# Load model
G = Generator(3, params.ngf, params.img_size, 3)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    G = torch.nn.DataParallel(G, device_ids=[0,1])  # DistributedDataParallel
    G.cuda()
else:
    G=G.cuda()
G.load_state_dict(torch.load(model_dir + 'generator_param_35.pkl'))

# Test
for i, (input, target) in enumerate(test_data_loader):
    # input & target image data
    x_ = Variable(input.cuda())
    y_ = Variable(target.cuda())

    gen_image = G(x_)
    gen_image = gen_image.cpu().data

    # Show result for test data
    utils1.plot_test_result(input, target, gen_image, i, training=False, save=True, save_dir=save_dir)

    print('%d images are generated.' % (i + 1))

