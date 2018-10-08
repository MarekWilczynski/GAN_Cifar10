import torch
import torch.nn as nn
import numpy as np

class Discriminator(nn.Module):
    # class representing the discriminator in GAN
    # lin_input_count1 most likely should be equal to image resolution multiplied
    # by count of conv2 outputs

    def __init__(self, conv_output_count, conv2_output_count, lin_output_count, image_size, kernel_size = 5):
        super(Discriminator,self).__init__()

        if(not(torch.cuda.is_available())):
            raise RuntimeError("No CUDA found!")
        if(len(image_size) < 2):
           raise IndexError("Incorrect image size!")

        padding_size = int((kernel_size - 1) / 2)
        padding = (padding_size, padding_size)

        self.device = torch.device("cuda:0")
        kernel_size = [kernel_size, kernel_size]
        input_channels = 3 # for rgb images
        
        
        conv_input_count1 = input_channels # single rgb image
        conv_output_count1 = conv_output_count

        conv_input_count2 = conv_output_count1
        conv_output_count2 = conv2_output_count

        lin_input_count1 = conv2_output_count * np.prod(image_size)
        lin_output_count1 = lin_output_count

        lin_input_count2 = lin_output_count
        lin_output_count2 = 2 # belongs to class or not

        self.conv1 = nn.Conv2d(conv_input_count1, conv_output_count1, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(conv_input_count2, conv_output_count2, kernel_size, padding=padding)
        self.lin1 = nn.Linear(lin_input_count1, lin_output_count1)
        self.lin2 = nn.Linear(lin_input_count2, lin_output_count2)
        self.softmax = nn.Softmax(dim=1) # sum alongside y axis
        self.to(self.device)

    def forward(self, input):
        # tensor_input = torch.Tensor(input)
        tensor_input = input
        tensor_input = tensor_input.to(self.device)

        x = torch.relu(self.conv1(tensor_input))
        x = torch.relu(self.conv2(x))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        #x = self.softmax(x)
        return x


class Generator(nn.Module):
    # Generates 32x32px image

    def __init__(self, input_length, hidden_lin_layer_size, output_image_size, kernel_size = 5):
        super(Generator, self).__init__()

        if (not (torch.cuda.is_available())):
            raise RuntimeError("No CUDA found!")
        self.device = torch.device("cuda:0")
        self.output_size = output_image_size


        number_of_channels = 3

        lin_input1_size = input_length # noise_length
        lin_output1_size = hidden_lin_layer_size

        lin_input2_size = lin_output1_size
        lin_output2_size = int(np.prod(output_image_size) / 16) # 16 -> ((a/2)^2)

        deconv_input1 = 1 # red
        deconv_output1 = 2 # green

        deconv_input2 = deconv_output1
        deconv_output2 = 3 # blue

        self.lin1 = nn.Linear(lin_input1_size, lin_output1_size)
        self.lin2 = nn.Linear(lin_input2_size, lin_output2_size)
        self.deconv1 = nn.ConvTranspose2d(deconv_input1, deconv_output1, kernel_size, stride=2, padding=2,output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(deconv_input2, deconv_output2, kernel_size, stride=2, padding=2,output_padding=1)
        self.to(self.device)


    def forward(self, input):
        tensor_input = torch.Tensor(input)
        tensor_input = tensor_input.to(self.device)

        x = torch.relu(self.lin1(tensor_input))
        x = torch.relu(self.lin2(x))
        img_before_scaling_size = int(self.output_size[0] / 4)
        x = x.view(1, 1, img_before_scaling_size, -1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        return x
