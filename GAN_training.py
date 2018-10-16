import torch

import gan_networks
import data_manager
from network_session_handling import discriminator_logger

import cv2 as cv
import numpy as np

def show_image(image):
    image = image.detach()
    num = image.numpy()
    num = np.flip(num.flatten(3).reshape(32, 32, 3), 2)
    num = np.rot90(num, axes=(1, 0))
    num = num - np.min(num)
    num = num / np.max(num)
    cv.imshow("test_image", num)
    cv.waitKey(0)


lin_layer1_size = 80
conv_layer1_size = 12
conv_layer2_size = 20
image_size = [32] * 2

discriminator = gan_networks.Discriminator(conv_layer1_size,
                                               conv_layer2_size,
                                               lin_layer1_size,
                                               image_size)
noise_length = 300
lin_hidden_layer_size = 300
output_size = [32] * 2
generator = gan_networks.Generator(noise_length,
                                   lin_hidden_layer_size,
                                   output_size)
batch_size = 5
correct_class_enum = data_manager.classes.FROG
data_loader = data_manager.data_loader(correct_class_enum, batch_size, True)

generator_optimizer = torch.optim.Adam(generator.parameters())
discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
criterion = torch.nn.CrossEntropyLoss()

input_size = 300
lin_hidden_layer_size = 800
output_size = [32] * 2

epoch_count = 4
incorrect_to_correct_ratio = 1

logger = discriminator_logger(batch_size, epoch_count,
                              incorrect_to_correct_ratio,
                              correct_class_enum,
                              noise_length)
 # rekord: 943 lin1 250/32/64 959 250/20/28
for epoch in range(epoch_count):
    for batch, labels in data_loader.get_ordered_iterator(incorrect_to_correct_ratio):
        labels = labels.to(discriminator.device)
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        outputs = discriminator(batch)
        discriminator_loss = criterion(outputs, labels)
        discriminator_loss.backward()

        discriminator_optimizer.step()


        noise_vector = torch.randn(noise_length)
        generated_image = generator(noise_vector)
        generated_image_classification = discriminator(generated_image)

        generator_loss = criterion(generated_image_classification, torch.cuda.LongTensor([0])) # 0 -> correct label
        generator_loss.backward()

        generator_optimizer.step()

        logger.store_loss(discriminator_loss.cpu().detach().numpy(),
                          generator_loss.cpu().detach().numpy())


    logger.next_epoch()
print("Discriminator average losses: \n %s" % str(logger.epochs_average_discriminator_losses))
print("Generator average losses: \n %s" % str(logger.epochs_average_generator_losses))


discriminator_state = discriminator.state_dict()
generator_state = generator.state_dict()
logger.save_results(discriminator_state, generator_state,'saved_session')

noise_vector = torch.randn(noise_length)
generated_image = generator(noise_vector)
print(discriminator(generated_image))

show_image(generated_image.cpu())
