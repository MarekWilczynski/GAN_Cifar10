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


lin_layer1_size = 250
conv_layer1_size = 20
conv_layer2_size = 28
image_size = [32] * 2

discriminator = gan_networks.Discriminator(conv_layer1_size,
                                               conv_layer2_size,
                                               lin_layer1_size,
                                               image_size)
noise_length = 400
lin_hidden_layer_size = 600
deconv1_size = 120
deconv2_size = 200
output_size = [32] * 2
generator = gan_networks.Generator(noise_length,
                                   lin_hidden_layer_size,
                                   deconv1_size,
                                   deconv2_size,
                                   output_size)
batch_size = 5
correct_class_enum = data_manager.classes.FROG
data_loader = data_manager.data_loader(correct_class_enum, batch_size, True)

generator_optimizer = torch.optim.Adam(generator.parameters())
discriminator_optimizer = torch.optim.Adam(discriminator.parameters())
criterion = torch.nn.CrossEntropyLoss()

epoch_count = 4
incorrect_to_correct_ratio = 1

logger = discriminator_logger(batch_size, epoch_count,
                              incorrect_to_correct_ratio,
                              correct_class_enum,
                              noise_length)
 # rekord: 943 250/32/64 959 250/20/28

iteration_counter = 0

for epoch in range(epoch_count):
    for batch, labels in data_loader.get_ordered_iterator(incorrect_to_correct_ratio):
        generator_loss = torch.Tensor([0])

        labels = labels.to(discriminator.device)
        discriminator_optimizer.zero_grad()

        outputs = discriminator(batch)
        discriminator_loss = criterion(outputs, labels)
        discriminator_loss.backward()

        discriminator_optimizer.step()
        if(epoch > 0):
            iteration_counter = iteration_counter + 1


            generator_optimizer.zero_grad()

            noise_vector = torch.randn(noise_length)
            generated_image = generator(noise_vector)
            generated_image_classification = discriminator(generated_image)

            # should fool the discriminator
            generator_loss = criterion(generated_image_classification, torch.cuda.LongTensor([0]))

            generator_loss.backward()
            generator_optimizer.step()

            if (iteration_counter == 1):
                # delay discriminator learning
                iteration_counter = 0
                discriminator_optimizer.zero_grad()
                # should not by fooled by generator

                discriminator_vs_generator_loss = criterion(generated_image_classification,
                                                            torch.cuda.LongTensor([1]))  # 1 -> wrong

                discriminator_vs_generator_loss.backward(retain_graph=True)
                discriminator_optimizer.step()

        logger.store_loss(discriminator_loss.cpu().detach().numpy(),
                          generator_loss.cpu().detach().numpy())


    logger.next_epoch()
print("Discriminator average losses: \n %s" % str(logger.epochs_average_discriminator_losses))
print("Generator average losses: \n %s" % str(logger.epochs_average_generator_losses))


discriminator_state = discriminator.state_dict()
generator_state = generator.state_dict()
logger.save_results(discriminator_state, generator_state,'saved_session')

while(True):

    noise_vector = torch.randn(noise_length)
    generated_image = generator(noise_vector)
    print(discriminator(generated_image))

    show_image(generated_image.cpu())
