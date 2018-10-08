import torch

import gan_networks
import data_manager
from network_session_handling import discriminator_logger


lin_layer1_size = 400
conv_layer1_size = 32
conv_layer2_size = 64
image_size = [32] * 2

discriminator = gan_networks.Discriminator(conv_layer1_size,
                                               conv_layer2_size,
                                               lin_layer1_size,
                                               image_size)
noise_length = 300
lin_hidden_layer_size = 800
output_size = [32] * 2
generator = gan_networks.Generator(noise_length,
                                   lin_hidden_layer_size,
                                   output_size)

input = torch.randn(noise_length)
batch_size = 15
correct_class_enum = data_manager.classes.FROG
data_loader = data_manager.data_loader(correct_class_enum, batch_size, True)
optimizer = torch.optim.Adam(discriminator.parameters())
criterion = torch.nn.CrossEntropyLoss()


epoch_count = 4
incorrect_to_correct_ratio = 1

logger = discriminator_logger(batch_size, epoch_count, incorrect_to_correct_ratio, correct_class_enum)

for epoch in range(epoch_count):
    for batch, labels in data_loader.get_ordered_iterator(incorrect_to_correct_ratio):
        labels = labels.to(discriminator.device)
        optimizer.zero_grad()

        outputs = discriminator(batch)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        logger.store_loss(loss.cpu().detach().numpy())
    logger.next_epoch()
print(logger.epochs_average_losses)

state = discriminator.state_dict()
logger.save_results(state)


