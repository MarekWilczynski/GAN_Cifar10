from numpy import mean
from torch import save, load
from gan_networks import Discriminator
from gan_networks import Generator


class discriminator_logger():
    def __init__(self, batch_size, epoch_count, correct_to_incorrect_ratio, class_enum, gen_noise_len):
        self.epochs_average_discriminator_losses = []
        self.epochs_average_generator_losses = []
        self.current_discriminator_losses = []
        self.current_generator_losses = []
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.discriminated_class = class_enum
        self.gen_noise_len = gen_noise_len
        if(correct_to_incorrect_ratio == 0):
            self.iterator_used = "Random"
        else:
            self.iterator_used = "Ordered, ratio : %d" % correct_to_incorrect_ratio
    def save_results(self, discriminator_state_model, generator_state_model, file_name = 'model'):

        state = {'discriminator_state_model': discriminator_state_model,
                 'generator_state_model': generator_state_model,
                 'epochs_average_discriminator_losses': self.epochs_average_discriminator_losses,
                 'epochs_average_generator_losses': self.epochs_average_generator_losses,
                 'batch_size': self.batch_size,
                 'iterator_type': self.iterator_used,
                 'epoch_count' : self.epoch_count,
                 'discriminated_class': self.discriminated_class,
                 'generator_noise_length': self.gen_noise_len}

        save(state, file_name + '.pth.tar')

    def next_epoch(self):
        epoch_discriminator_mean_loss = mean(self.current_discriminator_losses)
        epoch_generator_mean_loss = mean(self.current_generator_losses)

        print("Discriminator mean epoch loss: %f" % epoch_discriminator_mean_loss)
        print("Generator mean epoch loss: %f" % epoch_generator_mean_loss)

        self.epochs_average_discriminator_losses.append(epoch_discriminator_mean_loss)
        self.epochs_average_generator_losses.append(epoch_generator_mean_loss)

        self.current_discriminator_losses = []
        self.current_generator_losses = []

    def store_loss(self, discriminator_loss, generator_loss):
        self.current_discriminator_losses.append(discriminator_loss)
        self.current_generator_losses.append(generator_loss)

def load_session(file_name = 'model'):
    dict = load(file_name + '.pth.tar')
    discriminator_state_model = dict['discriminator_state_model']

    lin_layer1_size = discriminator_state_model['lin1.weight'].size(0)
    conv_layer1_size = discriminator_state_model['conv1.weight'].size(0)
    conv_layer2_size = discriminator_state_model['conv2.weight'].size(0)
    image_size = [32] * 2

    discriminator = Discriminator(conv_layer1_size, conv_layer2_size, lin_layer1_size, image_size)
    discriminator.load_state_dict(discriminator_state_model)

    training_data = {'epochs_average_losses': dict['epochs_average_discriminator_losses'],
                     'batch_size': dict['batch_size'],
                     'iterator_type': dict['iterator_type'],
                     'epoch_count': dict['epoch_count'],
                     'discriminated_class': dict['discriminated_class']}

    generator_state_model = dict['generator_state_model']
    noise_length = dict['generator_noise_length']
    generator_hidden_layer_size = generator_state_model['lin2.weight'].size(0)
    generator_deconv1_layer_size = generator_state_model['deconv1.weight'].size(0)
    generator_deconv2_layer_size = generator_state_model['deconv2.weight'].size(0)
    generator = Generator(noise_length,
                          generator_hidden_layer_size,
                          generator_deconv1_layer_size,
                          generator_deconv2_layer_size,
                          image_size)

    generator_data = {'noise_length': noise_length,
                      'epochs_average_generator_losses': dict['epochs_average_generator_losses']}

    return discriminator, training_data, generator, generator_data
