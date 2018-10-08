from numpy import sum
from torch import save, load
from gan_networks import Discriminator

class discriminator_logger():
    def __init__(self, batch_size, epoch_count, correct_to_incorrect_ratio, class_enum):
        self.epochs_average_losses = []
        self.current_losses = []
        self.batch_size = batch_size
        self.epoch_count = epoch_count
        self.discriminated_class = class_enum
        if(correct_to_incorrect_ratio == 0):
            self.iterator_used = "Random"
        else:
            self.iterator_used = "Ordered, ratio : %d" % correct_to_incorrect_ratio
    def save_results(self, discriminator_state_model, file_name = 'model'):

        state = {'state_model': discriminator_state_model,
                 'epochs_average_losses' : self.epochs_average_losses,
                 'batch_size': self.batch_size,
                 'iterator_type': self.iterator_used,
                 'epoch_count' : self.epoch_count,
                 'discriminated_class': self.discriminated_class}

        save(state, file_name + '.pth.tar')

    def next_epoch(self):
        epoch_mean_loss = sum(self.current_losses)
        print("Total epoch loss: %f" % epoch_mean_loss)
        self.epochs_average_losses.append(epoch_mean_loss)
        self.current_losses = []

    def store_loss(self, loss):
        self.current_losses.append(loss)

def load_discriminator(file_name = 'model'):
    dict = load(file_name + '.pth.tar')
    discriminator_state_model = dict['state_model']

    lin_layer1_size = discriminator_state_model['lin1.weight'].size(0)
    conv_layer1_size = discriminator_state_model['conv1.weight'].size(0)
    conv_layer2_size = discriminator_state_model['conv2.weight'].size(0)
    image_size = [32] * 2

    discriminator = Discriminator(conv_layer1_size, conv_layer2_size, lin_layer1_size, image_size)
    discriminator.load_state_dict(discriminator_state_model)

    training_data = {'epochs_average_losses': dict['epochs_average_losses'],
                     'batch_size': dict['batch_size'],
                     'iterator_type': dict['iterator_type'],
                     'epoch_count': dict['epoch_count'],
                     'discriminated_class': dict['discriminated_class']}

    return discriminator, training_data
