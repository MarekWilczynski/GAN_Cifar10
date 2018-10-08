from unittest import TestCase
import gan_networks as networks

import torch


class TestNetworks(TestCase):
    def test_discriminator_output_size(self):
        # given
        input_size = 100
        lin_layer1_size = 400
        conv_layer1_size = 32
        conv_layer2_size = 64
        image_size = [32] * 2
        number_of_channels = 3

        input = torch.randn([1, number_of_channels] + image_size) # one channel, one layer
        discriminator = networks.Discriminator(conv_layer1_size,
                                               conv_layer2_size,
                                               lin_layer1_size,
                                               image_size)
        # when
        output = discriminator(input)
        self.assertEqual(output.size(1), 2)

    def test_generator_output_size(self):
        # given
        input_size = 300
        lin_hidden_layer_size = 800
        output_size = [32] * 2

        input = torch.randn(input_size)
        generator = networks.Generator(input_size,
                                       lin_hidden_layer_size,
                                       output_size)
        # when
        output = generator(input)

        #then
        self.assertEqual(output.size(), torch.Size([1, 3] + output_size))

    def test_feeding_generated_to_discrimantor(self):
        # given

        # dicriminator
        lin_layer1_size = 400
        conv_layer1_size = 32
        conv_layer2_size = 64
        image_size = [32] * 2

        discriminator = networks.Discriminator(conv_layer1_size,
                                               conv_layer2_size,
                                               lin_layer1_size,
                                               image_size)

        # generator
        input_size = 300
        lin_hidden_layer_size = 800
        output_size = [32] * 2

        input = torch.randn(input_size)
        generator = networks.Generator(input_size,
                                       lin_hidden_layer_size,
                                       output_size)
        generated = generator(input)

        # when
        output = discriminator(generated)

        # then
        self.assertTrue(True)


    pass