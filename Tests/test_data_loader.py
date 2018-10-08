from unittest import TestCase
import data_manager as dm
import torch



class TestData_loader(TestCase):
    cifar_path = '../data'

    def test_should_return_single_image(self):
        # given
        batch_size = 4

        data_loader = dm.data_loader(dm.classes.FROG, batch_size, True, self.cifar_path)
        data_iter = iter(data_loader)
        image, isTrue = data_iter.next()

        self.assertEqual(image.shape, torch.Size([batch_size, 3, 32,32]))

    def test_iterator_should_return_2_images(self):
        # given
        batch_size = 2
        data_loader = dm.data_loader(dm.classes.FROG, batch_size, True, self.cifar_path)

        returned_batch_size = []
        # when
        for batch, labels in iter(data_loader):
            returned_batch_size = batch.size()
            break


        # then
        self.assertEqual(returned_batch_size, torch.Size([2,3,32,32]))

    def test_iterator_should_return_2_labels(self):
        # given
        batch_size = 2
        data_loader = dm.data_loader(dm.classes.FROG, batch_size, True, self.cifar_paths)

        label_size = []
        # when
        for batch, labels in iter(data_loader):
            label_size = len(labels)
            break

        # then
        self.assertEqual(label_size, 2)

    def test_data_iterator_should_not_throw_exception(self):
        # given
        batch_size = 15
        incorrect_per_correct = 2
        data_loader = dm.data_loader(dm.classes.FROG, batch_size, True, self.cifar_path)
        data_iterator = data_loader._data_iterator(data_loader._trainerloader,
                                                   data_loader._batch_size,
                                                   dm.classes.FROG,
                                                   incorrect_per_correct)

        # when
        for imgs, labels in data_iterator:
            continue

        # then
        self.assertTrue(True)

    def test_data_iterator_should_have_correct_data_ammount(self):
        # given
        batch_size = 15
        images_in_cifar_count = 5000
        incorrect_per_correct = 3
        data_loader = dm.data_loader(dm.classes.FROG, batch_size, True, self.cifar_path)

        # when
        data_iterator = data_loader._data_iterator(data_loader._trainerloader,
                                                   data_loader._batch_size,
                                                   dm.classes.FROG,
                                                   incorrect_per_correct,
                                                   )

        # then
        self.assertEqual(len(data_iterator.data), images_in_cifar_count * (incorrect_per_correct + 1))

    def test_data_iterator_should_remove_data(self):
        # given
        batch_size = 15
        incorrect_per_correct = 2
        data_loader = dm.data_loader(dm.classes.FROG, batch_size, True, self.cifar_path)
        data_iterator = data_loader._data_iterator(data_loader._trainerloader,
                                                   data_loader._batch_size,
                                                   dm.classes.FROG,
                                                   incorrect_per_correct)

        # when
        for imgs, labels in data_iterator:
            continue

        # then
        self.assertTrue(len(data_iterator.data) < batch_size)

    def test_data_iterator_should_return_correct_labels(self):
        # given
        batch_size = 15
        incorrect_per_correct = 2
        expected_labels = [0,1,1] * 2
        data_loader = dm.data_loader(dm.classes.FROG, batch_size, True, self.cifar_path)
        data_iterator = data_loader._data_iterator(data_loader._trainerloader,
                                                   data_loader._batch_size,
                                                   dm.classes.FROG,
                                                   incorrect_per_correct)

        # when
        imgs, labels = data_iterator.next()

        # then
        self.assertEqual(labels[:6].tolist(), expected_labels)




