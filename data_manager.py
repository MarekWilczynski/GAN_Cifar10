import torch

import torchvision
import torchvision.transforms as transforms

from enum import Enum

class data_loader():

    def __init__(self, correct_class_enum, batch_size, training, root='./data'):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=root, train=training,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=False, num_workers=0)

        self._batch_size = batch_size
        self.correct_class = correct_class_enum
        self._trainerloader = trainloader

    def __iter__(self):
        self._batch_iter = iter(self._trainerloader)
        return self

    def __next__(self):
        images, labels = self._batch_iter.next()
        correct_label = self.correct_class.value
        # label points index, first value is target value
        labels = [int(l != correct_label) for l in labels]
        labels = torch.Tensor(labels)
        labels = labels.type(torch.cuda.LongTensor)
        return images, labels

    def next(self):
        return self.__next__()

    def get_ordered_iterator(self, incorrect_per_correct):
        return self._data_iterator(self._trainerloader, self._batch_size, self.correct_class, incorrect_per_correct)

    class _data_iterator():

        def __init__(self, train_loader, batch_size, correct_label_enum, incorrect_per_correct):
            self.data = []
            self.labels = []
            self.batch_size = batch_size

            output_correct_label = 0
            output_wrong_label = 1

            correct = []
            wrong = []
            correct_label_tensor = torch.LongTensor([correct_label_enum.value])
            for batch, label in iter(train_loader):
                data_object = list(map(lambda img, lab: {'img': img, 'lab': lab}, batch, label))
                correct = correct + [o['img'] for o in data_object if o['lab'] == correct_label_tensor]
                wrong = wrong + [o['img'] for o in data_object if o['lab'] != correct_label_tensor]
                for o in data_object:
                    cos = o['lab']

            for correct_img in correct:
                self.data.append(correct_img)
                self.labels.append(output_correct_label)
                self.data = self.data + wrong[0:incorrect_per_correct]
                del wrong[0:incorrect_per_correct]
                self.labels = self.labels + [output_wrong_label] * (incorrect_per_correct)
            return

        def __iter__(self):
            return self

        def __next__(self):
            batch_size = self.batch_size
            if(len(self.data) < batch_size):
                raise StopIteration

            images = self.data[0:batch_size]
            del self.data[0:batch_size]

            labels = self.labels[0:batch_size]
            del self.labels[0:batch_size]

            images = torch.stack(images)

            labels = torch.Tensor(labels)
            labels = labels.type(torch.cuda.LongTensor)
            return images, labels

        def next(self):
            return self.__next__()



class classes(Enum):
    PLANE = 0
    CAR = 1
    BIRD = 2
    CAT = 3
    DEER = 4
    DOG = 5
    FROG = 6
    HORSE = 7
    SHIP = 8
    TRUCK = 9

