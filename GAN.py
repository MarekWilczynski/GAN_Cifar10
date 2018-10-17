import torch
import data_manager
import cv2 as cv
import numpy as np

from network_session_handling import load_session

import gan_networks

def show_image(image):
    image = image.detach()
    num = image.numpy()
    num = np.flip(num.flatten(3).reshape(32, 32, 3), 2)
    num = np.rot90(num, axes=(1, 0))
    num = num - np.min(num)
    num = num / np.max(num)
    cv.imshow("test_image", num)
    cv.waitKey(0)

discriminator, training_data, generator, generator_data = load_session('saved_session')
print(training_data)
print(generator_data)
batch_size = 1

data_loader = data_manager.data_loader(training_data['discriminated_class'], batch_size, False)
counter = 0
with torch.no_grad():
    for image, label in data_loader:
        output = discriminator(image)
        output_np = output.cpu().numpy()

        #print("Żaba: %f \n Coś innego: %f" % (output_np[0,0], output_np[0,1]))
        #show_image(image)
        i = int(np.argmax(output.detach()))
        label = int(label.to('cpu'))
        if(i == 0 and i == label):
            counter = counter + 1
print(counter)

