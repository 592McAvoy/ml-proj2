"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch

#from network.misc_functions import get_example_params, save_class_activation_images, tensor2rgb


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        # for module_pos, module in self.model.features._modules.items():
        #     x = module(x)  # Forward
        #     if int(module_pos) == self.target_layer:
        #         x.register_hook(self.save_gradient)
        #         conv_output = x  # Save the convolution output on that layer
        #f, *_ = self.model.features(x)
        f = self.model.features(x)
        f.register_hook(self.save_gradient)
        conv_output = f
        return conv_output, f

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        #x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """

    def __init__(self, model, target_layer=None, cuda=True):
        self.model = model
        # self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)
        self.cuda = cuda

    def generate_cam(self, inputs, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        if isinstance(inputs, tuple):
            input_image, input_dft = inputs
        else:
            input_image = inputs
        conv_output, model_output = self.extractor.forward_pass(inputs)
        #print("conv_output size: ", conv_output.size(),
        #      "model_output size: ", model_output.size())
        B, C = model_output.size()

        if target_class is None:
            target_class = np.argmax(model_output.cpu().data.numpy(), axis=1)
        #print('target_class: ', target_class)

        # Target for backprop
        # one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output = torch.FloatTensor(B, C).zero_()
        for i in range(B):
            one_hot_output[i][target_class[i]] = 1
        #print('one_hot_output: ', one_hot_output)
        if self.cuda:
            one_hot_output = one_hot_output.cuda()

        # Zero grads
        self.model.zero_grad()
        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()

        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        grad_cams = []
        for i in range(input_image.size(0)):
            # Get hooked gradients
            guided_gradients = self.extractor.gradients.cpu().data.numpy()[i]
            # guided_gradients = guided_gradients.reshape(guided_gradients.shape[0], 1, 1)
            # Get convolution outputs
            target = conv_output.cpu().data.numpy()[i]
            # target = guided_gradients = guided_gradients.reshape(target.shape[0], 1, 1)
            # Get weights from gradients
            # Take averages for each gradient
            weights = np.mean(guided_gradients, axis=(1, 2))
            
            # Create empty numpy array for cam
            cam = np.ones(target.shape[1:], dtype=np.float32)
            # Multiply each weight with its conv output and then, sum
            for j, w in enumerate(weights):
                cam += w * target[j, :, :]
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) -
                                         np.min(cam)+1e-5)  # Normalize between 0-1
            cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
            # cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
            #                                             input_image.shape[3]), Image.ANTIALIAS))/255
            # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
            # supports resizing numpy matrices with antialiasing, however,
            # when I moved the repository to PIL, this option was out of the window.
            # So, in order to use resizing with ANTIALIAS feature of PIL,
            # I briefly convert matrix to PIL image and then back.
            # If there is a more beautiful way, do not hesitate to send a PR.

            grad_cams.append(cam)
        return np.array(grad_cams)

if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
