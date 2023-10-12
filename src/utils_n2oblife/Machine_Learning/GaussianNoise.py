import torch 
import torchvision.transforms as transforms


# Transformation to add gaussian noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        tensor = self.to_tensor(image)
        tensor += torch.randn_like(tensor) * self.std + self.mean
        return transforms.ToPILImage()(tensor)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
