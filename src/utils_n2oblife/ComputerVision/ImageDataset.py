import os
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms 


class CurrentImage:
    def __init__(self, id=0, path:str=None) -> None:
        """A function to initiate the current image.

        Args:
            path (str): path of the dataset.
            id (int, optional): current image's id in the dataset. Defaults to 0.
        """
        self.path:str = path
        self.image:Image
        self.id:int
        self.load_image(id=id, path=path)
    

    def load_image(self, id:int, path:str=None)->None:
        """A function to initiate the current image and change it afterward.

        Args:
            path (str): path of the dataset.
            id (int): current image's id in the dataset.
        """
        if path:
            self.path = path
        elif not(self.path) :
            raise ValueError("The path needs to be instanciated")
        self.image = Image.open(self.path)
        self.id = id
        #map = self.image.load() to get the map of pixels

    def plot_image(self):
        self.image.show()


class CurrentImageToManipulate(CurrentImage):
    def __init__(self, id=0, path: str = None) -> None:
        super().__init__(id, path)
        self.dim_x:int
        self.dim_y:int
        self.current_border = []
        self.load_image_to_manipulate(id=id, path=path)
    
    def load_image_to_manipulate(self, id: int, path: str = None) -> None:
        self.dim_x, self.dim_y = self.image.size
        self.current_border = []
        return super().load_image(id, path)

    def get_pixel(self, x:int, y:int)->tuple[int,int,int]:
        """Get the value of a pixel.

        Args:
            x (int): Position on x axis from 0 to (dim_x-1)
            y (int): Position on y axis from 0 to (dim_y-1)

        Returns:
            tuple[int,int,int]: Tuple of RGB value from 0 to 255
        """
        assert(0<=x<self.dim_x and 0<=y<self.dim_y), ValueError(
            f"The pixel need to be in this {self.dim_x}x{self.dim_y} image with positive values, but your values are ({x},{y})")
        return self.image.load()[x,y]
    
    def pix_is_null(self, x:int, y:int)->bool:
        return self.get_pixel(x,y) == (0, 0, 0)
    
    def is_pix_mask(self, x:int, y:int, pixel=(255, 255, 255))->bool:
        """Compares the pixel x,y with another. By default to white pixel. 

        Args:
            x (int): Position on x axis from 0 to (dim_x-1)
            y (int): Position on y axis from 0 to (dim_y-1)
            value (tuple, optional): Pixel value. Defaults to (255,255,255)=(white).
        """
        return self.get_pixel(x,y) == pixel
              
    def define_border(self, x:int, y:int)->None:
        """Recursive function which looks around a pixel to add it to the current border research list.
        We assume there is only one pixel around a given mask pixel which is a border.

        Args:
            x (int): Position on x axis from 0 to (dim_x-1)
            y (int): Position on y axis from 0 to (dim_y-1)
        """
        assert(self.is_pix_mask(x,y)), IndexError(
            f"The pixel from which we should define a border need to be a masked pixel : ({x}, {y})")
        if not (x,y) in self.current_border :
            self.current_border.append((x, y))
            self.fill_pixel(x,y,(255,0,0))
            for val_y in (-1, 0, 1):
                for val_x in (-1, 0, 1):
                    if not (val_x==0 and val_y==0):
                        if 0<=x+val_x<self.dim_x and 0<=y+val_y<self.dim_y:
                            if self.is_pix_mask(x+val_x, y+val_y) :
                                self.define_border(x+val_x, y+val_y)                        

    def trace_mask(self):
        for _, x, y in self.loop_pixels():
            if self.is_pix_mask(x,y):
                self.define_border(x,y)
                self.from_border_to_mask()
                self.current_border = [] 

    def fill_pixel(self, x:int, y:int, new_pix:tuple[int])->None:
        self.image.load()[x,y] = new_pix
    
    def loop_pixels(self):
        """Itertate pixels through an image.

        Yields:
            tuple[tuple[int], int, int]: pixel value as tuple of 3 int and its position (x,y)
        """
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                #print(f'This are the values of x:{self.dim_x} and y:{self.dim_y}')
                yield self.image.load()[x,y], x,y 
    
class ImageDataset:
    def __init__(self, path:str=None) -> None:
        """A Dataset class to manipulate images with the PIL library.

        Args:
            path (str): Path to the dataset folder.
        """
        self.path = path
        self.all_images = []
        self.current_image = None
        self.load_dataset(path=path)
    
    def __len__(self)->int:
        return len(self.all_images)
    
    def load_dataset(self, path:str=None)->None:
        """A function to initiate the dataset and change it afterward.
        Labels managed in another class.

        Args:
            path (str): Path to the dataset.
        """
        if path:
            self.path = path
        if self.path:
            self.all_images = os.listdir(self.path)
            if not self.current_image:
                self.current_image = CurrentImageToManipulate(
                    path=self.path+'/'+self.all_images[0])
            self.load_current_image(id=0)
        else :
            raise ValueError("The path to the dataset must be instanciated")
        
    def load_current_image(self, id:int)->None:
        self.current_image.load_image(id=id,
                                      path=self.path+'/'+self.all_images[id])
    
    def plot_current_image(self)->None:
        self.current_image.image.show()

    def save_current_image(self, save_file='.')->None:
        file_name = self.all_images[self.current_image.id]        
        self.current_image.image.save(save_file+'/'+file_name)
    
    def save_dataset(self, save_file='.')->None:
        for _ in self.loop_all_images():
            self.save_current_image(save_file)
        
    def loop_all_images(self):
        """Itertaes over all the images of the current dataset.

        Yields:
            int: id of the current image
            CurrentImage: Custom class of image
        """
        n_images = len(self.all_images)
        for id in range(n_images):
            self.load_current_image(id=id)
            yield self.current_image
        
class ImageDatasetTorch(ImageDataset):
    def __init__(self, path:str=None, label_path:str=None) -> None:
        super().__init__(path)
        self.label_path = label_path
        self.all_labels = []
        self.current_label = None
        self.transform = transforms.Compose([transforms.PILToTensor()]) 
        self.load_labels(label_path=label_path)

    def load_current_label(self, id:int)->None:
        self.current_label.load_image(id=id,
                                      path=self.label_path+'/'+self.all_labels[id])

    def load_labels(self, label_path:str=None)->None:
        """A function to initiate the labels and change it afterward

        Args:
            label_path (str): Path to the labels of a dataset.
        """
        # TODO check if data and labels are well associated
        if label_path:
            self.label_path = label_path
        if self.label_path:
            self.all_labels = os.listdir(self.label_path)
            if not self.current_label:
                self.current_label = CurrentImage(
                    path=self.label_path+'/'+self.all_labels[0])
            self.load_current_label(id=0)
        else :
            raise ValueError("The path to the dataset must be instanciated")

    def plot_current_label(self)->None:
        self.current_label.image.show()

    def save_current_label(self, save_file='.')->None:
        file_name = self.all_labels[self.current_label.id]
        # TODO handle the case where names are the same for image and label and add something to differentiate        
        self.current_label.image.save(save_file+'/'+file_name)
    
    def save_dataset(self, save_file='.')->None:
        for _ in self.loop_all_dataset():
            self.save_current_image(save_file)  
            self.save_current_label(save_file)

    def loop_all_dataset(self):
        """Itertaes over all the images and labels of the current dataset and yields two tensors.

        Yields:
            int: id of the current image and label
            Tensor: Custom class of image for image
            Tensor: Custom class of image for label
        """
        n_images = len(self.all_images)
        for id in range(n_images):
            self.load_current_image(id=id)
            self.load_current_label(id=id)
            yield id, (self.transform(self.current_image.image), 
                       self.transform(self.current_label.image))