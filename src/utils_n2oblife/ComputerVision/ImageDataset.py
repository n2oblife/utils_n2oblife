import os
from PIL import Image

class CurrentImage:
    def __init__(self, id=0, path:str=None) -> None:
        """A function to initiate the current image.

        Args:
            path (str): path of the dataset.
            id (int, optional): current image's id in the dataset. Defaults to 0.
        """
        self.load_image(id=id, path=path)
        # self.path:str = path
        # self.image:Image
        # self.format:str = None
        # self.id:int
        # self.dim_x:int
        # self.dim_y:int
    

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
        self.dim_x, self.dim_y = self.image.size
        self.format = self.image.format
        #map = self.image.load() to get the map of pixels

    def get_pixel(self, x:int, y:int)->tuple[int,int,int]:
        """Get the value of a pixel.

        Args:
            x (int): Position on x axis from 0 to (dim_x-1)
            y (int): Position on y axis from 0 to (dim_y-1)

        Returns:
            tuple[int,int,int]: Tuple of RGB value from 0 to 255
        """
        assert(0<=x<self.dim_x and 0<=y<self.dim_y), ValueError(
            f"The pixel need to be in this {self.dim_x}x{self.dim_y} image with positive values")
        return self.image.load()[y,x]

    def plot_image(self):
        self.image.show()
    
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
    
    def fill_pixel(self, x:int, y:int, new_pix:tuple[int])->None:
        self.image.load()[y,x] = new_pix
    
    def loop_pixels(self):
        """Itertate pixels through an image.

        Yields:
            tuple[tuple[int], int, int]: pixel value as tuple of 3 int and its position (x,y)
        """
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                yield self.image.load()[y,x], x,y    

    
class ImageDataset:
    def __init__(self, path:str=None) -> None:
        """A Dataset class to manipulate images with the PIL library.

        Args:
            path (str): Path to the dataset folder.
        """
        self.load_dataset(path=path)
        # self.path = path
        # self.all_images = []
        # self.current_image = CurrentImage()
    
    def load_dataset(self, path:str=None)->None:
        """A function to initiate the dataset and change it afterward

        Args:
            path (str): Path to the dataset.
        """
        #TODO handle the label dataset for machine learning
        if path:
            self.path = path
        if self.path:
            self.all_images = os.listdir(self.path)
            self.load_current_image(id=0)
        else :
            raise ValueError("The path to the dataset must be instanciated")
        
    def load_current_image(self, id:int)->None:
        self.current_image.load_image(id=id,
                                      path=self.path+'/'+self.all_images[id])
    
    def plot_current_image(self)->None:
        self.current_image.plot_image()

    def save_current_image(self, save_file='.')->None:
        self.current_image.image.save(save_file)
    
    def save_dataset(self, save_file='.')->None:
        for img in self.loop_all_images():
            self.save_current_image(save_file)
        
    def loop_all_images(self):
        """Itertaes over all the images of the current dataset.

        Yields:
            CurrentImage: Custom class of image
        """
        for id,_ in enumerate(self.all_images):
            self.load_current_image(id=id)
            yield self.current_image
        