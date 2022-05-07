import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import io
from seq_clean import *
import fitz


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_image, data_transform):
        """Initialization"""
        self.list_image = list_image
        self.data_transform = data_transform

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_image)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        # Load data and get label
        X = self.data_transform(self.list_image[index])
        return X


class SeqClassifier:
    def __init__(self, path):
        model_path = path

        self.data_transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            # transforms.Normalize([0.9673, 0.9673, 0.9673], [0.1414, 0.1415, 0.1415])
                            transforms.Normalize([0.9669, 0.9669, 0.9669], [0.1405, 0.1405, 0.1405])
                        ])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()

    def mupdf_convert(self, path):
        """ convert pdf to images
            Args:
                pdf_path: a str. ex: "./data/001.pdf/"
            Returns:
                page: a list of Pil images [<PIL obj>, <PIL obj>...].
        """
        pdfDoc = fitz.open(path)
        images = []
        for pg in range(pdfDoc.pageCount):
            page = pdfDoc[pg]
            rotate = int(0)
            # default size：792X612, dpi=96
            # (1.33333333-->1056x816)   (2-->1584x1224)
            zoom_x = 2
            zoom_y = 2
            mat = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
            pix = page.getPixmap(matrix=mat, alpha=False)
            image = pix.tobytes()
            image = Image.open(io.BytesIO(image))
            images.append(image)
        return images

    def detect_sequence_page(self, pdf_path):
        """ detect pdf page if contains sequence
            Args:
                pdf_path: a str. ex: "./data/001.pdf/"
            Returns:
                page: a list [int, int]. ex: [10, 20] means sequence starts from 10 to 20, inclusively.
                        or 'broken'
        """
        # Try whether the pdf can be open with mupdf or not. if not, return broken.
        try:
            images = self.mupdf_convert(pdf_path)
        except Exception as e:
            print("broken pdf file {}".format(pdf_path))
            print(e)
            return None, pdf_path, 'broken'
        # Load images, use pretrained bio-seq classification model to detect start and end page.
        data = Dataset(images, self.data_transform)
        data_gen = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False)
        res = []
        for x in data_gen:
            out = self.model(x.cuda())
            r = out.argmax(1).cpu().detach().numpy()
            res.extend(r)
        # retrieve start page and end page according to the result of classification
        start_page = 0
        end_page = len(res) - 1
        page = []
        while start_page < len(res):
            if res[start_page] == 1:
                page.append(start_page + 1)
                break
            start_page += 1
        while end_page >= start_page:
            if res[end_page] == 1:
                page.append(end_page + 1)
                break
            end_page -= 1
        if len(page) == 0:
            return images, pdf_path, []
        # Add one more page at beginning and end, respectively.
        if page[0] > 0 and page[1] < len(res) - 1:
            page[0] -= 1
            page[1] += 1
        return images, pdf_path, page

