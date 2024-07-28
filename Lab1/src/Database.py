import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw
from random import choice
import os
from shutil import copy
import json

run = False
FACTOR_SCALE = 1

class Database:
    def __init__(self, nbr_of_file, repository_name="src/dataset", font_pack_path="src/Font_Pack", img_width=30, img_height=30, font_size=35):
        """
        Initialize the Database class with necessary parameters.

        Args:
            nbr_of_file (int): Number of files to generate.
            repository_name (str): Name of the dataset repository.
            font_pack_path (str): Path to the font pack.
            img_width (int): Width of the generated images.
            img_height (int): Height of the generated images.
            font_size (int): Font size for the text in images.
        """
        self.nbrOfFile = nbr_of_file
        self.alphaNum = "abcdefghijklmnopqrstuvw"  # Alphabet and numbers used for dataset
        self.repository = repository_name
        self.imgWidth = img_width
        self.imgHeight = img_height
        self.fontPackPath = font_pack_path
        self.fontSize = font_size
        self.labels = {}

    def define_labels(self):
        """
        Define and save labels for the characters in the dataset.
        """
        for char in self.alphaNum:
            self.labels[ord(char)] = char
        with open('src/dataset_flattened/labels.json', 'w') as f:
            json.dump(self.labels, f)

    def createDb(self):
        """
        Create the dataset by generating images and updating the flattened dataset.
        """
        for repo in [self.repository, '{}_flattened'.format(self.repository)]:
            if not os.path.exists(repo):
                os.makedirs(repo)
            clearRepository(repo)

        n = 0
        while n < self.nbrOfFile:
            if n % 200 == 0:
                print(n)
            rand_char = choice(self.alphaNum)
            font_files = os.listdir(self.fontPackPath)
            rand_font = self.fontPackPath + "/" + choice(font_files)
            success, img = self.createImage(n + 1, rand_char, rand_font)
            if success:
                n += 1
                self.updateFlattenedDataset(img, rand_char)
            else:
                print('error')

    def updateFlattenedDataset(self, img, char):
        """
        Update the flattened dataset with the generated image.

        Args:
            img (numpy.ndarray): The generated image.
            char (str): The character represented in the image.
        """
        with open(f'{self.repository}_flattened/X_dataset.txt', 'a') as x_db, open(f'{self.repository}_flattened/Y_dataset.txt', 'a') as y_db:
            img_str = " ".join(str(sum(pixel[0:3])) for row in img for pixel in row)
            x_db.write(img_str + '\n')
            y_db.write(str(ord(char)) + '\n')

    def center_text(self, img, font, text, color=(0, 0, 0)):
        """
        Center the text in the image.

        Args:
            img (PIL.Image.Image): The image to draw text on.
            font (PIL.ImageFont.FreeTypeFont): The font used for the text.
            text (str): The text to draw.
            color (tuple): The color of the text.

        Returns:
            PIL.Image.Image: The image with centered text.
        """
        draw = ImageDraw.Draw(img)
        text_width, text_height = draw.textsize(text, font)
        position = ((self.imgWidth - text_width) / 2, (self.imgHeight - text_height) / 2)
        draw.text(position, text, color, font=font)
        return img

    def createImage(self, n, char, font, updt=False, repo=False):
        """
        Create an image with a character.

        Args:
            n (int): The image number.
            char (str): The character to draw.
            font (str): Path to the font file.
            updt (bool): Update flag.
            repo (bool): Repository flag.

        Returns:
            tuple: Success flag and the generated image.
        """
        path = f"{self.repository}/data_{n}_{char}.jpeg"
        if updt:
            path = f"charsPlate/data_{repo}_{n}_{char}.jpeg"

        font = ImageFont.truetype(font, self.fontSize)
        img = Image.new("RGBA", (self.imgWidth, self.imgHeight), (255, 255, 255))
        self.center_text(img, font, char)
        img_original = np.array(img)
        cv2.imwrite(path + '_test.jpeg', img_original)
        return True, img_original

    def updateTestOrTrainingFlattened(self, i, j):
        """
        Update the flattened dataset for training and testing.

        Args:
            i (int): Start index for test set.
            j (int): End index for test set.
        """
        with open(f'{self.repository}_flattened/X_dataset.txt', 'r') as x_db, open(f'{self.repository}_flattened/Y_dataset.txt', 'r') as y_db:
            x_lines = x_db.readlines()
            y_lines = y_db.readlines()

        with open(f'{self.repository}_flattened/X_train.txt', 'w') as x_train, open(f'{self.repository}_flattened/Y_train.txt', 'w') as y_train, \
                open(f'{self.repository}_flattened/X_test.txt', 'w') as x_test, open(f'{self.repository}_flattened/Y_test.txt', 'w') as y_test:
            x_train.write(''.join(x_lines[:i] + x_lines[j:]))
            y_train.write(''.join(y_lines[:i] + y_lines[j:]))
            x_test.write(''.join(x_lines[i:j]))
            y_test.write(''.join(y_lines[i:j]))

    def kFoldCrossValidation(self, k, iteration):
        """
        Perform k-fold cross-validation.

        Args:
            k (int): Number of folds.
            iteration (int): Current iteration.
        """
        repository_names = ("src/dataset_img/training", "src/dataset_img/test")
        for repo in repository_names:
            if not os.path.exists(repo):
                os.makedirs(repo)
            clearRepository(repo)

        step = self.nbrOfFile // k
        index1 = step * (iteration - 1)
        index2 = step * iteration
        self.updateTestOrTrainingFlattened(index1, index2)

        for file in os.listdir(self.repository):
            if index1 < int(file.split('_')[1]) <= index2:
                copy(f"{self.repository}/{file}", "dataset_img/test")
            else:
                copy(f"{self.repository}/{file}", "dataset_img/training")

    def update(self, repo, i):
        """
        Update the repository with new images from README.

        Args:
            repo (str): Repository name.
            i (int): Line index in README.
        """
        with open('README.txt', 'r') as f:
            chars = f.readlines()[i].strip()

        clearRepository(repo)
        rand_font = self.fontPackPath + "/arialbd.ttf"

        for n, char in enumerate(chars):
            self.createImage(n, char, rand_font, True, repo)


def clearRepository(repo_name):
    """
    Clear all files in the given repository.

    Args:
        repo_name (str): The name of the repository to clear.
    """
    try:
        for file in os.listdir(repo_name):
            file_path = os.path.join(repo_name, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)
    except Exception as e:
        print(e)


def main():
    """
    Main function to create the database and perform k-fold cross-validation.
    """
    if run:
        test = Database(10000)
        test.createDb()
        test.kFoldCrossValidation(10, 1)
        print("Process finished")


if __name__ == "__main__":
    main()
