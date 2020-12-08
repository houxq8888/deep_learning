import base64
import json
from labelme import utils
import cv2 as cv
import numpy as np
import random
import os, sys, re, shutil
import tqdm


class DataAugment(object):

    def __init__(self, image_id):
        self.add_saltNoise = False
        self.gaussianBlur = True
        self.changeExposure = True
        self.rotation = True
        self.id = image_id
        self.src = cv.imread(data_dir + '//' + str(self.id) + '.jpg')
        try:
            # 原图的高、宽 以及通道数
            self.row, self.col, self.channel = self.src.shape
        except:
            print('No Such image!---'+str(self.id)+'.jpg')
            sys.exit(0)
        dst1 = cv.flip(self.src, 0, dst=None)
        dst2 = cv.flip(self.src, 1, dst=None)
        dst3 = cv.flip(self.src, -1, dst=None)
        self.flip_x = dst1
        self.flip_y = dst2
        self.flip_x_y = dst3
        cv.imwrite(str(self.id)+'_flip_x'+'.jpg', self.flip_x)
        cv.imwrite(str(self.id)+'_flip_y'+'.jpg', self.flip_y)
        cv.imwrite(str(self.id)+'_flip_x_y'+'.jpg', self.flip_x_y)

    def rotation_fun(self):
        if self.rotation:
            # 参数：旋转中心 旋转度数 scale
            center = cv.getRotationMatrix2D((self.col / 2, self.row / 2), 1, 1)
            # 参数：原始图像 旋转参数 元素图像宽高
            rotated = cv.warpAffine(self.src, center, (self.col, self.row))

            row_flip_x, col_flip_x, channel_flip_x = self.flip_x.shape
            center_flip_x = cv.getRotationMatrix2D((col_flip_x / 2, row_flip_x / 2), 1, 1)
            rotated_flip_x = cv.warpAffine(self.flip_x, center_flip_x, (col_flip_x, row_flip_x))

            row_flip_y, col_flip_y, channel_flip_y = self.flip_y.shape
            center_flip_y = cv.getRotationMatrix2D((col_flip_y / 2, row_flip_y / 2), 1, 1)
            rotated_flip_y = cv.warpAffine(self.flip_y, center_flip_y, (col_flip_y, row_flip_y))

            row_flip_x_y, col_flip_x_y, channel_flip_x_y = self.flip_x_y.shape
            center_flip_x_y = cv.getRotationMatrix2D((col_flip_x_y / 2, row_flip_x_y / 2), 1, 1)
            rotated_flip_x_y = cv.warpAffine(self.flip_x_y, center_flip_x_y, (col_flip_x_y, row_flip_x_y))

            cv.imwrite(str(self.id) + '_Rotation' + '.jpg', rotated)
            cv.imwrite(str(self.id) + '_flip_x' + '_Rotation' + '.jpg', rotated_flip_x)
            cv.imwrite(str(self.id) + '_flip_y' + '_Rotation' + '.jpg', rotated_flip_y)
            cv.imwrite(str(self.id) + '_flip_x_y' + '_Rotation' + '.jpg', rotated_flip_x_y)

    def gaussian_blur_fun(self):
        if self.gaussianBlur:
            dst1 = cv.GaussianBlur(self.src, (5, 5), 0)
            dst2 = cv.GaussianBlur(self.flip_x, (5, 5), 0)
            dst3 = cv.GaussianBlur(self.flip_y, (5, 5), 0)
            dst4 = cv.GaussianBlur(self.flip_x_y, (5, 5), 0)
            cv.imwrite(str(self.id)+'_Gaussian'+'.jpg', dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_Gaussian'+'.jpg', dst2)
            cv.imwrite(str(self.id)+'_flip_y'+'_Gaussian'+'.jpg', dst3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Gaussian'+'.jpg', dst4)

    def change_exposure_fun(self):
        if self.changeExposure:
            # contrast
            reduce = 0.5
            increase = 1.4
            # brightness
            g = 10
            h, w, ch = self.src.shape
            add = np.zeros([h, w, ch], self.src.dtype)
            dst1 = cv.addWeighted(self.src, reduce, add, 1-reduce, g)
            dst2 = cv.addWeighted(self.src, increase, add, 1-increase, g)
            dst3 = cv.addWeighted(self.flip_x, reduce, add, 1 - reduce, g)
            dst4 = cv.addWeighted(self.flip_x, increase, add, 1 - increase, g)
            dst5 = cv.addWeighted(self.flip_y, reduce, add, 1 - reduce, g)
            dst6 = cv.addWeighted(self.flip_y, increase, add, 1 - increase, g)
            dst7 = cv.addWeighted(self.flip_x_y, reduce, add, 1 - reduce, g)
            dst8 = cv.addWeighted(self.flip_x_y, increase, add, 1 - increase, g)
            cv.imwrite(str(self.id)+'_ReduceEp'+'.jpg', dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_ReduceEp'+'.jpg', dst3)
            cv.imwrite(str(self.id)+'_flip_y'+'_ReduceEp'+'.jpg', dst5)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_ReduceEp'+'.jpg', dst7)
            cv.imwrite(str(self.id)+'_IncreaseEp'+'.jpg', dst2)
            cv.imwrite(str(self.id)+'_flip_x'+'_IncreaseEp'+'.jpg', dst4)
            cv.imwrite(str(self.id)+'_flip_y'+'_IncreaseEp'+'.jpg', dst6)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_IncreaseEp'+'.jpg', dst8)

    def add_salt_noise(self):
        if self.add_saltNoise:
            percentage = 0.005
            dst1 = self.src
            dst2 = self.flip_x
            dst3 = self.flip_y
            dst4 = self.flip_x_y
            num = int(percentage * self.src.shape[0] * self.src.shape[1])
            for i in range(num):
                rand_x = random.randint(0, self.src.shape[0] - 1)
                rand_y = random.randint(0, self.src.shape[1] - 1)
                if random.randint(0, 1) == 0:
                    dst1[rand_x, rand_y] = 0
                    dst2[rand_x, rand_y] = 0
                    dst3[rand_x, rand_y] = 0
                    dst4[rand_x, rand_y] = 0
                else:
                    dst1[rand_x, rand_y] = 255
                    dst2[rand_x, rand_y] = 255
                    dst3[rand_x, rand_y] = 255
                    dst4[rand_x, rand_y] = 255
            cv.imwrite(str(self.id)+'_Salt'+'.jpg', dst1)
            cv.imwrite(str(self.id)+'_flip_x'+'_Salt'+'.jpg', dst2)
            cv.imwrite(str(self.id)+'_flip_y'+'_Salt'+'.jpg', dst3)
            cv.imwrite(str(self.id)+'_flip_x_y'+'_Salt'+'.jpg', dst4)

    def json_generation(self):
        image_names = [str(self.id)+'_flip_x', str(self.id)+'_flip_y', str(self.id)+'_flip_x_y']
        if self.gaussianBlur:
            image_names.append(str(self.id)+'_Gaussian')
            image_names.append(str(self.id)+'_flip_x'+'_Gaussian')
            image_names.append(str(self.id)+'_flip_y' + '_Gaussian')
            image_names.append(str(self.id)+'_flip_x_y'+'_Gaussian')
        if self.rotation:
            image_names.append(str(self.id) + '_Rotation')
            image_names.append(str(self.id) + '_flip_x' + '_Rotation')
            image_names.append(str(self.id) + '_flip_y' + '_Rotation')
            image_names.append(str(self.id) + '_flip_x_y' + '_Rotation')
        if self.changeExposure:
            image_names.append(str(self.id)+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_ReduceEp')
            image_names.append(str(self.id)+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_y'+'_IncreaseEp')
            image_names.append(str(self.id)+'_flip_x_y'+'_IncreaseEp')
        if self.add_saltNoise:
            image_names.append(str(self.id)+'_Salt')
            image_names.append(str(self.id)+'_flip_x' + '_Salt')
            image_names.append(str(self.id)+'_flip_y' + '_Salt')
            image_names.append(str(self.id)+'_flip_x_y' + '_Salt')
        for image_name in image_names:
            with open(image_name+".jpg", "rb")as b64:
                temp = base64.b64encode(b64.read())
                res_temp = str(temp).split("'")[1]
                base64_data_original = res_temp
                base64_data = base64_data_original
            match_rotation = re.compile(r'(.*)_Rotation(.*)')
            if match_rotation.match(image_name):
                json_name = image_name.split("_Rotation")[0]
                with open(data_dir + '//' + json_name + ".json", 'r')as js:

                    img_json = cv.imread(json_name + ".jpg")
                    row_json, col_json, channel_json = img_json.shape
                    center_json = cv.getRotationMatrix2D((col_json / 2, row_json / 2), 1, 1)

                    json_data = json.load(js)
                    img = utils.img_b64_to_arr(json_data['imageData'])
                    shapes = json_data['shapes']
                    for shape in shapes:
                        points = shape['points']
                        for point in points:
                            points_rotated = np.dot(center_json, np.array([[point[0]], [point[1]], [1]]))
                            point[0] = points_rotated[0][0]
                            point[1] = points_rotated[1][0]
            else:
                with open(data_dir + '//' + str(self.id) + ".json", 'r')as js:
                    json_data = json.load(js)
                    img = utils.img_b64_to_arr(json_data['imageData'])
                    height, width = img.shape[:2]
                    shapes = json_data['shapes']
                    for shape in shapes:
                        points = shape['points']
                        for point in points:
                            match_pattern2 = re.compile(r'(.*)_x(.*)')
                            match_pattern3 = re.compile(r'(.*)_y(.*)')
                            match_pattern4 = re.compile(r'(.*)_x_y(.*)')
                            if match_pattern4.match(image_name):
                                point[0] = width - point[0]
                                point[1] = height - point[1]
                            elif match_pattern3.match(image_name):
                                point[0] = width - point[0]
                                point[1] = point[1]
                            elif match_pattern2.match(image_name):
                                point[0] = point[0]
                                point[1] = height - point[1]
                            else:
                                point[0] = point[0]
                                point[1] = point[1]
            json_data['imagePath'] = image_name + ".jpg"
            json_data['imageData'] = base64_data
            json.dump(json_data, open(data_dir + '//' + image_name + ".json", 'w'), indent=4)


if __name__ == "__main__":
    # 根据jpg文件和json文件批量生成20类增强数据
    cur_dir = os.getcwd()
    data_dir = cur_dir + "\\data"
    os.chdir(data_dir)
    dirs = os.listdir(data_dir)
    assert dirs != [], "样本文件夹为空，请确认..."

    print("图片增强开始...")
    for i in tqdm.tqdm(range(len(dirs))):
       # id = dirs[i].split('.')[0] + '.' + dirs[i].split('.')[1] + '.' + dirs[i].split('.')[2] + '.' + dirs[i].split('.')[3]
        id = dirs[i].split('.')[0]
        DataAugment.id = id
        dataAugmentObject = DataAugment(image_id=id)
        dataAugmentObject.rotation_fun()
        dataAugmentObject.gaussian_blur_fun()
        dataAugmentObject.change_exposure_fun()
        # dataAugmentObject.add_salt_noise()
        dataAugmentObject.json_generation()
    print("图片增强完成...\n")

    # 批量移动文件
    json_dir = cur_dir + "\\train\\json"
    pic_dir = cur_dir + "\\train\\pic"
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
        print("json文件夹不存在，已重新创建...")
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
        print("pic文件夹不存在，已重新创建...")
    dirs = os.listdir(data_dir)
    for i in tqdm.tqdm(range(len(dirs))):
        file = os.path.join(data_dir, dirs[i])
        name = file.split('.')
        if name[-1] == 'json':
            oldFilePath = os.path.join(data_dir, file)
            newFilePath = os.path.join(json_dir, file)
            shutil.move(file, json_dir)
        else:
            oldFilePath = os.path.join(data_dir, file)
            newFilePath = os.path.join(pic_dir, file)
            shutil.move(file, pic_dir)

    print("数据增强完成...")

