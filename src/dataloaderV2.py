'''
Copyright@ Qiao-Mu(Albert) Ren. 
All Rights Reserved.
This is the code to generate Dataset.
'''

import numpy as np
import cv2
import re
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from preprocessing import gen_RT_matrix, get_rendered_convdata, gen_vis_weight, gasuss_noise
from tqdm import tqdm
class HairNetDataset(Dataset):
    def __init__(self, project_dir, train_flag=1, noise_flag=1):
        '''
        param project_dir: the path of project, such as '/home/albertren/Workspace/HairNet/HairNet-ren'
        param train_flag: train_flag=1 -> generate training dataset, train_flag=0 -> generate testing dataset
        '''
        self.project_dir = project_dir
        self.train_flag = train_flag
        self.noise_flag = noise_flag
        self.toTensor = transforms.ToTensor()
        self.training_data = []
        self.test_data = []
        # generate dataset
        if self.train_flag == 1:
            self.train_index = []
            self.train_index_path = self.project_dir + '/data/index_valid/train.txt'
            with open(self.train_index_path, 'r') as f:
                lines = f.readlines()
                for x in lines:
                    self.train_index.append(x.strip().split(' '))
            self.cache_train_data()
        if self.train_flag == 0:
            self.test_index = []
            self.test_index_path = self.project_dir + '/data/index_valid/test.txt'
            with open(self.test_index_path, 'r') as f:
                lines = f.readlines()
                for x in lines:
                    self.test_index.append(x.strip().split(' '))
            self.cache_test_data()
        print('Dataset generated')

    def cache_train_data(self,):
        for index in tqdm(range(len(self.train_index)), desc='Caching train data'):
            current_index = self.train_index[index]
            current_convdata_index = re.search('strands\d\d\d\d\d_\d\d\d\d\d_\d\d\d\d\d', str(current_index)).group(0)
            current_RT_mat = gen_RT_matrix(self.project_dir+'/data/txt/'+str(current_index[0])+'.txt')
            current_convdata_path = self.project_dir+'/data//convdata/'+str(current_convdata_index)+'.convdata'
            current_convdata = get_rendered_convdata(current_convdata_path, current_RT_mat)
            current_visweight = gen_vis_weight(self.project_dir+'/data/vismap/'+str(current_index[0])+'.vismap')
            if self.noise_flag == 1:
                #path = self.project_dir+'/data/png/'+str(current_index[0])+'.png'
                current_img = cv2.imread(self.project_dir+'/data/png/'+str(current_index[0])+'.png')
                current_img = gasuss_noise(current_img)
                # current_img = self.toTensor(current_img)
            else:
                current_img = cv2.imread(self.project_dir+'/data/png/'+str(current_index[0])+'.png')
                # current_img = self.toTensor(current_img)
            current_img = self.toTensor(current_img.astype(np.float32)/255.0)
            current_dict = {"current_img":current_img, "current_convdata":current_convdata, "current_visweight":current_visweight}
            self.training_data.append(current_dict)
    def cache_test_data(self,):
        for index in tqdm(range(len(self.test_index)), desc='Caching test data'):
            current_index = self.test_index[index]
            current_convdata_index = re.search('strands\d\d\d\d\d_\d\d\d\d\d_\d\d\d\d\d', str(current_index)).group(0)
            current_RT_mat = gen_RT_matrix(self.project_dir+'/data/txt/'+str(current_index[0])+'.txt')
            current_convdata_path = self.project_dir+'/data//convdata/'+str(current_convdata_index)+'.convdata'
            current_convdata = get_rendered_convdata(current_convdata_path, current_RT_mat)
            current_visweight = gen_vis_weight(self.project_dir+'/data/vismap/'+str(current_index[0])+'.vismap')
            if self.noise_flag == 1:
                #path = self.project_dir+'/data/png/'+str(current_index[0])+'.png'
                current_img = cv2.imread(self.project_dir+'/data/png/'+str(current_index[0])+'.png')
                current_img = gasuss_noise(current_img)
                # current_img = self.toTensor(current_img)
            else:
                current_img = cv2.imread(self.project_dir+'/data/png/'+str(current_index[0])+'.png')
                # current_img = self.toTensor(current_img)
            current_img = self.toTensor(current_img.astype(np.float32)/255.0)
            current_dict = {"current_img":current_img, "current_convdata":current_convdata, "current_visweight":current_visweight}
            self.test_data.append(current_dict)
    def __getitem__(self, index):
        if self.train_flag == 1:
            current_img = self.training_data[index]['current_img']
            current_convdata = self.training_data[index]['current_convdata']
            current_visweight = self.training_data[index]['current_visweight']
            return current_img, current_convdata, current_visweight
        else:
            current_img = self.test_data[index]['current_img']
            current_convdata = self.test_data[index]['current_convdata']
            current_visweight = self.test_data[index]['current_visweight']
            return current_img, current_convdata, current_visweight    
    def __len__(self):
        if self.train_flag == 1:
            return len(self.train_index)
        else:
            return len(self.test_index)    

def seperate_data_dir(data_dir):
    import os
    import shutil
    """
    seperate data_dir into txt_dir, vismap_dir,exr_dir,png_dir
    """
    txt_dir = data_dir + '/txt'
    vismap_dir = data_dir + '/vismap'
    exr_dir = data_dir + '/exr'
    png_dir = data_dir + '/png'
    if not os.path.exists(txt_dir):
        os.mkdir(txt_dir)
    if not os.path.exists(vismap_dir):
        os.mkdir(vismap_dir)
    if not os.path.exists(exr_dir):
        os.mkdir(exr_dir)
    if not os.path.exists(png_dir): 
        os.mkdir(png_dir)
    for file in os.listdir(data_dir):
        #file = data_dir+'/'+file
        
        if file.endswith('.txt'):
            shutil.copyfile(data_dir+'/'+file, txt_dir+'/'+file)
        elif file.endswith('.vismap'):
            shutil.copyfile(data_dir+'/'+file, vismap_dir+'/'+file)
        elif file.endswith('.exr'):
            shutil.copyfile(data_dir+'/'+file, exr_dir+'/'+file)
        elif file.endswith('.png'):
            shutil.copyfile(data_dir+'/'+file, png_dir+'/'+file)
        else:
            print(f"{file} is not a valid file")
def validate_files(data_dir):
    """
    Check that png, convdata, and vismap exists.
    Find file names in data_dir/index/list.txt
    """
    import os
    import shutil
    complete_files = 0
    # Check if dir exists:
    if not os.path.exists(data_dir+'/index_valid'):
        os.mkdir(data_dir+'/index_valid')
    else:
        raise Exception(f"{data_dir}/index_valid already exists. Please delete it and try again.")
    with open(data_dir+'/index/list.txt', 'r') as f:
        lines = f.readlines()
        for x in lines:
            x = x.strip().split(' ')[0]
            png_path = data_dir+'/png/'+x+'.png'
            convdata_path = data_dir+'/convdata/'+x[:-3]+'.convdata'
            vismap_path = data_dir+'/vismap/'+x+'.vismap'
            #exr_path = data_dir+'/exr/'+x+'.exr'
            txt_path = data_dir+'/txt/'+x+'.txt'
            # print(png_path)
            # print(convdata_path)
            # print(vismap_path)
            # print(exr_path)
            png_exists = os.path.exists(png_path)
            convdata_exists = os.path.exists(convdata_path)
            vismap_exists = os.path.exists(vismap_path)
            txt_exists = os.path.exists(txt_path)
            # print(f"png_exists: {png_exists}, convdata_exists: {convdata_exists}, vismap_exists: {vismap_exists}, exr_exists: {exr_exists}")
            # break
            if png_exists and convdata_exists and vismap_exists and txt_exists:
                complete_files += 1
                with open(data_dir+'/index_valid/list.txt', 'a') as f:
                    f.write(x+'\n')
                # Move 

    print(complete_files)
def split_to_train_test(index_list_path:str,ratio:float=0.9):
    import random
    import os
    import shutil
    index_list_folder = os.path.dirname(index_list_path)
    with open(index_list_path, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        train_lines = lines[:int(len(lines)*ratio)]
        test_lines = lines[int(len(lines)*ratio):]
        # if not os.path.exists(index_list_path[:-4]+'_train'):
        #     os.mkdir(index_list_path[:-4]+'_train')
        # if not os.path.exists(index_list_path[:-4]+'_test'):
        #     os.mkdir(index_list_path[:-4]+'_test')
        with open(index_list_folder+'/train.txt', 'w') as f:
            f.writelines(train_lines)
        with open(index_list_folder+'/test.txt', 'w') as f:
            f.writelines(test_lines)
        print(f"Train: {len(train_lines)}, Test: {len(test_lines)}")
        return train_lines, test_lines

            
if __name__=="__main__":
    validate_files("../HairNet_training_data/data")
    split_to_train_test("../HairNet_training_data/data/index_valid/list.txt")
        