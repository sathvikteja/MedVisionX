import os
import shutil
import random

base_path = "data/processed"

tumor_path = os.path.join(base_path,"tumor")
normal_path = os.path.join(base_path,"normal")

def split(files):

    random.shuffle(files)

    n = len(files)

    train = files[:int(0.7*n)]
    val = files[int(0.7*n):int(0.85*n)]
    test = files[int(0.85*n):]

    return train,val,test


def create_structure():

    for split in ["train","val","test"]:

        for cls in ["tumor","normal"]:

            path = os.path.join(base_path,split,cls)

            os.makedirs(path,exist_ok=True)


def move_files(files,split,cls):

    for f in files:

        src = os.path.join(base_path,cls,f)

        dst = os.path.join(base_path,split,cls,f)

        shutil.move(src,dst)


create_structure()

tumor_files = os.listdir(tumor_path)
normal_files = os.listdir(normal_path)

t_train,t_val,t_test = split(tumor_files)
n_train,n_val,n_test = split(normal_files)

move_files(t_train,"train","tumor")
move_files(t_val,"val","tumor")
move_files(t_test,"test","tumor")

move_files(n_train,"train","normal")
move_files(n_val,"val","normal")
move_files(n_test,"test","normal")

print("Dataset split done")


















