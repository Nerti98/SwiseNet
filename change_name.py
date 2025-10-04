import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--f',type=str,default =None,help='input file path')
args = parser.parse_args()
file_path = args.f
file_path = r'datasets/DIV2K/X2'

for name in os.listdir(file_path):
    new_name = name.replace('.','_LR.')
    os.rename(file_path+'/'+name,file_path+'/'+new_name)
