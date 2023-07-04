pip3 install torch==1.10.0 torchvision -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip3 install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip3 install torch-sparse==0.6.13 -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
pip3 install torch-geometric==2.0.4
pip3 install pytorch-lightning==1.6.3


#ssh -X pcuda01
#conda activate YAD_STAGIN
#cd /u4/surprise/tutorials/pytorch_lightning
#python test.py
#strings /usr/lib/libc.so.6 | grep ^GLIBC
