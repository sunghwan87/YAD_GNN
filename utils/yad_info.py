import os
import re

site_list = ['KAIST', 'SNU', 'Gachon', 'Samsung']

def parse_yad_id(subject_id):
    id_num = re.findall('\d+', subject_id)[0]
    return site_list[int(id_num[0])-1]

TR_dict = {
    'KAIST': 1.5,
    'SNU': 2.0,
    'Samsung': 2.0,
    'Gachon': 3.0
}