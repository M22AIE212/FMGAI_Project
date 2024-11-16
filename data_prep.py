import os
import gzip
import json
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Path to the folder containing .json.gz files
image_metadata = pd.read_csv("./abo-images-small/images/metadata/images.csv.gz")
folder_path = './abo-listings/listings/metadata'
data_dict = dict()
all_data = []
strat_column = 'target_class'
train_size = 0.7
val_size = 0.15
test_size = 0.15

# Iterate through all the files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.json.gz'):  # Check if the file is a .json.gz file
        file_path = os.path.join(folder_path, file_name)

        # Open and read the gzipped JSON file line by line
        with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
            print(f"Reading file: {file_name}")
            for idx , line in enumerate(gz_file):  # Read each line
                try:
                    data = json.loads(line)  # Load each line as a separate JSON object
                    for item_name in data['item_name'] :
                      if('en_' in item_name['language_tag']) :
                        data_dict[f"{file_name}_{idx}"] = dict()
                        data_dict[f"{file_name}_{idx}"]["product_name"] = item_name['value']
                        data_dict[f"{file_name}_{idx}"]["hierarchy"] = data.get('node',None)
                        all_image_id = [data.get('main_image_id',None)]
                        if data.get('other_image_id',None) :
                          pass
                          # all_image_id.extend(data.get('other_image_id',None))
                        data_dict[f"{file_name}_{idx}"]["image_id"] = all_image_id
                except json.JSONDecodeError as e:
                    print(f"Could not parse JSON in file {file_name}: {e}")


for key in data_dict.keys() :
  d = data_dict[key]
  d['key'] = key
  all_data.append(d)

ecomm_data = pd.DataFrame((all_data))
ecomm_data = ecomm_data.explode('image_id')
ecomm_data = ecomm_data.dropna()
ecomm_data.drop(columns = ['key'],inplace = True)
ecomm_data["clean_hierarchy"] = ecomm_data.hierarchy.apply(lambda x : x[-1]['node_name'])
ecomm_data = ecomm_data.dropna()
ecomm_data = ecomm_data[ecomm_data.clean_hierarchy.str.find('カテゴリー別') == -1]
ecomm_data["target_class"] = ecomm_data.clean_hierarchy.apply(lambda text :  text.split("/")[-2])
ecomm_data = ecomm_data[ecomm_data['target_class'] != ""]

ecomm_data_agg = ecomm_data.groupby('target_class').agg({'target_class':'count'}).rename(columns = {'target_class':'count'}).reset_index()
target_class_list = ecomm_data_agg[ecomm_data_agg['count'] > 222].target_class.unique()

ecomm_data = ecomm_data[ecomm_data.target_class.isin(target_class_list)]

ecomm_data_rest = ecomm_data[ecomm_data.target_class != 'Cases & Covers']
ecomm_data_cc = ecomm_data[ecomm_data.target_class == 'Cases & Covers']

ecomm_data = pd.concat([ecomm_data_rest,ecomm_data_cc.sample(3000)]).reset_index(drop =True)
ecomm_data_final = pd.merge(ecomm_data,image_metadata[['image_id','path']],on = 'image_id',how = 'inner')

## Train Test Split

label_encoder = LabelEncoder()
ecomm_data_final['label'] = label_encoder.fit_transform(ecomm_data_final[strat_column])
with open("./label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

train_data, temp_data = train_test_split(ecomm_data_final,
                                         stratify=ecomm_data_final[strat_column],
                                         test_size=(val_size + test_size),
                                         random_state=42)

val_data, test_data = train_test_split(temp_data,
                                       stratify=temp_data[strat_column],
                                       test_size=test_size/(test_size + val_size),
                                       random_state=42)

train_data['split'] = 'train'
val_data['split'] = 'validation'
test_data['split'] = 'test'

ecomm_data_final = pd.concat([train_data, val_data, test_data])

print(ecomm_data_final['split'].value_counts())  # To see the distribution of the flags
ecomm_data_final.to_csv("data.csv",index = False)
