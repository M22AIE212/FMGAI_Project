# Ecommerce Product Categorization

## Data Preparation : 
Downloading the data
```
bash data.sh
```

Data Preparation
```
python data_prep.py
```

## Clip Training :

### Cross Based Fusion : 

```
python clip_main.py --fusion cross
```

### Concat Based Fusion : 

```
python clip_main.py --fusion concat
```
### Attention Based Fusion : 
```
python clip_main.py --fusion attention_m
```
