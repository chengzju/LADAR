# LADAR

## Model preparation
    download BERT-base model to ./bert

## Data preparation
    download different datasets to ./data

## Run the code
For Eurlex-4k, perform:
    ./scripts/eur/process.sh
    ./scripts/eur/xmtc.sh
 
For Wiki10-31K, perform:
    ./scripts/w10/process.sh
    ./scripts/w10/xmtc.sh
    
For AmazonCat-13K, perform:
    ./scripts/cat/process.sh
    ./scripts/cat/xmtc.sh
    
For Amazon-670K, perform:
    ./scripts/a670/process.sh
    ./scripts/a670/cluster.sh
    ./scripts/a670/xmtc.sh
    
For Wiki-500K, perform:
    ./scripts/w500/process.sh
    ./scripts/w500/cluster.sh
    ./scripts/w500/xmtc.sh
