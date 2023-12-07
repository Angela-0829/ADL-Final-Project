from research.data.data_loader import load_document
from research.config.base_config import parse_argument
from research.data.data_processing import get_document_embs
from research.data.data_prepare import prepare_dataset, DocDataset
from research.utils.toolbox import get_free_gpu, same_seed
from research.model.attack_model import LLMAttackModel
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import warnings
from eda import EDA
import numpy as np
import pickle
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    # Initialize settings
    same_seed(123)
    args = parse_argument()
    config = vars(args)
    device = get_free_gpu()

     # Prepare data
    data_dict = load_document(dataset=config['dataset'])
    sent_list = data_dict['documents']

    if not config['testing']:
        # Only monitor when not testing
        wandb.init(project="EDA",
                   name=f"produce_data_{config['multiple']}_{config['count']}")
    else:
        # Mini batch config for testing
        config['exp_name'] = 'test'
        sent_list = sent_list[:100]
        config['surrogate_epoch'] = 1

    print(config)

    # Prepare embedding and train, val, test split
    doc_embs = get_document_embs(sent_list, config['blackbox_encoder'])
    train_dataset, val_dataset, test_dataset, external_dataset = prepare_dataset(
        config, sent_list, doc_embs)
    
    sent_list_insert = []
    sent_list_delete = []
    sent_list_replace = []
    sent_list_swap = []
    doc_embs_insert = []
    doc_embs_delete = []
    doc_embs_replace = []
    doc_embs_swap = []
    aug_num = config['multiple']
    from tqdm.auto import trange
    for i in trange(len(train_dataset)):
        for j in range(aug_num):
            try:
                text_insert = EDA(str(train_dataset[i][0]), "insert", config['count'])
                emb = train_dataset[i][1]
                sent_list_insert.append(text_insert)
                doc_embs_insert.append(emb)
            except:
                print("insert")
                print(i)
                print(train_dataset[i][0])
            try:
                text_delete = EDA(str(train_dataset[i][0]), "delete", config['count'])
                emb = train_dataset[i][1]
                sent_list_delete.append(text_delete)
                doc_embs_delete.append(emb)
            except:
                print("delete")
                print(i)
                print(train_dataset[i][0])
            try:
                text_replace = EDA(str(train_dataset[i][0]), "replace", config['count'])
                emb = train_dataset[i][1]
                sent_list_replace.append(text_replace)
                doc_embs_replace.append(emb)
            except:
                print("replace")
                print(i)
                print(train_dataset[i][0])
            try:
                text_swap = EDA(str(train_dataset[i][0]), "swap", config['count'])
                emb = train_dataset[i][1]
                sent_list_swap.append(text_swap)
                doc_embs_swap.append(emb)
            except:
                print("swap")
                print(i)
                print(train_dataset[i][0])
    doc_embs_insert = np.array(doc_embs_insert)
    doc_embs_delete = np.array(doc_embs_delete)
    doc_embs_replace = np.array(doc_embs_replace)
    doc_embs_swap = np.array(doc_embs_swap)
    insert_dataset = DocDataset(sent_list_insert, doc_embs_insert)
    delete_dataset = DocDataset(sent_list_delete, doc_embs_delete)
    replace_dataset = DocDataset(sent_list_replace, doc_embs_replace)
    swap_dataset = DocDataset(sent_list_swap, doc_embs_swap)
    train_dataset_insert = ConcatDataset([train_dataset, insert_dataset])
    train_dataset_delete = ConcatDataset([train_dataset, delete_dataset])
    train_dataset_replace = ConcatDataset([train_dataset, replace_dataset])
    train_dataset_swap = ConcatDataset([train_dataset, swap_dataset])
    print(f"Number of new training data (insert): {len(train_dataset_insert)}.")
    print(f"Number of new training data (delete): {len(train_dataset_delete)}.")
    print(f"Number of new training data (replace): {len(train_dataset_replace)}.")
    print(f"Number of new training data (swap): {len(train_dataset_swap)}.")
    with open(f"eda_data/train_dataset_insert_{aug_num}_{config['count']}.pickle", 'wb') as f:
        pickle.dump(train_dataset_insert, f)
    with open(f"eda_data/train_dataset_delete_{aug_num}_{config['count']}.pickle", 'wb') as f:
        pickle.dump(train_dataset_delete, f)
    with open(f"eda_data/train_dataset_replace_{aug_num}_{config['count']}.pickle", 'wb') as f:
        pickle.dump(train_dataset_replace, f)
    with open(f"eda_data/train_dataset_swap_{aug_num}_{config['count']}.pickle", 'wb') as f:
        pickle.dump(train_dataset_swap, f)
    with open(f"eda_data/val_dataset_{aug_num}_{config['count']}.pickle", 'wb') as f:
        pickle.dump(val_dataset, f)
    with open(f"eda_data/test_dataset_{aug_num}_{config['count']}.pickle", 'wb') as f:
        pickle.dump(test_dataset, f)

    print(f"Number of training data: {len(train_dataset)}.")
    print(f"Number of validation data: {len(val_dataset)}.")
    print(f"Number of testing data: {len(test_dataset)}.")

