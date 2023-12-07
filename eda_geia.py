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
                   name=f"{config['option']}_{config['multiple']}_{config['count']}")
        config['exp_name'] = f"EDA_{config['option']}_{config['multiple']}_{config['count']}"
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
    
    with open(f"/data1/emb_attack/train_dataset_{config['option']}_{config['multiple']}_{config['count']}.pickle", 'rb') as f:
        train_dataset = pickle.load(f)
    with open(f"/data1/emb_attack/val_dataset_{config['multiple']}_{config['count']}.pickle", 'rb') as f:
        val_dataset = pickle.load(f)
    with open(f"/data1/emb_attack/test_dataset_{config['multiple']}_{config['count']}.pickle", 'rb') as f:
        test_dataset = pickle.load(f)

    print(f"Number of training data: {len(train_dataset)}.")
    print(f"Number of validation data: {len(val_dataset)}.")
    print(f"Number of testing data: {len(test_dataset)}.")

    train_loader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=config['batch_size'])
    val_loader = DataLoader(dataset=val_dataset,
                            shuffle=False,
                            batch_size=config['batch_size'])
    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=False,
                             batch_size=config['batch_size'])
    print('load data done')

    llm_attacker = LLMAttackModel(config, doc_embs.shape[1], device)
    if config['model_epoch'] == -1:
        llm_attacker.fit(train_loader, val_loader)
    else:
        print(f"Load model from epoch {config['model_epoch']} without training.")
        llm_attacker.predict(test_loader, config, epoch=config['model_epoch'])
