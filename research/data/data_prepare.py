'''Data preparation for training and testing.'''
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from research.data.data_loader import load_document
from research.data.data_processing import get_document_embs


class DocDataset(Dataset):
    '''Personalized dataset for document embedding.'''

    def __init__(self, docs, doc_embs):
        self.docs = docs
        self.embs = doc_embs

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        return self.docs[index], self.embs[index]


class AdvDataset(Dataset):
    '''Personalized dataset for adv document embedding.'''

    def __init__(self, docs, doc_embs, domains):
        self.docs = docs
        self.embs = doc_embs
        self.domains = domains

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        return self.docs[index], self.embs[index], self.domains[index]


def generate_data_split(config, dataset):
    '''
    Generate train, val, test split.
    '''
    train_size = int(0.8 * len(dataset))
    val_size = int((len(dataset) - train_size) / 8)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    # Training data ratio split
    remain_size = int(len(train_dataset) * config['train_ratio'])
    train_dataset, remain_dataset = torch.utils.data.random_split(
        train_dataset, [remain_size, len(train_dataset) - remain_size])

    return train_dataset, val_dataset, test_dataset, remain_dataset


def prepare_dataset(config, sent_list, doc_embs):
    '''Prepare train, val, test, and external dataset'''
    # Use half of the original dataset for training and half for external data
    data_size = int(len(sent_list) / 2)
    dataset = DocDataset(sent_list[:data_size], doc_embs[:data_size])

    # Prepare external dataset
    if config['external_dataset'] != config['dataset']:
        print(f"Start loading external dataset: {config['external_dataset']}")
        external_sents = load_document(
            dataset=config['external_dataset'])['documents']
        external_sents = external_sents[:min(len(external_sents), data_size)]
        external_embs = get_document_embs(
            external_sents, config['external_encoder'])
        external_dataset = DocDataset(external_sents, external_embs)
    else:
        print("Using half of the original dataset as external data")
        external_dataset = DocDataset(
            sent_list[data_size:], doc_embs[data_size:])

    train_dataset, val_dataset, test_dataset, _ = generate_data_split(
        config, dataset)

    return train_dataset, val_dataset, test_dataset, external_dataset


def prepare_adv_additional_data(train_dataset, additional_dataset, surrogate, config):
    '''Prepare additional data for adv training'''
    # Use external data to generate additional data
    sent_list = []
    for sents, _ in additional_dataset:
        sent_list.append(sents)
    additional_embs = surrogate.encode(sent_list)

    # Prepare additional dataloader
    additional_dataset = DocDataset(sent_list, additional_embs)

    # Use weighted random sampler to sample train+additional data
    weights = [len(additional_dataset)] * len(train_dataset) + \
        [len(train_dataset)] * len(additional_dataset)

    adv_sents, adv_embs, adv_labels = [], [], []
    for idx, (sents, embs) in enumerate(ConcatDataset([train_dataset, additional_dataset])):
        adv_sents.append(sents)
        adv_embs.append(embs)
        if idx < len(train_dataset):
            adv_labels.append(0)
        else:
            adv_labels.append(1)
    adv_train_dataset = AdvDataset(adv_sents, adv_embs, adv_labels)

    sampler = WeightedRandomSampler(
        weights, len(adv_train_dataset), replacement=True)
    return DataLoader(dataset=adv_train_dataset,
                      batch_size=config['batch_size'],
                      sampler=sampler)


def prepare_additional_data(train_dataset, additional_dataset, surrogate, config):
    '''Prepare additional data for training'''
    # Use validation data to generate additional data
    sent_list = []
    for sents, _ in additional_dataset:
        sent_list.append(sents)
    additional_embs = surrogate.encode(sent_list)

    # Prepare additional dataloader
    additional_dataset = DocDataset(sent_list, additional_embs)

    # Use weighted random sampler to sample train+additional data
    weights = [len(additional_dataset)] * len(train_dataset) + \
        [len(train_dataset)] * len(additional_dataset)
    train_dataset = ConcatDataset([train_dataset, additional_dataset])
    sampler = WeightedRandomSampler(
        weights, len(train_dataset), replacement=True)
    return DataLoader(dataset=train_dataset,
                      batch_size=config['batch_size'],
                      sampler=sampler)
