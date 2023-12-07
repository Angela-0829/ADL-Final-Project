'''Data processing utils.'''
import os
import json
from typing import List

import yake
import numpy as np
from tqdm.auto import tqdm
from keybert import KeyBERT
from scipy import sparse
from scipy.sparse import lil_matrix
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from research.utils.toolbox import get_free_gpu


def generate_tfidf_voc(preprocessed_docs: List[str], save_folder: str):
    '''Generate gensim_tfidf, vocabulary, and voc2idx mapping and save them to folder.'''
    tfidf_path = os.path.join(save_folder, 'TFIDF.npz')
    voc_path = os.path.join(save_folder, 'vocabulary.npy')
    map_path = os.path.join(save_folder, 'mapping.json')

    vectorizer = TfidfVectorizer(norm=None)
    tfidf_vector = vectorizer.fit_transform(preprocessed_docs)
    vocabulary = np.array(vectorizer.get_feature_names_out())
    mapping = vectorizer.vocabulary_

    sparse.save_npz(os.path.join(save_folder, "TFIDF.npz"), tfidf_vector)
    np.save(os.path.join(save_folder, "vocabulary.npy"), vocabulary)
    with open(os.path.join(save_folder, "mapping.json"), "w", encoding='UTF-8') as map_f:
        json.dump(mapping, map_f)

    # Load tfidf and voc from files.
    tfidf_vector = sparse.load_npz(tfidf_path).toarray()
    vocabulary = np.load(voc_path, allow_pickle=True)
    with open(map_path, 'r', encoding='UTF-8') as map_f:
        mapping = json.load(map_f)

    return tfidf_vector, vocabulary, mapping


def generate_keybert(preprocessed_docs: List[str], mapping: np.ndarray,
                     shape: any, save_folder: str):
    '''Generate keybert target.'''
    keybert_path = os.path.join(save_folder, 'KeyBERT.npz')
    if not os.path.exists(keybert_path):
        kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        keybert_vector = lil_matrix((shape), dtype='float')
        for i, doc in enumerate(tqdm(preprocessed_docs)):
            keywords = kw_model.extract_keywords(
                doc, stop_words='english', top_n=10000)
            for k in keywords:
                # skip words not in vocab
                if k[0] not in mapping:
                    continue
                keybert_vector[i, mapping[k[0]]] = k[1]

        print("saving KeyBERT.npz")
        sparse.save_npz(os.path.join(keybert_path), keybert_vector.tocsr())

    # Load keybert from file.
    keybert_vector = sparse.load_npz(keybert_path).toarray()
    return keybert_vector


def generate_yake(preprocessed_docs: List[str], mapping: np.ndarray,
                  shape: any, save_folder: str):
    '''Generate yake target.'''
    yake_path = os.path.join(save_folder, 'Yake.npz')
    if not os.path.exists(yake_path):
        kw_extractor = yake.KeywordExtractor(lan="en", n=1,
                                             dedupLim=0.9,
                                             dedupFunc='seqm',
                                             windowsSize=1,
                                             top=10000,
                                             features=None)
        yake_vector = lil_matrix((shape), dtype='float')
        for i, doc in enumerate(tqdm(preprocessed_docs)):
            keywords = kw_extractor.extract_keywords(doc)
            for k in keywords:
                # skip words not in vocab
                if k[0] not in mapping:
                    continue
                # smaller score, more important. 1 - x for all scores
                yake_vector[i, mapping[k[0]]] = 1 - k[1]

        print("saving Yake.npz")
        sparse.save_npz(os.path.join(yake_path), yake_vector.tocsr())

    # Load yake from file.
    yake_vector = sparse.load_npz(yake_path).toarray()
    return yake_vector


def get_document_labels(preprocessed_docs: List[str],
                        data_folder: str = "./", dataset: str = "20news"):
    '''
    Generate target and put them under "data/precompute_target/data_name/" .
    args:
      preprocessed_docs: document after processing.
      data_folder: relative path for the data folder.

    Prepare all needed path.
    '''
    config_dir = os.path.join(data_folder, f"precompute_target/{dataset}")
    os.makedirs(config_dir, exist_ok=True)

    # Read precompute labels from files.
    # TFIDF && vocabulary
    tfidf_vector, vocabulary, _ = generate_tfidf_voc(
        preprocessed_docs, config_dir)

    # # Keybert
    # keybert_vector = generate_keybert(preprocessed_docs, mapping,
    #                                   tfidf_vector.shape, config_dir)

    # # Yake
    # yake_vector = generate_yake(preprocessed_docs, mapping,
    #                             tfidf_vector.shape, config_dir)

    labels = {'tf-idf': tfidf_vector,
              'keybert': None, 'yake': None}

    return labels, vocabulary


def get_document_embs(preprocessed_docs: list[str], encoder_type: str, device: str = None):
    '''
    Returns embeddings(input) for document decoder

            Parameters:
                    preprocessed_docs (list): 
                    model_name (str):
            Returns:
                    doc_embs (array):
                    model (class):
    '''
    print('Getting preprocess documents embeddings')
    if device is None:
        device = get_free_gpu()
    if encoder_type == 'average':
        model = SentenceTransformer(
            "average_word_embeddings_glove.840B.300d", device=device)
        doc_embs = np.array(model.encode(preprocessed_docs,
                            show_progress_bar=True, batch_size=16))
    elif encoder_type == 'doc2vec':
        doc_embs = []
        preprocessed_docs_split = [doc.split() for doc in preprocessed_docs]
        documents = [TaggedDocument(doc, [i])
                     for i, doc in enumerate(preprocessed_docs_split)]
        model = Doc2Vec(documents, vector_size=200, workers=4)
        for doc_split in preprocessed_docs_split:
            doc_embs.append(model.infer_vector(doc_split))
        doc_embs = np.array(doc_embs)
    elif encoder_type == 'sbert':
        model = SentenceTransformer(
            "all-mpnet-base-v2", device=device)
        doc_embs = np.array(model.encode(preprocessed_docs,
                            show_progress_bar=True, batch_size=16))
    elif encoder_type == 'mpnet':
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        doc_embs = np.array(model.encode(preprocessed_docs))
    elif encoder_type == 'st5':
        model = SentenceTransformer('sentence-transformers/sentence-t5-large', device=device)
        doc_embs = np.array(model.encode(preprocessed_docs))
    elif encoder_type == 'minilm':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        doc_embs = np.array(model.encode(preprocessed_docs))
    else:
        print(f"Encoder type {encoder_type} not implemented.")
        raise NotImplementedError
        # TODO: Self define encoder model implement. Please implement encoder model in model folder.
        # if model is None:
        #     model = Black(device, encoder_type, num_classes)
        # doc_embs = model.encode_all(preprocessed_docs)

    del model
    return doc_embs
