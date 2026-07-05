"""Dump per-(pair) KnowDDI predictions on a fixed split for two-stream fusion.

Rebuilds the BioSNAP SubgraphDataset exactly as train.py does, loads a trained
classifier, runs a shuffle=False forward over the chosen split, and writes
preds[N,200], labels[N,200], polarity[N], pair_ids[N,2] to an .npz. pair_ids come
from the LMDB datum's nodes[:2] (subgraph roots == the drug pair, see
data_processor/subgraph_extraction.py:156 and datasets.py:72-74), read in the
same index order the shuffle=False DataLoader iterates.

Usage (per seed, run from third_party/knowddi/pytorch/):
    python dump_predictions.py -e BioSNAP_seed1 --dataset=BioSNAP --seed=1 \\
        --load_model --num_dig_layers=3 --gsl_rel_emb_dim=24 --MLP_hidden_dim=24 \\
        --MLP_num_layers=3 --MLP_dropout=0.2 \\
        --out_test ../../../runs/fusion/seed_1/knowddi_test.npz \\
        --out_valid ../../../runs/fusion/seed_1/knowddi_valid.npz

NOTE: This script requires the KnowDDI DGL/LMDB runtime. Do NOT import or run
it outside of the KnowDDI environment; it is a manual GPU gate for the user.

DEPLOYMENT: This file is the version-controlled source of truth. The vendored
KnowDDI tree (third_party/knowddi) is a SEPARATE, gitignored git repo, so a
patch cannot be committed there. To run, copy this file into the KnowDDI tree
under its expected name, then run from there:

    cp experiments/fusion/knowddi_dump_predictions.py \\
       third_party/knowddi/pytorch/dump_predictions.py
    cd third_party/knowddi/pytorch && python dump_predictions.py ...

(It must run from third_party/knowddi/pytorch/ so its relative imports and
initialize_experiment(__file__) resolve to the KnowDDI data/experiment dirs.)
"""
import argparse
import logging
import os
import random
from warnings import simplefilter

import lmdb
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import SparseEfficiencyWarning
from torch.utils.data import DataLoader

from data_processor.datasets import SubgraphDataset
from manager.evaluator import init_fn
from model.Classifier_model import Classifier_model
from utils.graph_utils import collate_dgl, deserialize, move_batch_to_device_dgl
from utils.initialization_utils import initialize_experiment, initialize_model

# Replicate train.py environment setup
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def _pair_ids_in_order(db_path: str, db_name: str) -> np.ndarray:
    """Read nodes[0], nodes[1] for LMDB indices 0..num_graphs-1 (= loader order).

    KnowDDI stores each subgraph as a serialised dict whose first field 'nodes'
    contains the ordered node list; nodes[0] and nodes[1] are the two drug roots
    (see data_processor/subgraph_extraction.py:156 and datasets.py:74).  Reading
    them here in strict index order guarantees alignment with the shuffle=False
    DataLoader output accumulated in dump_split().
    """
    env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
    db = env.open_db(db_name.encode())
    with env.begin(db=db) as txn:
        n = int.from_bytes(txn.get(b"num_graphs"), byteorder="little")
        out = np.zeros((n, 2), dtype=np.int64)
        for idx in range(n):
            nodes, *_ = deserialize(txn.get("{:08}".format(idx).encode("ascii"))).values()
            out[idx, 0], out[idx, 1] = int(nodes[0]), int(nodes[1])
    return out


def dump_split(params, classifier, dataset, db_name: str, out_path: str) -> None:
    """Run a shuffle=False forward pass over dataset and write the .npz dump."""
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False,
                        num_workers=params.num_workers, collate_fn=collate_dgl,
                        worker_init_fn=init_fn)
    preds_all, labels_all, pol_all = [], [], []
    classifier.eval()
    with torch.no_grad():
        for batch in loader:
            data, r_labels, polarity = move_batch_to_device_dgl(batch, params.device, multi_type=2)
            pred = nn.Sigmoid()(classifier(data))
            preds_all.append(pred.cpu().numpy())
            labels_all.append(r_labels.cpu().numpy())
            pol_all.append(polarity.cpu().numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    polarity = np.concatenate(pol_all)
    pair_ids = _pair_ids_in_order(params.db_path, db_name)
    assert pair_ids.shape[0] == preds.shape[0], (pair_ids.shape, preds.shape)
    np.savez(out_path, preds=preds, labels=labels, polarity=polarity, pair_ids=pair_ids)
    print(f"wrote {out_path}: {preds.shape[0]} examples")


def process_dataset(params):
    """Replicated verbatim from train.py:19-69 — builds SubgraphDatasets from LMDB."""
    params.db_path = os.path.join(params.main_dir, f'../data/{params.dataset}/digraph_hop_{params.hop}_{params.BKG_file_name}')

    from data_processor.subgraph_extraction import generate_subgraph_datasets
    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train_data = SubgraphDataset(db_path=params.db_path,
                                 db_name='train_subgraph',
                                 raw_data_paths=params.file_paths,
                                 add_traspose_rels=params.add_traspose_rels,
                                 use_pre_embeddings=params.use_pre_embeddings,
                                 dataset=params.dataset,
                                 kge_model=params.kge_model,
                                 dig_layer=params.num_dig_layers,
                                 BKG_file_name=params.BKG_file_name)

    test_data = SubgraphDataset(db_path=params.db_path,
                                db_name='test_subgraph',
                                use_pre_embeddings=params.use_pre_embeddings,
                                dataset=params.dataset,
                                kge_model=params.kge_model,
                                ssp_graph=train_data.ssp_graph,
                                id2entity=train_data.id2entity,
                                id2relation=train_data.id2relation,
                                rel=train_data.num_rels,
                                global_graph=train_data.global_graph,
                                dig_layer=params.num_dig_layers,
                                BKG_file_name=params.BKG_file_name)

    valid_data = SubgraphDataset(db_path=params.db_path,
                                 db_name='valid_subgraph',
                                 use_pre_embeddings=params.use_pre_embeddings,
                                 dataset=params.dataset,
                                 kge_model=params.kge_model,
                                 ssp_graph=train_data.ssp_graph,
                                 id2entity=train_data.id2entity,
                                 id2relation=train_data.id2relation,
                                 rel=train_data.num_rels,
                                 global_graph=train_data.global_graph,
                                 dig_layer=params.num_dig_layers,
                                 BKG_file_name=params.BKG_file_name)

    params.num_rels = train_data.num_rels
    params.global_graph = train_data.global_graph.to(params.device)
    params.aug_num_rels = train_data.aug_num_rels
    params.num_nodes = 35000
    logging.info(f"Device: {params.device}")
    logging.info(f" # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}")

    return train_data, valid_data, test_data


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=SparseEfficiencyWarning)

    # Argparse block replicated from train.py:96-182 exactly, with two new args added.
    parser = argparse.ArgumentParser(description="model params")

    # global
    parser.add_argument('--seed', type=int, default=1111, help="seeds for random initial")
    parser.add_argument("--gpu", type=int, default=3, help="Which GPU to use?")
    parser.add_argument('--disable_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--load_model', action='store_true', help='Load existing model?')
    parser.add_argument("--experiment_name", "-e", type=str, default="default",
                        help="A folder with this name would be created to dump saved models and log files")

    # dataset
    parser.add_argument('--dataset', "-d", type=str, default='drugbank')
    parser.add_argument("--train_file", "-tf", type=str, default="train",
                        help="Name of file containing training triplets")
    parser.add_argument("--valid_file", "-vf", type=str, default="valid",
                        help="Name of file containing validation triplets")
    parser.add_argument("--test_file", "-ttf", type=str, default="test",
                        help="Name of file containing validation triplets")
    parser.add_argument("--kge_model", type=str, default="TransE",
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument("--use_pre_embeddings", type=bool, default=False,
                        help='whether to use pretrained KGE embeddings')
    parser.add_argument('--BKG_file_name', type=str, default='BKG_file')

    # extract subgraphs
    parser.add_argument("--max_links", type=int, default=250000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument("--hop", type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument("--max_nodes_per_hop", "-max_h", type=int, default=200,
                        help="if > 0, upper bound the # nodes per hop by subsampling")
    parser.add_argument('--enclosing_subgraph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='whether to append adj matrix list with symmetric relations')

    # trainer
    parser.add_argument("--eval_every_iter", type=int, default=526,
                        help="Interval of iterations to evaluate the model")
    parser.add_argument("--save_every_epoch", type=int, default=10,
                        help="Interval of epochs to save a checkpoint of the model")
    parser.add_argument("--early_stop_epoch", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        help="Which optimizer to use?")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate of the optimizer")
    parser.add_argument("--lr_decay_rate", type=float, default=0.93,
                        help="adjust the learning rate via epochs")
    parser.add_argument("--weight_decay_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_epochs", "-ne", type=int, default=50,
                        help="numer of epochs")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of dataloading processes")

    # GraphSAGE params
    parser.add_argument("--emb_dim", "-dim", type=int, default=32,
                        help="Entity embedding size")
    parser.add_argument("--num_gcn_layers", type=int, default=2,
                        help="Number of GCN layers")
    parser.add_argument('--gcn_aggregator_type', type=str,
                        choices=['mean', 'gcn', 'pool'], default='mean')
    parser.add_argument("--gcn_dropout", type=float, default=0.2,
                        help="node_dropout rate in GCN layers")

    # gsl_Model params
    parser.add_argument("--num_infer_layers", type=int, default=3,
                        help="Number of infer layers")
    parser.add_argument("--num_dig_layers", type=int, default=3)
    parser.add_argument("--MLP_hidden_dim", type=int, default=16)
    parser.add_argument("--MLP_num_layers", type=int, default=2)
    parser.add_argument("--MLP_dropout", type=float, default=0.2)
    parser.add_argument("--func_num", type=int, default=1)
    parser.add_argument("--sparsify", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--edge_softmax", type=int, default=1)
    parser.add_argument("--gsl_rel_emb_dim", type=int, default=32)
    parser.add_argument("--lamda", type=float, default=0.7)
    parser.add_argument("--gsl_has_edge_emb", type=int, default=1)

    # dump-specific args (new, not in train.py)
    parser.add_argument("--out_test", type=str, required=True,
                        help="Output .npz path for test-split dump")
    parser.add_argument("--out_valid", type=str, required=True,
                        help="Output .npz path for valid-split dump")

    params = parser.parse_args()

    def set_seed(seed):
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    set_seed(params.seed)
    initialize_experiment(params, __file__)

    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device('cuda:%d' % params.gpu)
    else:
        params.device = torch.device('cpu')

    params.file_paths = {
        'train': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.train_file)),
        'valid': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.valid_file)),
        'test': os.path.join(params.main_dir, '../data/{}/{}.txt'.format(params.dataset, params.test_file)),
    }

    # Ensure output directories exist
    os.makedirs(os.path.dirname(os.path.abspath(params.out_test)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(params.out_valid)), exist_ok=True)

    train_data, valid_data, test_data = process_dataset(params)

    # Load trained classifier (requires --load_model flag)
    classifier = initialize_model(params, Classifier_model)

    dump_split(params, classifier, valid_data, 'valid_subgraph', params.out_valid)
    dump_split(params, classifier, test_data, 'test_subgraph', params.out_test)
