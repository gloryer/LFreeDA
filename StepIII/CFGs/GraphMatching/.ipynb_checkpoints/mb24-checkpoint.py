import numpy as np
from spektral.data import Dataset, Graph
import scipy.sparse as sp
import os
import glob
from pathlib import Path
import sys
from pathlib import Path
script_path = Path(__file__).resolve().parent.parent
sys.path.append(str(script_path))
from Utils.utils import list_to_spektral_dataset, sparse_to_tuple





def determine_base_cfg_dir(img_path: str) -> str:
    """
    Determine the base directory for CFG embeddings based on the image path.\
    """
    data_dir= Path(__file__).resolve().parent.parent.parent.parent / "data"
    #print(data_dir)

    if "mb24" in img_path:
        parts = img_path.split(os.sep)
        idx = parts.index("mb24")
        month = parts[idx + 1].capitalize()
        return os.path.join(data_dir, "graph_features/mb24", month)
    # Normal dataset mappings
    if "benign_source/dataset1" in img_path:
        return os.path.join(data_dir, "graph_features/benign_source/dataset1")
    if "benign_source/dataset2" in img_path:
        return os.path.join(data_dir,  "graph_features/benign_source/dataset2")
    if "benign_source/dataset3" in img_path:
        return os.path.join(data_dir,  "graph_features/benign_source/dataset3")
    if "benign_source/dataset4" in img_path:
        return os.path.join(data_dir,  "graph_features/benign_source/dataset4")
    if "benign_target/dataset1" in img_path:
        return os.path.join(data_dir,  "graph_features/benign_target/dataset1")
    if "benign_target/dataset2" in img_path:
        return os.path.join(data_dir,  "graph_features/benign_target/dataset2")
    if "benign_target/dataset3" in img_path:
        return os.path.join(data_dir,  "graph_features/benign_target/dataset3")
    if "benign_target/dataset4" in img_path:
        return os.path.join(data_dir,  "graph_features/benign_target/dataset4")
    raise ValueError(f"Cannot determine CFG directory for path: {img_path}")





class GraphData_normal(Dataset):
    def __init__(self, cfg_path, base, pred, label, **kwargs):
        """
        Load a single normal CFG graph matching the given base name.
        :param cfg_path: Directory containing cfg_embeddings for the normal dataset
        :param base: Base filename without extension
        """
        self.cfg_path = cfg_path
        self.base = base
        self.pred = pred
        self.label = label
        super().__init__(**kwargs)

    def read(self):

        # all datasets live under a "cfg_embeddings" subfolder
        cfg_emb_dir = os.path.join(self.cfg_path, 'cfg_embeddings')

        # --- special case for benign_source/dataset1 ---
        if "benign_source/dataset1" in self.cfg_path:
            # base is already "{digit}_goodware", so pull out the digit
            digit = self.base.split('_')[0]
            node_fp   = os.path.join(cfg_emb_dir, f"{digit}_graph_goodware.npz")
            sparse_fp = os.path.join(cfg_emb_dir, f"{digit}_graph_goodware_sparse_matrix.npz")

            # if node_fp:
            #     print(node_fp)

            # bail out early if either file is missing
            if not (os.path.exists(node_fp) and os.path.exists(sparse_fp)):
                return []
            
        else:
            pattern = os.path.join(cfg_emb_dir,  f"*_{self.base}.npz")
            matches = [p for p in glob.glob(pattern) if '_sparse_matrix.npz' not in p]
            #print(matches)
            if not matches:
                return []
            
            node_fp = matches[0]
            sparse_fp = node_fp.replace('.npz', '_sparse_matrix.npz')
            if not os.path.exists(sparse_fp):
                return []

        data = np.load(node_fp)
        sparse_matrix = sp.load_npz(sparse_fp).astype('float32')

        # Skip if too large to handle
        if sparse_matrix.shape[0] > 46000:
            return []

        # Remove self-loops
        adj = sparse_matrix - sp.dia_matrix((sparse_matrix.diagonal()[np.newaxis, :], [0]),
                                            shape=sparse_matrix.shape)
        adj.eliminate_zeros()
        assert np.all(adj.diagonal() == 0)

        adj_triu = sp.triu(adj)
        edges = sparse_to_tuple(adj_triu)[0]

        # Filter by node and edge counts
        if data['x'].shape[0] >= 10 and edges.shape[0] >= 3:
            return [Graph(x=data['x'], a=sparse_matrix, y=self.pred)]
        #else: 
            #print("Too small")
        return []


class GraphData_mb24(Dataset):
    def __init__(self, cfg_path, base, pred, label, **kwargs):
        """
        Load a single mb24 CFG graph matching the given base name.
        Uses the same logic as GraphData_normal, but searches both '0' and '1' subfolders.
        """
        self.cfg_path = cfg_path
        self.base = base
        self.pred = pred
        self.label = label
        super().__init__(**kwargs)

    def read(self):
        # Iterate both class folders
        for cls in ['0', '1']:
            root = os.path.join(self.cfg_path, cls)
            node_fp = os.path.join(root, 'cfg_embeddings', f"{self.base}.npz")
            #print(node_fp)
            if not os.path.exists(node_fp):
                pattern = os.path.join(root, 'cfg_embeddings', f"*_{self.base}_exe.npz")
                #print(pattern)
                matches = [p for p in glob.glob(pattern) if '_exe_sparse_matrix.npz' not in p]
                #print(matches)
                if not matches:
                    continue
                node_fp = matches[0]

            sparse_fp = node_fp.replace('_exe.npz', '_exe_sparse_matrix.npz')
            if not os.path.exists(sparse_fp):
                continue

            data = np.load(node_fp)
            sparse_matrix = sp.load_npz(sparse_fp).astype('float32')

            # Skip if too large to handle
            if sparse_matrix.shape[0] > 46000:
                continue

            # Remove self-loops
            adj = sparse_matrix - sp.dia_matrix((sparse_matrix.diagonal()[np.newaxis, :], [0]),
                                                shape=sparse_matrix.shape)
            adj.eliminate_zeros()
            assert np.all(adj.diagonal() == 0)

            adj_triu = sp.triu(adj)
            edges = sparse_to_tuple(adj_triu)[0]

            # Filter graphs by size
            if data['x'].shape[0] >= 10 and edges.shape[0] >= 3:
                return [Graph(x=data['x'], a=sparse_matrix, y=self.pred)]
            # else: 
            #     print("Too small")
            
        return []


def load_matched_graphs(npz_path, pass_key, pred_key, true_label_key):
    """
    Load matched CFG graphs for images listed in an .npz file.
    """
    data = np.load(npz_path, allow_pickle=True)
    image_paths = data[pass_key]
    pred_labels = data[pred_key]
    true_labels = data[true_label_key]

    matched = []
    #i = 0
    for img_path, pred, label in zip(image_paths, pred_labels, true_labels):
        # Extract filename without extension
        fname = os.path.splitext(os.path.basename(img_path))[0]
        #print(img_path)
        if "benign_source/dataset1" in img_path:
            base = fname  # e.g., "1_goodware"
            #print(base)
        else:
            if '_' in fname and fname.split('_', 1)[0].isdigit():
                base = fname.split('_', 1)[1]
            else:
                base = fname
        #print(base)
        try:
            base_dir = determine_base_cfg_dir(img_path)
            #print(base_dir)
        except ValueError as e:
            print(e)
            continue


        if "mb24" in img_path:
            
            #print(f"{img_path}")
            ds = GraphData_mb24(base_dir, base, pred, label)
            
        else:
  
            #print(f"{img_path}")
            ds = GraphData_normal(base_dir, base, pred, label)
        

        if ds:
            #print(ds[0].y)
            matched.append(ds[0])
        # else:
        #     print(f"No graph representation for {img_path}")
            
                    
        # print("\n")

        # i += 1 
        # if i == 5: 
        #     break 
    return list_to_spektral_dataset(matched)

