"""Shared loader: IDs and derived-label diagnostics never enter X."""
from pathlib import Path
import numpy as np

def load_training(root='processed', validation_fold=None):
    """Return fit and validation dictionaries; validation retains ALL observations.

    X has shape (observations, candidates, features). Flatten candidate rows for
    binary classifier fitting; reshape positive probabilities for segment choice.
    Masks restrict fitting, never silently remove evaluation failures.
    """
    def collect(files, fitting):
        rows=[]
        for p in files:
            with np.load(p,allow_pickle=False) as d:
                mask=d['training_observation_mask'] if fitting else np.ones(len(d['X']),dtype=bool)
                rows.append((d['X'][mask],d['y'][mask],d['candidate_arc_ids'][mask],d['reference_arc_id'][mask],np.full(mask.sum(),p.stem),d['label_reliable'][mask]))
        if not rows:raise ValueError('No trips found')
        return dict(zip(['X','y','candidate_arc_ids','reference_arc_id','trip_id','label_reliable'],[np.concatenate(x) for x in zip(*rows)]))
    train=[];val=[]
    for p in sorted((Path(root)/'train').glob('*.npz')):
        with np.load(p,allow_pickle=False) as d: fold=int(d['fold'])
        (val if fold==validation_fold else train).append(p)
    if validation_fold is None: val=sorted((Path(root)/'student_test').glob('*.npz'))
    return collect(train,True),collect(val,False)
