import pickle
import numpy as np
from dex_frax.acceptance import export_training


def test_training_export_keeps_original_time_map(tmp_path):
    p=tmp_path/'motion.pkl';d=dict(dof_pos=np.zeros((3,29),np.float32),joint_names=['q'+str(i) for i in range(29)],root_pos=np.zeros((3,3),np.float32),root_rot=np.tile([0.,0.,0.,1.],(3,1)),source_time_s=np.array([0.,.01,.02]),source_frame_float=np.array([0.,.3,.6]),source_fps=30.)
    p.write_bytes(pickle.dumps(d));out=tmp_path/'training.npz'
    export_training(p,out,dict(assets={'mjcf/dex_29.xml':'model'},fingerprint='test'))
    with np.load(out,allow_pickle=False) as v:
        for key in ('source_time_s','source_frame_float','source_fps'):np.testing.assert_array_equal(v[key],d[key])
