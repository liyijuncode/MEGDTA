import torch

N_CHEM_NODE_FEAT = 22
N_PROT_NODE_FEAT = 34
N_CHEM_EDGE_FEAT = 12
N_PROT_EDGE_FEAT = 7
N_CHEM_ECFP = 2048

SEED = 47

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

HP = {
 'chem_ecfp_post_fc': True,
 'chem_ecfp_post_fc_dropout_0': 0.5,
 'chem_ecfp_post_fc_n_out_0': 1024,
 'chem_ecfp_post_fc_use_bn': True,
 'chem_ecfp_post_n_fc_layers': 1,
 'final_fc_dropout_0': 0.5,
 'final_fc_dropout_1': 0.5,
 'final_fc_dropout_2': 0.5,
 'final_fc_n_out_0': 1024,
 'final_fc_n_out_1': 1024,
 'final_fc_n_out_2': 2048,
 'final_fc_use_bn': True,
 'final_n_fc_layers': 3,
 'prot_gnn_arch': 'staked',
 'prot_gnn_post_fc': True,
}
