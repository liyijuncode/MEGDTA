import copy
import os
import argparse
import pandas as pd
import torch
import numpy as np
from utils import logger, train_val, get_metrics_reg, save_pkl
from params import SEED, DEVICE, HP
from graph_loader import CustomTrial, CustomDataLoader, load_data
from models import DTIProtGraphChemGraphECFP


if torch.cuda.is_available():
    logger.info(f"GPU will be used for training ({torch.cuda.get_device_name()})")
else:
    logger.info("CPUs will be used for training")


def run_dataset_test(dataset, model, folds):
    epochs = 800
    df_train_val, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)
    all_folds_best = []

    for fold, idx_val in enumerate(val_folds):
        if fold not in folds:
            continue
        logger.info(f"Testing fold {fold} on {dataset} dataset")

        df_train = df_train_val[~ df_train_val.index.isin(idx_val)]

        test_dl = CustomDataLoader(df=df_test, batch_size=128, device=DEVICE,
                                   e1_key_to_graph=ligand_to_graph,
                                   e2_key_to_graph=protein_to_graph,
                                   e1_key_to_fp=ligand_to_ecfp,
                                   shuffle=False)
        train_dl = CustomDataLoader(df=df_train, batch_size=128, device=DEVICE,
                                    e1_key_to_graph=ligand_to_graph,
                                    e2_key_to_graph=protein_to_graph,
                                    e1_key_to_fp=ligand_to_ecfp,
                                    shuffle=True)

        model_copy = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()
        epoch_to_metrics, best_y_true, best_y_pred = train_val(model=model_copy, optimizer=optimizer, criterion=criterion,
                                     train_dl=train_dl, val_dl=test_dl, epochs=epochs,
                                     score_fn=get_metrics_reg, fold=fold, verbose=True, with_rm2=True, with_ci=True)
        save_pkl(epoch_to_metrics, f"results/{dataset}-fold_{fold}-results.pkl")

        df = pd.DataFrame({
            'True_Value': best_y_true,
            'Predicted_Value': best_y_pred
        })

        csv_filename = f'fold_{fold}_best_predictions_{dataset}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Saved fold {fold} predictions to {csv_filename}")

        all_folds_best.append({
            'fold': fold,
            'best_val_mse': np.min([v['metrics_val']['mse'] for v in epoch_to_metrics.values()]),
            'csv_file': csv_filename
        })

    pd.DataFrame(all_folds_best).to_csv(f'cross_validation_summary_{dataset}.csv', index=False)


def main(folds, datasets, gpu):
    if gpu is not None:
        torch.cuda.set_device(gpu)
    else:
        torch.cuda.set_device(0)

    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    model = DTIProtGraphChemGraphECFP(trial=CustomTrial(hp=HP)).to(DEVICE)
    os.makedirs("results/", exist_ok=True)
    if "davis" in datasets:
        run_dataset_test("davis", model, folds)
    if "kiba" in datasets:
        run_dataset_test("kiba", model, folds)
    if "metz" in datasets:
        run_dataset_test("metz", model, folds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, required=False, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--datasets', type=str, required=False, nargs='+',
                        default=["davis", "kiba", "metz"], choices=["davis", "kiba", "metz"])
    parser.add_argument('--gpu', type=int, required=False, help='GPU id to use (e.g., 0, 1, 2, 3, 4)')
    args = parser.parse_args()
    main(args.folds, args.datasets, args.gpu)
