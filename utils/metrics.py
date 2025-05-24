import numpy as np
from sklearn.metrics import mean_squared_error


def get_cindex(y, p):
    summ = 0
    pair = 0

    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    summ += 1 * (p[i] > p[j]) + 0.5 * (p[i] == p[j])

    if pair != 0:
        return summ / pair
    else:
        return 0


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    y_pred_mean = [np.mean(y_pred) for _ in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for _ in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def get_metrics_reg(y_true, y_pred, with_rm2=False, with_ci=False):
    metrics = dict()
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))

    if with_rm2:
        metrics["rm2"] = get_rm2(y_true.flatten().tolist(), y_pred.flatten().tolist())
    if with_ci:
        metrics["ci"] = get_cindex(y_true.flatten().tolist(), y_pred.flatten().tolist())

    return metrics

