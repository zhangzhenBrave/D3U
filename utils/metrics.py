import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def quantile_loss(target, forecast, q: float) -> float:
    return 2 * np.sum(
        np.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )

def calc_quantile_CRPS(all_preds, true): 

    target = true
    forecast = all_preds

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(np.quantile(forecast[j : j + 1], quantiles[i], axis=1))
        q_pred = np.concatenate(q_pred, axis=0)
        q_loss = quantile_loss(target, q_pred, quantiles[i])
        CRPS += q_loss / denom
    return CRPS / len(quantiles)

def quantile_loss(target, forecast, q: float) -> float:
    return 2 * np.sum(
        np.abs((forecast - target) * ((target <= forecast) * 1.0 - q))
    )


def calc_quantile_CRPS_sum(all_preds, true): 

    target = true.sum(-1)
    forecast = all_preds

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = np.sum(np.abs(target))
    CRPS = 0
    for i in range(len(quantiles)):
        #targer(b,seq_len)
        #forecast(b,sampe,seq_len)
        q_pred = np.quantile(forecast.sum(-1),quantiles[i], axis=1)
      #   print(q_pred.shape)
      #   print(target.shape)
        q_loss = quantile_loss(target, q_pred, quantiles[i])
        CRPS += q_loss /denom
    return CRPS / len(quantiles)
