import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SMAPE(y_pred, y_true, epsilon=1e-10):
    """
    Вычисляет SMAPE (Symmetric Mean Absolute Percentage Error) между истинными и предсказанными значениями.
    
    Аргументы:
        y_true (np.ndarray): Истинные значения, форма (n,) или (n, k).
        y_pred (np.ndarray): Предсказанные значения, той же формы, что y_true.
        epsilon (float): Малое значение для предотвращения деления на ноль.
    
    Возвращает:
        float: Значение SMAPE.
    """
    # Преобразуем входные данные в массивы NumPy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Проверяем, что формы совпадают
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую форму")
    
    # Вычисляем абсолютную ошибку
    abs_error = np.abs(y_true - y_pred)
    
    # Вычисляем знаменатель: max(|y_i| + |\hat{y_i}|, \epsilon)
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), epsilon)
    
    # Вычисляем SMAPE для каждого элемента
    smape_terms = abs_error / denominator
    
    # Усредняем и умножаем на 2/n
    n = y_true.size
    smape_value = (2 / n) * np.sum(smape_terms)
    
    return smape_value

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    
    return mae,mse,rmse,mape,mspe,smape