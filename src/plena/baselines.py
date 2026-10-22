"""Referencias que sí usan la etiqueta: clase mayoritaria, logística y red ligera."""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

torch.set_num_threads(1)


def best_threshold(y, score):
    """Mayor balanced accuracy en validación. Si empatan, el umbral más cercano a 0.5."""
    values = np.unique(score)
    cuts = np.r_[0.0, (values[:-1] + values[1:]) / 2, 0.5, np.nextafter(1.0, 2.0)]
    quality = np.array([balanced_accuracy_score(y, score >= cut) for cut in cuts])
    winners = np.flatnonzero(np.isclose(quality, quality.max(), atol=1e-12, rtol=0))
    chosen = winners[np.argmin(np.abs(cuts[winners] - 0.5))]
    return float(cuts[chosen]), float(quality[chosen])


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def fit_logistic(x, y, x_val, y_val):
    """Adam y tres valores de L2. Se queda con la balanced accuracy de validación."""
    train = np.column_stack([x, np.ones(len(x))])
    valid = np.column_stack([x_val, np.ones(len(x_val))])
    best = None
    for l2 in (1e-4, 1e-2, 1e-1):
        beta = np.zeros(train.shape[1])
        moment, second = np.zeros_like(beta), np.zeros_like(beta)
        kept, kept_loss = beta.copy(), np.inf
        for step in range(1, 1201):
            grad = train.T @ (_sigmoid(train @ beta) - y) / len(y)
            grad[:-1] += l2 * beta[:-1]
            moment = 0.9 * moment + 0.1 * grad
            second = 0.999 * second + 0.001 * grad**2
            beta = beta - 0.03 * (moment / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
            if step % 10 == 0:
                loss = np.mean(np.logaddexp(0, valid @ beta) - y_val * (valid @ beta))
                if loss < kept_loss:
                    kept, kept_loss = beta.copy(), loss
        score = _sigmoid(valid @ kept)
        threshold, balanced = best_threshold(y_val, score)
        candidate = {"beta": kept, "l2": l2, "threshold": threshold, "val_ba": balanced, "val_loss": kept_loss}
        if best is None or (balanced, -kept_loss) > (best["val_ba"], -best["val_loss"]):
            best = candidate
    return best


def predict_logistic(x, beta):
    return _sigmoid(np.column_stack([x, np.ones(len(x))]) @ beta)


def fit_mlp(x, y, x_val, y_val, seed):
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(x.shape[1], 16), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.001)
    loss_fn = nn.BCEWithLogitsLoss()
    xt, yt = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    xv, yv = torch.tensor(x_val), torch.tensor(y_val, dtype=torch.float32)
    best_loss, best_state, patience = np.inf, None, 0
    for _epoch in range(250):
        net.train()
        for batch in torch.randperm(len(y)).split(64):
            optimizer.zero_grad()
            loss_fn(net(xt[batch]).ravel(), yt[batch]).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(xv).ravel(), yv).item()
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def predict_mlp(net, x):
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(x)).ravel()).numpy()


def majority_prediction(y_train, n):
    label = int(np.argmax(np.bincount(y_train)))
    return np.full(n, label), np.full(n, float(y_train.mean()))
