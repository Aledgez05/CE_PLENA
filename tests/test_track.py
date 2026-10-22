"""La etiqueta no entra en X y la partición no se solapa."""

import numpy as np

from plena.data import load, scale, split


def test_etiqueta_fuera_de_x():
    x, y, names = load()
    assert "depressed" not in names
    assert x.shape == (1429, 20)
    assert int(y.sum()) == 238


def test_particion_sin_solape():
    _, y, _ = load()
    train, val, test = split(y)
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(train) + len(val) + len(test) == len(y)
    assert np.bincount(y[train]).tolist() == [714, 142]
    assert np.bincount(y[val]).tolist() == [238, 47]
    assert np.bincount(y[test]).tolist() == [239, 49]


def test_escalado_solo_con_train():
    x, y, names = load()
    train, val, _ = split(y)
    apply, used = scale(x[train], names)
    assert used[-1] == "no_lasting_investmen__faltante"
    scaled = apply(x[val])
    assert scaled.shape[1] == 21
    assert np.isfinite(scaled).all()
