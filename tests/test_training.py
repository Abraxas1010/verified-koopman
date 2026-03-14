from __future__ import annotations

from pathlib import Path

import verified_koopman as vk
from verified_koopman.utils.training import TrainConfig, train_model


def test_training_smoke(tmp_path: Path) -> None:
    (train, test, dt, meta) = vk.generate_system_data("toy2d", n_traj=32, time=51, seed=0)
    state_dim = int(train[0].shape[-1])
    model = vk.NucleusBottleneckAE(state_dim=state_dim, latent_dim=4, hidden_dim=32, depth=2, dt=dt, nucleus_type="relu")

    out = train_model(
        model=model,
        train=train,
        test=test,
        dt=dt,
        output_dir=tmp_path / "run",
        cfg=TrainConfig(epochs=2, batch_size=256, lr=1e-3),
        device="cpu",
        seed=0,
        extra_meta=meta,
    )

    assert out.best_path.exists()
    assert out.last_path.exists()

