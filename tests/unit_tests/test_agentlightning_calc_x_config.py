# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


_CALC_X_CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "agent"
    / "agentlightning"
    / "calc_x"
    / "config"
)


def _compose_calc_x_config(config_name: str) -> DictConfig:
    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(_CALC_X_CONFIG_DIR),
    ):
        return compose(config_name=config_name)


@pytest.mark.parametrize(
    ("config_name", "expected_eval", "expected_advantage_mode", "expected_pack_traj"),
    [
        ("qwen2.5-1.5b-enginehttp-multiturn", False, "turn", False),
        ("qwen2.5-1.5b-enginehttp-trajectory", False, "trajectory", True),
        ("qwen2.5-1.5b-enginehttp-multiturn_eval", True, "turn", False),
        ("qwen2.5-1.5b-enginehttp-trajectory_eval", True, "trajectory", True),
    ],
)
def test_agentlightning_calc_x_configs_compose(
    config_name: str,
    expected_eval: bool,
    expected_advantage_mode: str,
    expected_pack_traj: bool,
):
    cfg = _compose_calc_x_config(config_name)

    assert cfg.eval is expected_eval
    assert cfg.runner.task_type == "reasoning"
    assert cfg.cluster.num_nodes == 1
    assert cfg.algorithm.advantage_mode == expected_advantage_mode
    assert cfg.actor.pack_traj is expected_pack_traj
    assert cfg.rollout.model.model_type == "qwen2.5"
    assert cfg.inference.model_type == cfg.rollout.model.model_type

    assert cfg.data.train_data_paths
    assert cfg.data.val_data_paths
    assert str(cfg.data.train_data_paths[0]).endswith(".parquet")
    assert str(cfg.data.val_data_paths[0]).endswith(".parquet")
    assert cfg.actor.tokenizer.tokenizer_model == cfg.rollout.model.model_path


@pytest.mark.parametrize(
    "config_name",
    [
        "qwen2.5-1.5b-enginehttp-multiturn",
        "qwen2.5-1.5b-enginehttp-trajectory",
    ],
)
def test_agentlightning_calc_x_configs_accept_path_overrides(config_name: str):
    model_path = "/tmp/Qwen2.5-1.5B-Instruct"
    train_path = "/tmp/calc/train.parquet"
    val_path = "/tmp/calc/test.parquet"
    with initialize_config_dir(
        version_base="1.1",
        config_dir=str(_CALC_X_CONFIG_DIR),
    ):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"rollout.model.model_path={model_path}",
                f"data.train_data_paths=[{train_path}]",
                f"data.val_data_paths=[{val_path}]",
            ],
        )

    assert cfg.rollout.model.model_path == model_path
    assert cfg.actor.tokenizer.tokenizer_model == model_path
    assert cfg.data.train_data_paths == [train_path]
    assert cfg.data.val_data_paths == [val_path]
