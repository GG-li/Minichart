# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Type
import math
import json

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

from ..tooluse.parse import Parser
from ..tooluse.execution import CodeExecutor
from ..tooluse.tools import *
from PIL import Image
from ..utils.dataset import collate_fn

from ..utils import torch_functional as VF

from tensordict import TensorDict

import sys
import traceback

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    try:
        if "token_level_rewards" in data.batch.keys():
            data.batch.set_("token_level_rewards", token_level_scores - kl_ctrl.kl_coef * kld)
        else:
            data.batch = data.batch.clone()
            data.batch.set("token_level_rewards", token_level_scores - kl_ctrl.kl_coef * kld)
    except RuntimeError:
        data.batch = data.batch.clone()
        data.batch.set("token_level_rewards", token_level_scores - kl_ctrl.kl_coef * kld)

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

def custom_unraisablehook(unraisable):
    print("=== Unraisable Exception Caught ===")
    print(f"Type: {unraisable.exc_type.__name__}")
    print(f"Value: {unraisable.exc_value}")
    if unraisable.object:
        print(f"In object: {unraisable.object}")
    if unraisable.traceback:
        print("Traceback:")
        traceback.print_tb(unraisable.traceback)
    print("===================================")

class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.tool_parser = Parser()

        sys.unraisablehook = custom_unraisablehook

        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        self._trace_enable = bool(getattr(config.trainer, "trace", None) and config.trainer.trace.enable)
        self._trace_dir = None
        self._trace_sample_size = 0
        self._trace_save_images = False
        self._trace_max_steps = 0
        if getattr(config.trainer, "trace", None) is not None:
            self._trace_dir = config.trainer.trace.output_dir
            self._trace_sample_size = int(config.trainer.trace.sample_size_per_step)
            self._trace_save_images = bool(config.trainer.trace.save_images)
            self._trace_max_steps = int(config.trainer.trace.max_steps)
        if self._trace_enable and self._trace_dir:
            os.makedirs(self._trace_dir, exist_ok=True)

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _maybe_trace_step(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_metrics: Dict[str, List[float]],
    ) -> None:
        if not self._trace_enable or not self._trace_dir:
            return
        if self._trace_max_steps > 0 and self.global_step > self._trace_max_steps:
            return
        if self._trace_sample_size <= 0:
            return

        step_dir = os.path.join(self._trace_dir, f"step_{self.global_step:06d}")
        os.makedirs(step_dir, exist_ok=True)

        response_ids = batch.batch.get("responses", None)
        response_mask = batch.batch.get("response_mask", None)
        reward_scalar = reward_tensor.sum(dim=-1).detach().float().cpu().tolist()

        prompts = batch.non_tensor_batch.get("prompt", None)
        queries = batch.non_tensor_batch.get("query", None)
        gts = batch.non_tensor_batch.get("ground_truth", None)
        penalties = batch.non_tensor_batch.get("penalty", None)
        uids = batch.non_tensor_batch.get("uid", None)
        figure_ids = batch.non_tensor_batch.get("figure_id", None)
        figure_paths = batch.non_tensor_batch.get("figure_path", None)
        image_1_pil = batch.non_tensor_batch.get("image_1_pil", None)
        multi_modal_data = batch.non_tensor_batch.get("multi_modal_data", None)
        rollout_rounds = batch.non_tensor_batch.get("rollout_round", None)
        tool_codes = batch.non_tensor_batch.get("tool_code", None)
        tool_parse_status = batch.non_tensor_batch.get("tool_parse_status", None)
        tool_error_code = batch.non_tensor_batch.get("tool_error_code", None)
        tool_exec_success = batch.non_tensor_batch.get("tool_exec_success", None)
        tool_second_prompt = batch.non_tensor_batch.get("tool_second_prompt", None)
        tool_edited_image_path = batch.non_tensor_batch.get("tool_edited_image_path", None)
        tool_original_image_path = batch.non_tensor_batch.get("tool_original_image_path", None)

        indices: List[int] = []
        seen = set()
        for i in range(len(batch)):
            uid = str(uids[i]) if uids is not None else str(i)
            if uid in seen:
                continue
            seen.add(uid)
            indices.append(i)
            if len(indices) >= self._trace_sample_size:
                break

        trace_path = os.path.join(step_dir, "trace.jsonl")
        with open(trace_path, "w", encoding="utf-8") as f:
            for i in indices:
                resp_text = None
                if response_ids is not None and response_mask is not None:
                    resp_len = int(response_mask[i].sum().item())
                    resp_text = self.tokenizer.decode(response_ids[i][:resp_len], skip_special_tokens=True)

                record: Dict[str, Any] = {
                    "global_step": int(self.global_step),
                    "index": int(i),
                    "uid": str(uids[i]) if uids is not None else None,
                    "rollout_round": int(rollout_rounds[i]) if rollout_rounds is not None else 0,
                    "figure_id": str(figure_ids[i]) if figure_ids is not None else None,
                    "figure_path": str(figure_paths[i]) if figure_paths is not None else None,
                    "query": str(queries[i]) if queries is not None else None,
                    "ground_truth": str(gts[i]) if gts is not None else None,
                    "prompt": str(prompts[i]) if prompts is not None else None,
                    "response": resp_text,
                    "penalty": float(penalties[i]) if penalties is not None else 0.0,
                    "reward_overall": float(reward_scalar[i]),
                }
                for k, v in reward_metrics.items():
                    if i < len(v):
                        record[f"reward_{k}"] = float(v[i])
                record["tool_parse_status"] = int(tool_parse_status[i]) if tool_parse_status is not None else None
                record["tool_error_code"] = str(tool_error_code[i]) if tool_error_code is not None else None
                record["tool_code"] = str(tool_codes[i]) if tool_codes is not None else None
                record["tool_exec_success"] = int(tool_exec_success[i]) if tool_exec_success is not None else None
                record["tool_second_prompt"] = str(tool_second_prompt[i]) if tool_second_prompt is not None else None
                record["tool_edited_image_path"] = (
                    str(tool_edited_image_path[i]) if tool_edited_image_path is not None else None
                )
                record["tool_original_image_path"] = (
                    str(tool_original_image_path[i]) if tool_original_image_path is not None else None
                )

                if self._trace_save_images:
                    saved_original = None
                    if tool_original_image_path is not None and i < len(tool_original_image_path) and tool_original_image_path[i]:
                        record["saved_original_image"] = str(tool_original_image_path[i])
                        saved_original = str(tool_original_image_path[i])
                    if figure_paths is not None and figure_paths[i]:
                        try:
                            img = Image.open(str(figure_paths[i]))
                            saved_original = os.path.join(step_dir, f"img_{i}_orig.png")
                            img.save(saved_original)
                        except Exception:
                            saved_original = None
                    if saved_original is None and image_1_pil is not None and i < len(image_1_pil):
                        try:
                            if isinstance(image_1_pil[i], Image.Image):
                                saved_original = os.path.join(step_dir, f"img_{i}_orig.png")
                                image_1_pil[i].save(saved_original)
                        except Exception:
                            saved_original = None
                    if saved_original is None and multi_modal_data is not None and i < len(multi_modal_data):
                        try:
                            mm = multi_modal_data[i]
                            if isinstance(mm, dict):
                                imgs = mm.get("image", None)
                                if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                                    saved_original = os.path.join(step_dir, f"img_{i}_orig.png")
                                    imgs[0].save(saved_original)
                        except Exception:
                            saved_original = None
                    if saved_original is not None:
                        record["saved_original_image"] = saved_original

                    if tool_edited_image_path is not None and tool_edited_image_path[i]:
                        record["saved_edited_image"] = str(tool_edited_image_path[i])

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def display(self, obj):
        self.captured_output = obj

    def quick_save(self, file_name, text):
        with open(file_name, 'w') as file:
            file.write(text)

    def get_tool_context(self):
        context = {
            "display": self.display,
            "focus_on_columns_with_mask": focus_on_columns_with_mask,
            "focus_on_rows_with_mask": focus_on_rows_with_mask,
            "focus_on_columns_with_draw": focus_on_columns_with_draw,
            "focus_on_rows_with_draw": focus_on_rows_with_draw,
            "focus_on_columns_with_highlight": focus_on_columns_with_highlight,
            "focus_on_rows_with_highlight": focus_on_rows_with_highlight,
            "focus_on_x_values_with_mask": focus_on_x_values_with_mask,
            "focus_on_y_values_with_mask": focus_on_y_values_with_mask,
            "focus_on_x_values_with_draw": focus_on_x_values_with_draw,
            "focus_on_y_values_with_draw": focus_on_y_values_with_draw,
            "focus_on_x_values_with_highlight": focus_on_x_values_with_highlight,
            "focus_on_y_values_with_highlight": focus_on_y_values_with_highlight,
        }
        return context
    
    def is_image_closed(self, img):
        try:
            img.convert("RGB")
            return False
        except Exception as e:
            if "closed image" in str(e):
                print("Image is closed.")
            return True

    def lcm(self, a, b):
        return abs(a * b) // math.gcd(a, b)
    
    def get_second_rollout_batch(self, og_output_gen_batch, original_full_batch, append=False, save_images=False):
        #print(test_batch.non_tensor_batch.keys())

        figure_paths = original_full_batch.non_tensor_batch["figure_path"]
        prompts = original_full_batch.non_tensor_batch["prompt"]
        figure_ids = original_full_batch.non_tensor_batch["figure_id"]
        metadata_batch = original_full_batch.non_tensor_batch["metadata"]

        # Store generated outputs
        output_ids = og_output_gen_batch.batch["responses"]
        output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

        #only implemented for h_chart

        parsed_results = [self.tool_parser.parse(output_text) for output_text in output_texts]

        trace_enable = self._trace_enable and bool(self._trace_dir) and (self._trace_max_steps <= 0 or self.global_step <= self._trace_max_steps)
        trace_step_dir = None
        if trace_enable:
            trace_step_dir = os.path.join(self._trace_dir, f"step_{self.global_step:06d}")
            os.makedirs(trace_step_dir, exist_ok=True)

            n = len(parsed_results)
            def ensure_key(key: str, default):
                if key not in original_full_batch.non_tensor_batch:
                    original_full_batch.non_tensor_batch[key] = np.array([default] * n, dtype=object)

            ensure_key("tool_parse_status", 0)
            ensure_key("tool_error_code", "")
            ensure_key("tool_code", "")
            ensure_key("tool_exec_success", 0)
            ensure_key("tool_second_prompt", "")
            ensure_key("tool_edited_image_path", "")
            ensure_key("tool_original_image_path", "")
            if "image_1_pil" not in original_full_batch.non_tensor_batch:
                original_full_batch.non_tensor_batch["image_1_pil"] = np.array([None] * n, dtype=object)

        # If parsed result returns a failure then we have some intermediate nonreward??

        edited_images = []

        tool_use_indices = []

        second_rollout_datas = []

        num_tool_calls = 0
        num_direct = 0
        num_success_tool_calls = 0
        num_failed_tool_calls = 0

        tool_functions = [
            "focus_on_columns_with_mask",
            "focus_on_rows_with_mask",
            "focus_on_columns_with_draw",
            "focus_on_rows_with_draw",
            "focus_on_columns_with_highlight",
            "focus_on_rows_with_highlight",
            "focus_on_x_values_with_mask",
            "focus_on_y_values_with_mask",
            "focus_on_x_values_with_draw",
            "focus_on_y_values_with_draw",
            "focus_on_x_values_with_highlight",
            "focus_on_y_values_with_highlight"
        ]

        # this actually takes some time
        for idx, result in enumerate(parsed_results):
            if not result["status"]:
                if result["error_code"] == "NOTOOL":
                    num_direct += 1
                    if trace_enable:
                        original_full_batch.non_tensor_batch["tool_parse_status"][idx] = 0
                        original_full_batch.non_tensor_batch["tool_error_code"][idx] = "NOTOOL"
                else:
                    original_full_batch.non_tensor_batch["penalty"][idx] = -10
                    num_tool_calls += 1
                    num_failed_tool_calls += 1
                    if trace_enable:
                        original_full_batch.non_tensor_batch["tool_parse_status"][idx] = 0
                        original_full_batch.non_tensor_batch["tool_error_code"][idx] = str(result.get("error_code", ""))

                continue

            metadata = json.loads(metadata_batch[idx])
            
            '''y_values = metadata["y_values"]
            y_bboxes = metadata["y_bboxes"]

            headers = y_values  # these are your column names
            bbox_mapping = {label: bbox for label, bbox in zip(y_values, y_bboxes)}'''

            if metadata["type"] == "v_bar":
                bbox_mapping = metadata["x_values_bbox"]
            elif metadata["type"] == "h_bar":
                bbox_mapping = metadata["y_values_bbox"]

            #code_executor = CodeExecutor("executor")
            code = result["content"]
            #exit_code, output, file_paths = code_executor.execute(result["content"])
            figure_path = figure_paths[idx]

            mentions_any_tool = any((f"{name}(" in code or f"{name} (" in code) for name in tool_functions)
            if not mentions_any_tool:
                num_direct += 1
                original_full_batch.non_tensor_batch["penalty"][idx] = 0
                if trace_enable:
                    original_full_batch.non_tensor_batch["tool_parse_status"][idx] = 1
                    original_full_batch.non_tensor_batch["tool_error_code"][idx] = "NO_TOOL_CALL"
                    original_full_batch.non_tensor_batch["tool_code"][idx] = code
                continue

            num_tool_calls += 1

            #print(code)
            successful = True
            exec_error = None

            self.captured_output = None

            context = self.get_tool_context()
            if trace_enable:
                original_full_batch.non_tensor_batch["tool_parse_status"][idx] = 1
                original_full_batch.non_tensor_batch["tool_error_code"][idx] = ""
                original_full_batch.non_tensor_batch["tool_code"][idx] = code

            base_tools = dict(context)
            allowed_tool_set = set(tool_functions)
            tool_state = {"called_any": False, "called_allowed": False}

            if metadata["type"] == "table":
                context["columns_bbox"] = metadata["columns_bbox"]
                context["rows_bbox"] = metadata["row_starters"]
            else:
                if metadata["type"] == "v_bar":
                    context["columns_bbox"] = bbox_mapping
                    context["rows_bbox"] = {}
                elif metadata["type"] == "h_bar":
                    context["columns_bbox"] = {}
                    context["rows_bbox"] = bbox_mapping
                else:
                    context["columns_bbox"] = bbox_mapping
                    context["rows_bbox"] = bbox_mapping

            original_image = None
            base_image = None
            image_1_pil = original_full_batch.non_tensor_batch.get("image_1_pil", None)
            if image_1_pil is not None and idx < len(image_1_pil) and isinstance(image_1_pil[idx], Image.Image):
                base_image = image_1_pil[idx]
            multi_modal_data = original_full_batch.non_tensor_batch.get("multi_modal_data", None)
            if multi_modal_data is not None and idx < len(multi_modal_data):
                mm = multi_modal_data[idx]
                if isinstance(mm, dict):
                    imgs = mm.get("image", None)
                    if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                        base_image = imgs[0]

            if base_image is None:
                try:
                    fp = str(figure_path)
                    candidates = []
                    if os.path.isabs(fp):
                        candidates.append(fp)
                    else:
                        candidates.append(fp)
                        candidates.append(os.path.join(os.getcwd(), fp))
                        candidates.append(os.path.join(self._project_root, fp))

                    found = None
                    for c in candidates:
                        if os.path.exists(c):
                            found = c
                            break
                    if found is None:
                        raise FileNotFoundError(fp)
                    base_image = Image.open(found)
                except Exception:
                    successful = False
                    exec_error = "MISSING_IMAGE"

            if base_image is not None:
                original_image = base_image.copy()
                if trace_enable:
                    original_full_batch.non_tensor_batch["image_1_pil"][idx] = base_image
                    if self._trace_save_images and trace_step_dir is not None:
                        try:
                            orig_dst = os.path.join(trace_step_dir, f"img_{idx}_orig.png")
                            original_image.save(orig_dst)
                            original_full_batch.non_tensor_batch["tool_original_image_path"][idx] = orig_dst
                        except Exception:
                            pass

            if successful:
                context["image_1"] = original_image.copy()
                context["image"] = context["image_1"]
                context["x_values_bbox"] = context["columns_bbox"]
                context["y_values_bbox"] = context["rows_bbox"]
                context["all_x_values_bounding_boxes"] = context["x_values_bbox"]
                context["all_y_values_bounding_boxes"] = context["y_values_bbox"]
                context["figure_path"] = figure_path
                context["image_1_path"] = figure_path

                def make_wrapper(name, fn):
                    def _inner(*args, **kwargs):
                        tool_state["called_any"] = True
                        if name in allowed_tool_set:
                            tool_state["called_allowed"] = True
                        out = fn(*args, **kwargs)
                        if isinstance(out, Image.Image):
                            self.display(out)
                        return out

                    return _inner

                for tool_name, tool_fn in list(base_tools.items()):
                    if tool_name == "display":
                        continue
                    context[tool_name] = make_wrapper(tool_name, tool_fn)

            try:
                if successful:
                    exec(code, context)
            except BaseException as e:
                successful = False
                exec_error = e

            if successful and not tool_state["called_any"]:
                all_tool_names = [k for k in base_tools.keys() if k != "display"]
                mentions_any_tool_fallback = any((f"{name}(" in code or f"{name} (" in code) for name in all_tool_names)
                if mentions_any_tool_fallback:
                    successful = False
                    exec_error = "NO_TOOL_CALL"
                else:
                    num_direct += 1
                    original_full_batch.non_tensor_batch["penalty"][idx] = 0
                    continue

            if successful:
                if self.captured_output is not None:
                    successful = isinstance(self.captured_output, Image.Image) and not self.is_image_closed(self.captured_output)
                else:
                    auto_img = None
                    for k, v in context.items():
                        if k in {"image_1", "image"}:
                            continue
                        if isinstance(v, Image.Image) and not self.is_image_closed(v):
                            auto_img = v
                            break
                    if auto_img is not None:
                        self.captured_output = auto_img
                        successful = True
                    else:
                        successful = False

            if successful:
                num_success_tool_calls += 1
                if trace_enable:
                    original_full_batch.non_tensor_batch["tool_exec_success"][idx] = 1

                edited_images.append(self.captured_output)

                trim_to_action_end = self.tool_parser.trim_to_action_end(output_texts[idx])

                #we need to add image repsonse here:
                trim_to_action_end += "\nOBSERVATION: Execution success. The output is as follows:"
                trim_to_action_end += "\n<the image outputs of the code is added as the second image>"

                #image isn't actually used here so we insert two dummy images
                messages = self.val_dataloader.dataset.tu_build_message(prompts[idx], [None, None], trim_to_action_end)
                
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                if trace_enable:
                    original_full_batch.non_tensor_batch["tool_second_prompt"][idx] = prompt

                edited_image = self.captured_output
                if trace_enable and self._trace_save_images and trace_step_dir is not None:
                    try:
                        edited_dst = os.path.join(trace_step_dir, f"img_{idx}_edited.png")
                        edited_image.save(edited_dst)
                        original_full_batch.non_tensor_batch["tool_edited_image_path"][idx] = edited_dst
                    except Exception:
                        pass

                images = [self.val_dataloader.dataset.process_image(image) for image in [original_image, edited_image]]

                model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

                #we assume this is not for QWEN2, see dataset.py for code
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
                
                second_rollout_data = {}

                second_rollout_data["multi_modal_data"] = {"image": images}
                second_rollout_data["multi_modal_inputs"] = dict(model_inputs)

                input_ids, attention_mask, position_ids = VF.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_length=self.val_dataloader.dataset.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation=self.val_dataloader.dataset.truncation,
                )

                max_prompt_length = self.val_dataloader.dataset.max_prompt_length

                raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(raw_prompt_ids) > max_prompt_length:
                    if self.truncation == "left":
                        raw_prompt_ids = raw_prompt_ids[-max_prompt_length :]
                    elif self.truncation == "right":
                        raw_prompt_ids = raw_prompt_ids[: max_prompt_length]
                    elif self.truncation == "error":
                        raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")
                    
                second_rollout_data["input_ids"] = input_ids
                second_rollout_data["attention_mask"] = attention_mask
                second_rollout_data["position_ids"] = position_ids
                second_rollout_data["raw_prompt_ids"] = raw_prompt_ids
                second_rollout_data["metadata"] = metadata_batch[idx]
                second_rollout_data["rollout_round"] = 1

                # we add all the keys that aren't yet assigned
                for key in original_full_batch.non_tensor_batch.keys():
                    if key not in second_rollout_data:
                        second_rollout_data[key] = original_full_batch.non_tensor_batch[key][idx]

                second_rollout_datas.append(second_rollout_data)

                tool_use_indices.append(idx)

                #original_full_batch.non_tensor_batch["penalty"][idx] += 1

                use_proper_function = any(func in code for func in tool_functions)
                if use_proper_function:
                    original_full_batch.non_tensor_batch["penalty"][idx] = 1
                else:
                    original_full_batch.non_tensor_batch["penalty"][idx] = -1

            else:
                original_full_batch.non_tensor_batch["penalty"][idx] = -10
                #original_full_batch.non_tensor_batch["penalty"][idx] -= 0.05
                num_failed_tool_calls += 1
                edited_images.append(None)
                if trace_enable:
                    original_full_batch.non_tensor_batch["tool_exec_success"][idx] = 0
                    if exec_error is not None:
                        original_full_batch.non_tensor_batch["tool_error_code"][idx] = (
                            type(exec_error).__name__ if isinstance(exec_error, BaseException) else str(exec_error)
                        )
                    else:
                        original_full_batch.non_tensor_batch["tool_error_code"][idx] = "NO_OUTPUT"
                    if self._trace_save_images and trace_step_dir is not None and isinstance(original_image, Image.Image):
                        try:
                            orig_dst = os.path.join(trace_step_dir, f"img_{idx}_orig.png")
                            original_image.save(orig_dst)
                            original_full_batch.non_tensor_batch["tool_original_image_path"][idx] = orig_dst
                        except Exception:
                            pass

            #print(f"SUCCESSFUL {successful}")
            #print("------")

        #if the model makes 0 calls to rollout
        if len(second_rollout_datas) == 0:
            stats = {
                "num_tool_calls" : num_tool_calls,
                "num_direct" : num_direct,
                "num_success_tool_calls" : num_success_tool_calls,
                "num_failed_tool_calls" : num_failed_tool_calls
            }
            return original_full_batch, stats
        

        if append:
            # Repeat last for n times
            world_size = self.actor_rollout_wg.world_size
            micro_batch_size = self.config.worker.actor.micro_batch_size_per_device_for_experience
            global_batch_size = self.config.worker.actor.global_batch_size
            rollout_n = self.config.worker.rollout.n

            # Calculate the LCM of all three
            lcm1 = self.lcm(world_size, micro_batch_size)
            lcm2 = self.lcm(lcm1, global_batch_size)
            full_lcm = self.lcm(lcm2, rollout_n)

            current_len = len(second_rollout_datas) + len(figure_paths)
            append_padding_size = (full_lcm - current_len % full_lcm) % full_lcm

            second_rollout_datas.extend([second_rollout_datas[-1]] * append_padding_size)

        second_rollout_batch_dict = collate_fn(second_rollout_datas)

        ##SECOND ROLLOUT

        second_rollout_batch = DataProto.from_single_dict(second_rollout_batch_dict)

        #print(test_batch.batch["figure_id"])
        # Store original inputs
        input_ids = second_rollout_batch.batch["input_ids"]

        #print("starting second rollout")

        if "multi_modal_data" in second_rollout_batch.non_tensor_batch.keys():
            second_test_gen_batch = second_rollout_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
        else:
            second_test_gen_batch = second_rollout_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )

        second_test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
        second_test_gen_batch, pad_size = pad_dataproto_to_divisor(second_test_gen_batch, self.actor_rollout_wg.world_size)

        #second output gen batch corresponds to test_output_gen_batch
        second_output_gen_batch = self.actor_rollout_wg.generate_sequences(second_test_gen_batch)
        second_output_gen_batch = unpad_dataproto(second_output_gen_batch, pad_size=pad_size)

        #this corrsponds to the test_batch in original
        second_rollout_batch = second_rollout_batch.union(second_output_gen_batch)

        # Store generated outputs
        second_output_ids = second_output_gen_batch.batch["responses"]
        #second_output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in second_output_ids]

        '''second_output_eval = ""
        for indice, og_id in enumerate(tool_use_indices):'''
            
        #print(second_output_texts)
        #self.quick_save("second_rollout.txt", "\n\n--------\n\n".join(second_output_texts))

        #for combining the two datasets
        #we should probably
        #keep the original prompt
        #concatenate instead of replacing: attention mask, input ids
        #since to the model they are the same things.
        #MASKING should also be implemented here

        if not append:
            #if we do not append this for double reward purposes, we replace it
            keys_to_remain = {"prompts", "input_ids", "position_ids", "attention_mask"}
            keys_to_merge = {"responses", "response_mask"}

            keys_to_remain = {""}
            keys_to_merge = {""}

            for key in original_full_batch.batch.keys():
                if key not in second_rollout_batch.batch:
                    continue

                if key in keys_to_remain:
                    continue

                if key in keys_to_merge:
                    for index, og_index in enumerate(tool_use_indices):
                        original_full_batch.batch[key][og_index] = torch.cat(
                            (original_full_batch.batch[key][og_index], second_rollout_batch.batch[key][index])
                        )
                    continue

                for index, og_index in enumerate(tool_use_indices):
                    original_full_batch.batch[key][og_index] = second_rollout_batch.batch[key][index]

            for key in original_full_batch.non_tensor_batch.keys():
                if key not in second_rollout_batch.non_tensor_batch:
                    continue

                for index, og_index in enumerate(tool_use_indices):
                    original_full_batch.non_tensor_batch[key][og_index] = second_rollout_batch.non_tensor_batch[key][index]
        else:
            first_key, first_value = next(iter(second_rollout_batch.batch.items()))

            #append_padding_size = (self.actor_rollout_wg.world_size - len(second_rollout_batch.batch[first_key]) % self.actor_rollout_wg.world_size) % self.actor_rollout_wg.world_size

            final_batch_size = current_len + append_padding_size
            
            merged_data = {}
            for key in second_rollout_batch.batch.keys():
                if key not in original_full_batch.batch:
                    continue

                first = original_full_batch.batch[key]
                second = second_rollout_batch.batch[key]

                #apply masking
                '''if key == "attention_mask":
                    if append_padding_size > 0:
                        second[-append_padding_size:] *= 0'''
                    #padding *= 0'''

                merged_data[key] = torch.cat((first, second), dim=0)

            new_batch = TensorDict(merged_data, batch_size=[final_batch_size])

            new_non_tensor = {}
            for key in second_rollout_batch.non_tensor_batch.keys():
                if key not in original_full_batch.non_tensor_batch:
                    continue

                v1 = original_full_batch.non_tensor_batch[key]
                v2 = second_rollout_batch.non_tensor_batch[key]

                if isinstance(v1, list) and isinstance(v2, list):
                    new_non_tensor[key] = v1 + v2
                elif isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                    new_non_tensor[key] = np.concatenate([v1, v2], axis=0)
                else:
                    raise Exception("invalid types")

            original_full_batch = DataProto(batch=new_batch, non_tensor_batch=new_non_tensor)


        stats = {
            "num_tool_calls" : num_tool_calls,
            "num_direct" : num_direct,
            "num_success_tool_calls" : num_success_tool_calls,
            "num_failed_tool_calls" : num_failed_tool_calls
        }

        return original_full_batch, stats

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)

        tool_stats = {
            "num_tool_calls" : 0,
            "num_direct" : 0,
            "num_success_tool_calls" : 0,
            "num_failed_tool_calls" : 0
        }

        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)

            #print(test_batch.batch["figure_id"])
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                mm = test_batch.non_tensor_batch.get("multi_modal_data", None)
                if mm is not None:
                    image_1_pil = []
                    for item in mm:
                        img0 = None
                        if isinstance(item, dict):
                            imgs = item.get("image", None)
                            if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                                img0 = imgs[0]
                        image_1_pil.append(img0)
                    test_batch.non_tensor_batch["image_1_pil"] = np.array(image_1_pil, dtype=object)
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)

            #this is generated output / search r1
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            
            #call second rollout here

            ##END SECOND ROLLOUT

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            test_batch, batch_tool_stats = self.get_second_rollout_batch(test_output_gen_batch, test_batch, save_images=True)
            for key in batch_tool_stats:
                tool_stats[key] += batch_tool_stats[key]

            #We merge TEST_BATCH (1st rollout) and SECOND_ROLLOUT_BATCH here
            #for index in tool_use_indices:
            #NAIVE IMPLEMENTATION
            #Hope this is correct: https://verl.readthedocs.io/en/latest/data.html

            #validation_index = tool_use_indices[0]
            #print(test_batch.non_tensor_batch["ground_truth"][validation_index])

            #to validate??

            #print(test_batch.non_tensor_batch["ground_truth"][validation_index])

            #----END OF MERGING
            #We have successfully replaced second rollouts w/ the first rollouts.

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        tool_call_ratio = tool_stats["num_tool_calls"] / (tool_stats["num_direct"] + tool_stats["num_tool_calls"] + 0.001)
        tool_call_success_rate = tool_stats["num_success_tool_calls"] / (tool_stats["num_tool_calls"] + 0.001)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        return {"val/reward_score": reward_score, "val/tool_call_ratio" : tool_call_ratio, "val/tool_call_success_rate": tool_call_success_rate, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        try:
            # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
            remove_obsolete_ckpt(
                self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
            )
            folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
            actor_path = os.path.join(folder_path, "actor")
            self.actor_rollout_wg.save_checkpoint(actor_path)

            if self.use_critic:
                critic_path = os.path.join(folder_path, "critic")
                self.critic_wg.save_checkpoint(critic_path)

            dataloader_path = os.path.join(folder_path, "dataloader.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_path)

            last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
            with open(last_global_step_path, "w") as f:
                f.write(str(self.global_step))
        except Exception as e:
            print("ERROR")
            print(e)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size

        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _make_batch_data_online_filtering(self, metrics: Dict[str, Any]) -> DataProto:
        batch = None
        num_try_make_batch = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            new_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch))], dtype=object)

            if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                mm = new_batch.non_tensor_batch.get("multi_modal_data", None)
                if mm is not None and "image_1_pil" not in new_batch.non_tensor_batch:
                    image_1_pil = []
                    for item in mm:
                        img0 = None
                        if isinstance(item, dict):
                            imgs = item.get("image", None)
                            if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                                img0 = imgs[0]
                        image_1_pil.append(img0)
                    new_batch.non_tensor_batch["image_1_pil"] = np.array(image_1_pil, dtype=object)
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                gen_batch = new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch = new_batch.batch.clone()
                new_batch.batch.set("reward_baselines", reward_baseline_tensor)
                del gen_baseline_batch, gen_baseline_output

            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)

            if "metadata" in new_batch.non_tensor_batch:
                new_batch, _ = self.get_second_rollout_batch(
                    gen_batch_output, new_batch, append=self.config.worker.reward.double_reward
                )
                new_batch.non_tensor_batch.pop("multi_modal_data", None)

            reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(new_batch))
            new_batch.batch = new_batch.batch.clone()
            new_batch.batch.set("token_level_scores", reward_tensor)

            filter_scores = reward_metrics.get(self.config.algorithm.filter_key, None)
            if filter_scores is None:
                raise KeyError(f"Missing filter_key={self.config.algorithm.filter_key} in reward_metrics.")

            uids = new_batch.non_tensor_batch["uid"]
            uid2scores = defaultdict(list)
            for uid, score in zip(uids, filter_scores):
                uid2scores[uid].append(score)

            uid2mean = {uid: float(np.mean(scores)) for uid, scores in uid2scores.items()}
            kept_uids = [
                uid
                for uid, avg_score in uid2mean.items()
                if avg_score > self.config.algorithm.filter_low and avg_score < self.config.algorithm.filter_high
            ]
            kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
            if len(kept_sample_idxs) == 0:
                raise RuntimeError("No sample is kept after filtering. Please check your data.")

            new_batch.reorder(torch.tensor(kept_sample_idxs, dtype=torch.long))

            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = getattr(self.config.trainer, "max_try_make_batch", 20)
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data.")
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        if self.config.algorithm.online_filtering:
            self.data_iterator = iter(self.train_dataloader)
            while self.global_step < self.training_steps:
                self.global_step += 1

                metrics, timing_raw = {}, {}
                with timer("step", timing_raw):
                    with timer("gen", timing_raw):
                        batch = self._make_batch_data_online_filtering(metrics)

                    with timer("reward", timing_raw):
                        reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(batch))

                    self._maybe_trace_step(batch, reward_tensor, reward_metrics)
                    batch.batch = batch.batch.clone()
                    batch.batch.set("token_level_scores", reward_tensor)

                    self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with timer("adv", timing_raw):
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch = batch.batch.clone()
                            batch.batch.set("token_level_rewards", batch.batch["token_level_scores"])

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    if self.config.trainer.critic_warmup <= self.global_step:
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
                self.logger.log(data=metrics, step=self.global_step)
        else:
            for _ in tqdm(range(self.config.trainer.total_epochs), desc="Epoch", position=0):
                for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                    self.global_step += 1
                    if self.global_step > self.training_steps:
                        break

                    metrics, timing_raw = {}, {}
                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                    if "multi_modal_data" in batch.non_tensor_batch.keys():
                        mm = batch.non_tensor_batch.get("multi_modal_data", None)
                        if mm is not None and "image_1_pil" not in batch.non_tensor_batch:
                            image_1_pil = []
                            for item in mm:
                                img0 = None
                                if isinstance(item, dict):
                                    imgs = item.get("image", None)
                                    if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                                        img0 = imgs[0]
                                image_1_pil.append(img0)
                            batch.non_tensor_batch["image_1_pil"] = np.array(image_1_pil, dtype=object)
                        gen_batch = batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                        )
                    else:
                        gen_batch = batch.pop(
                            batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids"],
                        )

                    with timer("step", timing_raw):
                        with timer("gen", timing_raw):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        if self.config.algorithm.adv_estimator == "remax":
                            with timer("gen_max", timing_raw):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["temperature"] = 0
                                gen_baseline_batch.meta_info["n"] = 1
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                batch = batch.union(gen_baseline_output)

                                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                                batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                                batch.batch["reward_baselines"] = reward_baseline_tensor
                                del gen_baseline_batch, gen_baseline_output

                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                        )
                        batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)

                        batch = batch.union(gen_batch_output)
                        if "metadata" in batch.non_tensor_batch:
                            batch, _ = self.get_second_rollout_batch(
                                gen_batch_output, batch, append=self.config.worker.reward.double_reward
                            )

                        batch.non_tensor_batch.pop("multi_modal_data", None)

                        with timer("reward", timing_raw):
                            reward_ref = self.reward_fn.compute_reward.remote(batch)

                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        self._maybe_trace_step(batch, reward_tensor, reward_metrics)
                        try:
                            if "token_level_scores" in batch.batch.keys():
                                batch.batch.set_("token_level_scores", reward_tensor)
                            else:
                                batch.batch = batch.batch.clone()
                                batch.batch.set("token_level_scores", reward_tensor)
                        except RuntimeError:
                            batch.batch = batch.batch.clone()
                            batch.batch.set("token_level_scores", reward_tensor)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with timer("adv", timing_raw):
                        # get token level scores
                        
                        #These two lines are moved before the "balance_batch" command since otherwise we would not be able to retrieve the rewards.
                        #reward_tensor, reward_metrics = ray.get(reward_ref)
                        #batch.batch["token_level_scores"] = reward_tensor


                        #print("THIS IS THE REWARDS")
                        #print(batch.batch["token_level_scores"])
                        #print(type(batch.batch["token_level_scores"]))
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            try:
                                if "token_level_rewards" in batch.batch.keys():
                                    batch.batch.set_("token_level_rewards", batch.batch["token_level_scores"])
                                else:
                                    batch.batch = batch.batch.clone()
                                    batch.batch.set("token_level_rewards", batch.batch["token_level_scores"])
                            except RuntimeError:
                                batch.batch = batch.batch.clone()
                                batch.batch.set("token_level_rewards", batch.batch["token_level_scores"])

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # update critic
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with timer("save_checkpoint", timing_raw):
                            print("saving checkpoint")
                            self._save_checkpoint()
                            print("done")

                    # collect metrics
                    num_gpus = self.resource_pool_manager.get_num_gpus()
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                    self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
