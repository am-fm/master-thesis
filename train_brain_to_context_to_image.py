import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_dream_and_update_latents, compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import torch.nn as nn
import re

if is_wandb_available():
    import wandb

import nibabel as nib

from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.utils.data import Dataset, DataLoader


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dream_training",
        action="store_true",
        help=(
            "Use the DREAM training method, which makes training more efficient and accurate at the ",
            "expense of doing an extra forward pass. See: https://arxiv.org/abs/2312.00210",
        ),
    )
    parser.add_argument(
        "--dream_detail_preservation",
        type=float,
        default=1.0,
        help="Dream detail preservation factor p (should be greater than 0; default=1.0, as suggested in the paper)",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--prepare_path", type=str, default=None)
    parser.add_argument("--subj_id", type=int, default=1, help="Training subject number.")
    parser.add_argument("--session", type=int, default=0, help="Training last session number.")
    parser.add_argument(
        "--b_learning_rate",
        type=float,
        default=5e-5,
        help="Initial brain-to-context transformer learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--b_scale", type=float, default=0.2)
    parser.add_argument("--l_lambda", type=float, default=0.01)
    parser.add_argument("--wandb_resume", type=str, default=None, help="wandb old run id.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def compute_contrastive_loss(pred_states, target_states, temperature=0.07):
    pred_feat = F.normalize(pred_states.mean(dim=1), dim=-1)
    target_feat = F.normalize(target_states.mean(dim=1), dim=-1)
    logits = torch.matmul(pred_feat, target_feat.T) / temperature
    batch_size = pred_feat.shape[0]
    labels = torch.arange(batch_size, device=pred_feat.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2


def main():
    args = parse_args()
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, context_length=77):
            super().__init__()
            pe = torch.zeros(context_length, d_model)
            position = torch.arange(0, context_length).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer('pe', pe)

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]

    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.1)
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            residual = x
            x = self.norm1(x)
            attn_output, _ = self.attn(x, x, x)
            x = residual + attn_output

            residual = x
            x = self.norm2(x)
            ff_output = self.ff(x)
            x = residual + ff_output
            return x
        
    class ROIMultiTokenEncoder(nn.Module):
        def __init__(self, v1_dim, v24_dim, high_dim, other_dim, hidden_dim=768):
            super().__init__()
            self.proj_v1 = nn.Linear(v1_dim, 10 * hidden_dim)
            self.proj_v24 = nn.Linear(v24_dim, 20 * hidden_dim)
            self.proj_high = nn.Linear(high_dim, 20 * hidden_dim)
            self.proj_other = nn.Linear(other_dim, 27 * hidden_dim)
            
            self.hidden_dim = hidden_dim

        def forward(self, x_v1, x_v24, x_high, x_other):
            t_v1 = self.proj_v1(x_v1).view(-1, 10, self.hidden_dim)
            t_v24 = self.proj_v24(x_v24).view(-1, 20, self.hidden_dim)
            t_high = self.proj_high(x_high).view(-1, 20, self.hidden_dim)
            t_other = self.proj_other(x_other).view(-1, 27, self.hidden_dim)
            
            tokens = torch.cat([t_v1, t_v24, t_high, t_other], dim=1)
            return tokens

    class ROITransformerEncoder(nn.Module):
        def __init__(self, input_dim, v1_dim, v24_dim, high_dim, other_dim, hidden_dim=768, n_heads=8, n_layers=6, context_length=77):
            super().__init__()
            self.input_dim = input_dim
            self.roi_encoder = ROIMultiTokenEncoder(v1_dim, v24_dim, high_dim, other_dim, hidden_dim)

            for m in self.roi_encoder.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)
            
            self.context_length = context_length
            self.hidden_dim = hidden_dim

            self.pos_encoder = PositionalEncoding(hidden_dim, context_length)
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, n_heads) for _ in range(n_layers)
            ])

        def forward(self, x, gen_mask, vis_mask):
            v1_indices = (vis_mask == 1) & (gen_mask > 0)
            v1_roi = x[:, v1_indices]
            v24_indices = (vis_mask >= 2) & (vis_mask <= 4) & (gen_mask > 0)
            v24_roi = x[:, v24_indices]
            high_indices = (vis_mask >= 5) & (gen_mask > 0)
            high_roi = x[:, high_indices]
            other_indices = (vis_mask == 0) & (gen_mask > 0)
            other_roi = x[:, other_indices]
            x = self.roi_encoder(v1_roi, v24_roi, high_roi, other_roi)
            x = self.pos_encoder(x)
            for i, layer in enumerate(self.layers):
                x = layer(x)
            return x

    class FullROITransformerEncoder(nn.Module):
        def __init__(self, input_dim=15724, hidden_dim=768, n_heads=8, n_layers=6, context_length=77):
            super().__init__()

            self.to_sequence = nn.Linear(input_dim, context_length * hidden_dim)

            nn.init.xavier_uniform_(self.to_sequence.weight)
            nn.init.constant_(self.to_sequence.bias, 0)
            
            self.context_length = context_length
            self.hidden_dim = hidden_dim

            self.pos_encoder = PositionalEncoding(hidden_dim, context_length)
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, n_heads) for _ in range(n_layers)
            ])

        def forward(self, x, gen_mask, vis_mask):
            gen_indices = (gen_mask > 0)
            x = x[:, gen_indices]
            x = self.to_sequence(x) 
            x = x.view(-1, self.context_length, self.hidden_dim)
            x = self.pos_encoder(x)
            for layer in self.layers:
                x = layer(x)
            return x

    class PartROITransformerEncoder(nn.Module):
        def __init__(self, input_dim, roi_indices, hidden_dim=768, n_heads=8, n_layers=6, context_length=77):
            super().__init__()
            self.roi_indices = roi_indices
            self.to_sequence = nn.Linear(input_dim, context_length * hidden_dim)

            nn.init.xavier_uniform_(self.to_sequence.weight)
            nn.init.constant_(self.to_sequence.bias, 0)
            
            self.context_length = context_length
            self.hidden_dim = hidden_dim

            self.pos_encoder = PositionalEncoding(hidden_dim, context_length)
            self.layers = nn.ModuleList([
                TransformerEncoderLayer(hidden_dim, n_heads) for _ in range(n_layers)
            ])

        def forward(self, x, gen_mask, vis_mask):
            x = x[:, self.roi_indices]
            x = self.to_sequence(x) 
            x = x.view(-1, self.context_length, self.hidden_dim)
            x = self.pos_encoder(x)
            for layer in self.layers:
                x = layer(x)
            return x
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    genmask_img = nib.load(f'{args.dataset_path}/natural-scenes-dataset/nsddata/ppdata/subj{args.subj_id:02d}/func1pt8mm/roi/nsdgeneral.nii.gz')
    vismask_img = nib.load(f'{args.dataset_path}/natural-scenes-dataset/nsddata/ppdata/subj{args.subj_id:02d}/func1pt8mm/roi/prf-visualrois.nii.gz')
    genmask_data = genmask_img.get_fdata()
    vismask_data = vismask_img.get_fdata()

    temp_x = torch.load(f'{args.prepare_path}/subj{args.subj_id:02d}_sess01_betas.pt')

    full_indices = (genmask_data > 0)
    full_mask = temp_x[:, full_indices]
    v1_indices = (vismask_data == 1) & (genmask_data > 0)
    v1_mask = temp_x[:, v1_indices]
    v24_indices = (vismask_data >= 2) & (vismask_data <= 4) & (genmask_data > 0)
    v24_mask = temp_x[:, v24_indices]
    high_indices = (vismask_data >= 5) & (genmask_data > 0)
    high_mask = temp_x[:, high_indices]
    other_indices = (vismask_data == 0) & (genmask_data > 0)
    other_mask = temp_x[:, other_indices]

    brain_encoder = ROITransformerEncoder(input_dim=full_mask.shape[1], v1_dim=v1_mask.shape[1], v24_dim=v24_mask.shape[1], high_dim=high_mask.shape[1], other_dim=other_mask.shape[1]).to(device)
    # brain_encoder = FullROITransformerEncoder(full_mask.shape[1]).to(device)
    # brain_encoder = PartROITransformerEncoder(v1_mask.shape[1], v1_indices).to(device)
    # brain_encoder = PartROITransformerEncoder(v24_mask.shape[1], v24_indices).to(device)
    # brain_encoder = PartROITransformerEncoder(high_mask.shape[1], high_indices).to(device)
    # brain_encoder = PartROITransformerEncoder(other_mask.shape[1], other_indices).to(device)
        
    print(brain_encoder, flush=True)

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    brain_encoder.train()
    brain_encoder.requires_grad_(True)
    for p in brain_encoder.parameters():
        p.requires_grad = True
    unet.train()
    unet.requires_grad_(False)
    def unfreeze_cross_attention_kv(new_unet):
        for module in new_unet.modules():
            if hasattr(module, 'attn2'):
                attn = module.attn2
                if hasattr(attn, 'to_k'):
                    for param in attn.to_k.parameters():
                        param.requires_grad = True
                if hasattr(attn, 'to_v'):
                    for param in attn.to_v.parameters():
                        param.requires_grad = True
    unfreeze_cross_attention_kv(unet)
    trainable = [n for n, p in unet.named_parameters() if p.requires_grad]
    print(f"Trainable UNet params: {trainable}")
    

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=ema_unet.config,
            foreach=args.foreach_ema,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DConditionModel, foreach=args.foreach_ema
                )
                ema_unet.load_state_dict(load_model.state_dict())
                if args.offload_ema:
                    ema_unet.pin_memory()
                else:
                    ema_unet.to(accelerator.device)
                del load_model

            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    
    trainable_params = list(brain_encoder.parameters())
    kv_params = []
    trainable_all = [n for n, p in brain_encoder.named_parameters() if p.requires_grad]
    for name, param in unet.named_parameters():
        if "attn2.to_k" in name or "attn2.to_v" in name:
            trainable_params.append(param)
            kv_params.append(param)
            trainable_all.append(name)
    print(f"Trainable ALL params: {trainable_all}")

    optimizer = optimizer_cls([
        {"params": brain_encoder.parameters(), "lr": args.b_learning_rate, 
        "betas": (args.adam_beta1, args.adam_beta2),
        "weight_decay": args.adam_weight_decay,
        "eps": args.adam_epsilon},
        {"params": kv_params, "lr": args.learning_rate,
        "betas": (args.adam_beta1, args.adam_beta2),
        "weight_decay": args.adam_weight_decay,
        "eps": args.adam_epsilon},
    ])

    class NSDOnDemandDataset(Dataset):
        def __init__(self, p_id, session_list, base_path):
            self.base_path = base_path
            self.p_id = p_id
            self.session_list = session_list
            self.trials_per_session = 750
            
            self.shuffle_data()
            
            self.current_session = -1
            self.cached_betas = None
            self.cached_images = None
            self.cached_captions = None

        def shuffle_data(self):
            random.shuffle(self.session_list)
            
            self.total_indices = []
            for s in self.session_list:
                trial_ids = list(range(self.trials_per_session))
                random.shuffle(trial_ids)
                for t in trial_ids:
                    self.total_indices.append((s, t))
            
            self.current_session = -1
            self.cached_betas = self.cached_images = self.cached_captions = None

        def __len__(self):
            return len(self.total_indices)

        def __getitem__(self, idx):
            sess_num, trial_idx = self.total_indices[idx]
            
            if sess_num != self.current_session:
                self.cached_betas = self.cached_images = self.cached_captions = None
                
                self.cached_betas = torch.load(os.path.join(self.base_path, f"subj{self.p_id:02d}_sess{sess_num:02d}_betas.pt"), map_location='cpu')
                self.cached_images = torch.load(os.path.join(self.base_path, f"subj{self.p_id:02d}_sess{sess_num:02d}_images.pt"), map_location='cpu')
                self.cached_captions = torch.load(os.path.join(self.base_path, f"subj{self.p_id:02d}_sess{sess_num:02d}_captions.pt"), map_location='cpu')
                self.current_session = sess_num
                
            return {
                "fmri_patches": self.cached_betas[trial_idx],
                "pixel_values": self.cached_images[trial_idx],
                "coco_captions": self.cached_captions[trial_idx]
            }

    # データセットの作成
    train_sessions = list(range(1, args.session + 1))
    train_dataset = NSDOnDemandDataset(p_id=args.subj_id, session_list=train_sessions, base_path=args.prepare_path)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=False,
        num_workers=0
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        if args.offload_ema:
            ema_unet.pin_memory()
        else:
            ema_unet.to(accelerator.device)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.wandb_resume is None:
        wandb.init(project="b2c_betas", name=f"batch_size-{args.train_batch_size} learning_rate-{args.learning_rate}")
        wandb.config.epochs = args.num_train_epochs 
    else:
        wandb.init(
            project="b2c_betas",
            id=args.wandb_resume,
            resume="must"
        )
        wandb.config.update({"epochs": args.num_train_epochs}, allow_val_change=True)


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            checkpoint_num = re.sub("\\D", "", args.resume_from_checkpoint)
            state_dict = torch.load(f"args.output_dir/brain_encoder-{checkpoint_num}.pth", map_location="cuda")
            brain_encoder.load_state_dict(state_dict)

            del state_dict
            torch.cuda.empty_cache()

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, wandb.config.epochs):
        train_loss = 0.0
        train_dataset.shuffle_data()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(device=device, dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                cap_target = batch["coco_captions"].to(device=device, dtype=weight_dtype)

                # Get the text embedding for conditioning
                encoder_hidden_states = brain_encoder(batch["fmri_patches"].to(device=device, dtype=weight_dtype), genmask_data, vismask_data)

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.dream_training:
                    noisy_latents, target = compute_dream_and_update_latents(
                        unet,
                        noise_scheduler,
                        timesteps,
                        noise,
                        noisy_latents,
                        target,
                        encoder_hidden_states,
                        args.dream_detail_preservation,
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                batch_std = torch.sqrt(encoder_hidden_states.var(dim=[1,2]))
                # 標準偏差
                loss_vtok = torch.mean(torch.relu(1.0 - batch_std)) * args.l_lambda

                # CLIP embedding
                loss_clip_mse = F.mse_loss(encoder_hidden_states.float(), cap_target.float())
                loss_clip_cos = 1.0 - F.cosine_similarity(encoder_hidden_states.float(), cap_target.float(), dim=-1).mean()
                loss_contrastive = compute_contrastive_loss(encoder_hidden_states, cap_target)
                # loss_clip_mse = 0.0
                # loss_clip_cos = 0.0
                # loss_contrastive = 0.0

                # MSE
                alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps]
                alpha_prod_t_batch = alpha_prod_t.view(-1, 1, 1, 1)
                x0_pred = (noisy_latents - (1 - alpha_prod_t_batch).sqrt() * model_pred) / alpha_prod_t_batch.sqrt()
                loss_x0_mse = F.mse_loss(x0_pred, latents)
                # コサイン類似度 / SSIM
                B = x0_pred.shape[0]
                pred = x0_pred.reshape(B, -1)
                latent = latents.reshape(B, -1)
                y = torch.ones(B).to(device)
                loss_x0_cos = F.cosine_embedding_loss(pred, latent, y)
                loss_x0_ssim = 1.0 - ssim(x0_pred, latents, data_range=x0_pred.max() - x0_pred.min())

                # ピアソン相関係数
                pred_mean = pred.mean(dim=-1, keepdim=True)
                latent_mean = latent.mean(dim=-1, keepdim=True)
                p_centered = pred - pred_mean
                l_centered = latent - latent_mean
                corr = torch.nn.functional.cosine_similarity(p_centered, l_centered, dim=-1)
                loss_x0_corr = 1 - corr.mean()

                # velocity
                target_v = noise_scheduler.get_velocity(latents, noise, timesteps)
                loss_v = F.mse_loss(model_pred.float(), target_v.float(), reduction="mean")


                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # 1. CLIP領域
                loss_clip = (loss_clip_mse * 1.0) + (loss_clip_cos * 4.0) + (loss_contrastive * 1.0)

                # 2. x0領域
                loss_x0 = (loss_x0_mse * 0.5) + (loss_x0_cos * 1.0) + (loss_x0_ssim * 2.0) + (loss_x0_corr * 1.0)

                # 3. x領域
                loss_e = loss_v * 0.2 + loss

                # 全体合算
                loss = loss_vtok + loss_x0 * 0.8 + loss_e + loss_clip
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    if args.offload_ema:
                        ema_unet.to(device="cuda", non_blocking=True)
                    ema_unet.step(unet.parameters())
                    if args.offload_ema:
                        ema_unet.to(device="cpu", non_blocking=True)
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        torch.save(brain_encoder.state_dict(), f"{args.output_dir}/brain_encoder-{global_step}.pth") # B2C保存

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            wandb.log({"loss": loss, "velocity prediction loss": loss_v,
                       "x0 MSE loss": loss_x0_mse, "x0 CosSim": loss_x0_cos, "x0 SSIM": loss_x0_ssim, "x0 CorrCoef": loss_x0_corr,
                       "token var loss": loss_vtok, "CLIP token MSE loss": loss_clip_mse, "CLIP token CosSim": loss_clip_cos, "CLIP token contrastive": loss_contrastive
                       })

            if global_step >= args.max_train_steps:
                break
    
    wandb.finish()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir) # 変更！！（追加）
        torch.save(brain_encoder.state_dict(), f"args.output_dir/brain_encoder.pth")

        # Run a final round of inference.
        images = []
        if args.validation_prompts is not None:
            logger.info("Running inference for collecting generated images...")
            pipeline = pipeline.to(accelerator.device)
            pipeline.torch_dtype = weight_dtype
            pipeline.set_progress_bar_config(disable=True)

            if args.enable_xformers_memory_efficient_attention:
                pipeline.enable_xformers_memory_efficient_attention()

            if args.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

            for i in range(len(args.validation_prompts)):
                with torch.autocast("cuda"):
                    image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]
                images.append(image)

    accelerator.end_training()


if __name__ == "__main__":
    main()
