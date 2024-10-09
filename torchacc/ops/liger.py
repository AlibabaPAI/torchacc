import importlib.util

import torch

import torchacc.utils.logger as logger

IS_LIGER_KERNEL_AVAILABLE = importlib.util.find_spec("liger_kernel") is not None


def rms_forward(self, hidden_states, offset=0.0, casting_mode="llama"):
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    return LigerRMSNormFunction.apply(
        hidden_states,
        self.weight,
        self.variance_epsilon,
        offset,
        casting_mode,
    )


def mlp_forward(self, x):
    if not isinstance(self.act_fn, torch.nn.SiLU):
        # if self.config.hidden_act not in ["silu", "swish"]:
        raise ValueError(f"Activation function {self.act_fn} not supported.")

    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    return self.down_proj(
        LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))


# Below is adapted from liger_kernel.transformers.apply_liger_kernel_to_llama
def apply_liger_kernel_to_llama(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Llama models (2 and 3)

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but
            more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """
    if not IS_LIGER_KERNEL_AVAILABLE:
        raise ImportError(
            "Liger kernel is not available. Please install it by running `pip install liger-kernel`."
        )
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    # Note: We cannot import liger_kernel in advance because liger_kernel will import transformers,
    # causing some patches (such as FA) to become ineffective.
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.model.llama import \
        lce_forward as llama_lce_forward
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from transformers.models.llama import modeling_llama

    if rope:
        modeling_llama.apply_rotary_pos_emb = liger_rotary_pos_emb

    if cross_entropy:
        modeling_llama.CrossEntropyLoss.forward = LigerCrossEntropyLoss.forward

    if fused_linear_cross_entropy:
        modeling_llama.LlamaForCausalLM.forward = llama_lce_forward

    if rms_norm:
        modeling_llama.LlamaRMSNorm.forward = rms_forward

    if swiglu:
        modeling_llama.LlamaMLP.forward = mlp_forward


# Below is adapted from liger_kernel.transformers.apply_liger_kernel_to_qwen2
def apply_liger_kernel_to_qwen2(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2 models

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but
            more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """
    if not IS_LIGER_KERNEL_AVAILABLE:
        raise ImportError(
            "Liger kernel is not available. Please install it by running `pip install liger-kernel`."
        )
    assert not (
        cross_entropy and fused_linear_cross_entropy
    ), "cross_entropy and fused_linear_cross_entropy cannot both be True."

    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.model.qwen2 import \
        lce_forward as qwen2_lce_forward
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from transformers.models.qwen2 import modeling_qwen2

    if rope:
        modeling_qwen2.apply_rotary_pos_emb = liger_rotary_pos_emb
    if rms_norm:
        modeling_qwen2.Qwen2RMSNorm.forward = rms_forward
    if cross_entropy:
        modeling_qwen2.CrossEntropyLoss.forward = LigerCrossEntropyLoss.forward
    if fused_linear_cross_entropy:
        modeling_qwen2.Qwen2ForCausalLM.forward = qwen2_lce_forward
    if swiglu:
        modeling_qwen2.Qwen2MLP.forward = mlp_forward


def apply_liger_kernel():
    """
    Apply Liger kernels to replace original kernel implementations in HuggingFace Llama and Qwen2 models
    """
    if not IS_LIGER_KERNEL_AVAILABLE:
        logger.warning(
            "The liger kernel will not be used to accelerate the model. " \
            "You can enable it by installing liger-kernel: `pip install liger-kernel`."
        )
        return
    try:
        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
    except Exception as e:
        logger.warning(
            f"Failed to apply liger kernels to the model due to error: {e}. " \
             "This is most likely because of the version mismatch between the transformers and the liger-kernel."
        )
    else:
        logger.info("Liger kernel successfully replaced the operators in the model to enhance performance. " \
                    "You can disable this feature by setting `config.disable_kernel_patches`.")
