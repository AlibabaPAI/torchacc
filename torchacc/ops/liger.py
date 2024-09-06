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
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
    """
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
    from liger_kernel.transformers.model.llama import \
        lce_forward as llama_lce_forward
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from transformers.models.llama import modeling_llama

    def rms_forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            0.0,  # offset
            "llama",  # casting_mode
        )

    def mlp_forward(self, x):
        if self.config.hidden_act not in ["silu", "swish"]:
            raise ValueError(
                f"Activation function {self.config.hidden_act} not supported.")
        return self.down_proj(
            LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))

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
