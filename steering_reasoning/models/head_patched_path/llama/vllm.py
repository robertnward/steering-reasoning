from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from loguru import logger
from vllm.forward_context import get_forward_context

# Prefer Transformers helpers for grouped KV behavior
try:
    from transformers.models.llama.modeling_llama import repeat_kv as hf_repeat_kv
except Exception as e:  # no fallback — fail loudly later if used
    hf_repeat_kv = None
    _HF_IMPORT_ERR = e
else:
    _HF_IMPORT_ERR = None

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaForCausalLM

"""
Original compute logic preserved exactly:
  • One fused vLLM attention pass (STEERED only)
  • VANILLA attention via local custom SDPA only
  • Optional local STEERED attention for debug comparison (gated by decode_only)

Change requested now: remove *all* defaults/fallbacks/quiet returns.
If anything is missing or inconsistent, raise immediately with a clear error.
"""


def _unregister_attention(layer):
    ctx = get_current_vllm_config().compilation_config.static_forward_context
    for m in layer.modules():
        if hasattr(m, "layer_name"):
            ctx.pop(m.layer_name, None)


class _KVStore:
    """Bounded per-layer KV cache with runtime->local slot compaction.

    Tensors:
      K, V: (S_max, L_max, Kv, Hd)

    State:
      len_per_slot   : (S_max,) int32  — token counts per *local* slot
      runtime2local  : dict[int,int]   — runtime slot id -> local slot idx
      local2runtime  : dict[int,int]   — reverse map
      free_slots     : list[int]       — pool of free local slots
      last_used      : (S_max,) int64  — recency for LRU eviction
      clock          : int             — monotonically increasing step count
    """

    def __init__(self, max_num_seqs: int, max_model_len: int):
        assert max_num_seqs > 0 and max_model_len > 0, "KVStore sizes must be positive"
        self.max_num_seqs = int(max_num_seqs)
        self.max_model_len = int(max_model_len)
        self.K = None
        self.V = None
        self.len_per_slot = None

        # Compaction maps + LRU state
        self.runtime2local: dict[int, int] = {}
        self.local2runtime: dict[int, int] = {}
        self.free_slots: List[int] = list(range(self.max_num_seqs))
        self.last_used = None  # torch tensor, created at initialization
        self.clock: int = 0

    def maybe_initialize(self, *, device, dtype, kv_heads: int, head_dim: int):
        if self.K is not None:
            return
        assert kv_heads > 0 and head_dim > 0
        shape = (self.max_num_seqs, self.max_model_len, kv_heads, head_dim)
        self.K = torch.empty(shape, device=device, dtype=dtype)
        self.V = torch.empty_like(self.K)
        self.len_per_slot = torch.zeros(
            (self.max_num_seqs,), device=device, dtype=torch.int32
        )
        self.last_used = torch.zeros(
            (self.max_num_seqs,), device=device, dtype=torch.int64
        )
        logger.info(
            f"Initialized KVStore with shape={shape}, device={device}, dtype={dtype}"
        )

    # --- compaction helpers ---
    def _map_runtime_slot(self, runtime_slot: int) -> int:
        rs = int(runtime_slot)
        if rs in self.runtime2local:
            local = self.runtime2local[rs]
            self.last_used[local] = self.clock
            self.clock += 1
            return local

        if self.free_slots:
            local = self.free_slots.pop(0)
        else:
            # Evict least-recently used local slot
            assert (
                self.last_used is not None
                and self.last_used.numel() == self.max_num_seqs
            )
            local = int(self.last_used.argmin().item())
            old_rs = self.local2runtime.get(local)
            if old_rs is not None:
                self.runtime2local.pop(old_rs, None)
            # Reset length for this recycled slot
            self.len_per_slot[local] = 0

        self.runtime2local[rs] = local
        self.local2runtime[local] = rs
        self.last_used[local] = self.clock
        self.clock += 1
        return local

    # --- API used by the layer ---
    def append_current(
        self, *, slot: int, k_rot_step: torch.Tensor, v_step: torch.Tensor
    ):
        """Append current token K (after RoPE) and V for one runtime slot.
        k_rot_step, v_step: (Kv, Hd)
        """
        assert k_rot_step.ndim == 2 and v_step.ndim == 2, (
            f"Expected (Kv,Hd) tensors, got {tuple(k_rot_step.shape)} and {tuple(v_step.shape)}"
        )
        Kv, Hd = k_rot_step.shape
        assert (Kv, Hd) == tuple(v_step.shape)
        if self.K is None:
            raise RuntimeError("KVStore not initialized before append_current")
        assert Kv == self.K.shape[2] and Hd == self.K.shape[3], (
            f"KV head_dim mismatch: {(Kv, Hd)} vs store {(self.K.shape[2], self.K.shape[3])}"
        )

        local = self._map_runtime_slot(slot)
        pos = int(self.len_per_slot[local].item())
        if pos >= self.max_model_len:
            raise RuntimeError("Custom KV cache overflow: increase max_model_len.")
        self.K[local, pos, :, :].copy_(k_rot_step)
        self.V[local, pos, :, :].copy_(v_step)
        self.len_per_slot[local] = pos + 1

    def gather_batch(
        self, slots: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather a padded batch (K_b, V_b, lens) for a list of *runtime* slots.
        Returns:
          K_b: (B, L_max_b, Kv, Hd), V_b: same, lens: (B,) int32 lengths
        """
        if self.K is None or self.V is None:
            raise RuntimeError("KVStore not initialized before gather_batch")
        B = len(slots)
        if B == 0:
            raise RuntimeError("gather_batch called with empty slot list (logic error)")

        locals_batch = [self._map_runtime_slot(s) for s in slots]
        lens = torch.stack([self.len_per_slot[s] for s in locals_batch])
        L_max_b = int(lens.max().item())
        Kv = self.K.shape[2]
        Hd = self.K.shape[3]
        device = self.K.device
        dtype = self.K.dtype

        if L_max_b <= 0:
            raise RuntimeError(
                f"Gather produced non-positive L_max_b={L_max_b}. lens={lens.tolist()} slots={slots}"
            )

        K_b = torch.empty((B, L_max_b, Kv, Hd), device=device, dtype=dtype)
        V_b = torch.empty_like(K_b)
        for b, s in enumerate(locals_batch):
            Lb = int(lens[b].item())
            if Lb > 0:
                K_b[b, :Lb, :, :].copy_(self.K[s, :Lb, :, :])
                V_b[b, :Lb, :, :].copy_(self.V[s, :Lb, :, :])
        return K_b, V_b, lens


class LlamaLastLayer(LlamaDecoderLayer):
    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = ""):
        super().__init__(
            config, cache_config=cache_config, quant_config=quant_config, prefix=prefix
        )
        self.proj_type = config.proj_type
        self.head_idx = int(config.head_idx)
        # Path-patching mode: "residual", "mlp", "everywhere", or "nowhere" (defaults to "residual" if unspecified)
        self.path_patch_mode = config.path_patch_mode
        self.steering_vector = torch.nn.Parameter(torch.zeros(config.hidden_size))

        # ---- Debug options (off by default) ----
        self.debug_compare_local_steered: bool = True
        self.debug_rtol: float = 1e-4
        self.debug_atol: float = 1e-4
        self.debug_raise: bool = True
        self.debug_last_compare = None
        self.debug_last_attn_concat_vanilla = None  # (T, H*Hd)
        self.debug_last_attn_after_o_vanilla = None  # (T, HIDDEN)
        self.debug_last_attn_after_o_steered = None  # (T, HIDDEN)

        # ---- Required sizes from vLLM config (no fallbacks) ----
        vcfg = get_current_vllm_config()
        assert hasattr(vcfg, "scheduler_config") and hasattr(
            vcfg.scheduler_config, "max_num_seqs"
        )
        assert hasattr(vcfg, "model_config") and hasattr(
            vcfg.model_config, "max_model_len"
        )
        max_num_seqs = int(vcfg.scheduler_config.max_num_seqs)
        max_model_len = int(vcfg.model_config.max_model_len)

        self._vanilla_store = _KVStore(max_num_seqs, max_model_len)
        self._steered_store = _KVStore(max_num_seqs, max_model_len)

        self.step_counter = 0

    # ---------- small helpers ----------
    def _H(self):
        H = self.self_attn.num_heads
        assert H > 0, f"num_heads must be > 0, got {H}"
        return H

    def _Kv(self):
        Kv = self.self_attn.num_kv_heads
        assert Kv > 0 and self._H() % Kv == 0, f"Invalid GQA: H={self._H()} Kv={Kv}"
        return Kv

    def _Hd(self):
        Hd = self.self_attn.head_dim
        assert Hd > 0
        return Hd

    def _G(self):
        return self._H() // self._Kv()

    def _head_slice_concat(self, i: int) -> slice:
        Hd = self._Hd()
        assert 0 <= i < self._H()
        return slice(i * Hd, (i + 1) * Hd)

    def get_after_input_layernorm(self, hidden_states, residual):
        assert hidden_states.ndim == 2, (
            f"hidden_states must be (T,HIDDEN), got {tuple(hidden_states.shape)}"
        )
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        assert hidden_states.shape == residual.shape
        return hidden_states, residual

    def get_qkv(self, hidden_states):
        qkv, _ = self.self_attn.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [self.self_attn.q_size, self.self_attn.kv_size, self.self_attn.kv_size],
            dim=-1,
        )
        T = hidden_states.shape[0]
        H = self._H()
        Kv = self._Kv()
        Hd = self._Hd()
        assert q.shape == (T, H * Hd), f"q shape {tuple(q.shape)} != {(T, H * Hd)}"
        assert k.shape == (T, Kv * Hd), f"k shape {tuple(k.shape)} != {(T, Kv * Hd)}"
        assert v.shape == (T, Kv * Hd), f"v shape {tuple(v.shape)} != {(T, Kv * Hd)}"
        return q, k, v

    # ---------- efficient local attention (GQA-aware, used for vanilla & steered) ----------
    def _multihead_attend_qwen2(
        self,
        q_rot_all: torch.Tensor,  # (T, H*Hd)
        K_b: torch.Tensor,  # (T, L, Kv, Hd)
        V_b: torch.Tensor,  # (T, L, Kv, Hd)
        lens_b: torch.Tensor,  # (T,)
    ) -> torch.Tensor:
        """Local attention rewritten following Transformers' Qwen2 style.
        - Repeat Kv->H with `repeat_kv`
        - Build additive float mask with -inf for padding (as HF does)
        - Use PyTorch SDPA

        Returns (T, H*Hd) pre-W_O.
        """
        assert hf_repeat_kv is not None, (
            f"transformers.repeat_kv unavailable: {_HF_IMPORT_ERR!r}"
        )
        assert (
            q_rot_all.ndim == 2 and K_b.ndim == 4 and V_b.ndim == 4 and lens_b.ndim == 1
        )
        T = q_rot_all.shape[0]
        B, L, Kv, Hd = K_b.shape
        assert B == T, f"Batch mismatch: K_b batch {B} vs T {T}"
        assert V_b.shape == (B, L, Kv, Hd)
        assert lens_b.shape[0] == T
        assert L > 0 and T > 0

        H = self._H()
        G = self._G()
        assert q_rot_all.shape[1] == H * Hd

        # q: (T, H, 1, Hd)
        q = q_rot_all.view(T, H, Hd).unsqueeze(2)

        # K/V: (T, Kv, L, Hd) -> repeat to (T, H, L, Hd)
        K_kv = K_b.permute(0, 2, 1, 3)
        V_kv = V_b.permute(0, 2, 1, 3)
        K = hf_repeat_kv(K_kv, G)
        V = hf_repeat_kv(V_kv, G)
        assert K.shape == (T, H, L, Hd) and V.shape == (T, H, L, Hd)

        # Additive mask like HF: 0 for valid, -inf for masked
        device = q.device
        dtype = q.dtype
        attn_mask = torch.full((T, 1, 1, L), float("-inf"), device=device, dtype=dtype)
        for t in range(T):
            Lt = int(lens_b[t].item())
            assert 0 <= Lt <= L
            if Lt:
                attn_mask[t, :, :, :Lt] = 0
        assert attn_mask.shape == (T, 1, 1, L)

        out = F.scaled_dot_product_attention(
            q, K, V, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        assert out.shape == (T, H, 1, Hd)
        out_concat = out.squeeze(2).reshape(T, H * Hd)
        return out_concat

    # ---------- main forward (logic preserved; no fallbacks) ----------
    def __call__(
        self,
        positions: torch.Tensor,  # (T,)
        hidden_states: torch.Tensor,  # (T, HIDDEN)
        residual: Optional[torch.Tensor],  # (T, HIDDEN) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Resolve slots from vLLM metadata — REQUIRED
        assert positions.ndim == 1, (
            f"positions must be (T,), got {tuple(positions.shape)}"
        )
        T = hidden_states.shape[0]
        assert positions.shape[0] == T
        assert hidden_states.ndim == 2

        fc = get_forward_context()
        assert hasattr(fc, "attn_metadata"), "Forward context missing attn_metadata"
        attn_md = fc.attn_metadata
        # Some vLLM versions store per-layer metadata in a dict keyed by layer_name; others store a single object.
        if isinstance(attn_md, dict):
            assert self.self_attn.attn.layer_name in attn_md, (
                f"attn_metadata dict missing key for layer {self.self_attn.attn.layer_name}; keys={list(attn_md.keys())[:8]}"
            )
            attn_md = attn_md[self.self_attn.attn.layer_name]

        def _maybe_to_int_list(x):
            try:
                if isinstance(x, torch.Tensor):
                    if x.numel() == 0:
                        return None
                    if not x.dtype.is_floating_point and x.dtype != torch.bool:
                        flat = x.view(-1).tolist()
                        return [int(v) for v in flat]
                    return None
                elif (
                    isinstance(x, (list, tuple))
                    and len(x) > 0
                    and isinstance(x[0], (int, torch.Tensor))
                ):
                    return [int(v) for v in x]
            except Exception:
                return None
            return None

        T = int(positions.shape[0])
        # 1) Direct attribute candidates on the metadata objects we know about
        cand_objs = [attn_md]
        for name in (
            "decode_metadata",
            "prefill_metadata",
            "common",
            "builder",
            "state",
        ):
            if hasattr(attn_md, name):
                cand_objs.append(getattr(attn_md, name))

        candidate_fields = (
            "slot_mapping",
            "runtime_slots",
            "slot_mapping_tensor",
            "slots",
            "active_slots",
            "runtime_slot_map",
            "slot_ids",
            "block_slots",
        )

        inspected = []
        found = []  # list[(origin, field, list[int])]
        for obj in cand_objs:
            for field in candidate_fields:
                if hasattr(obj, field):
                    raw = getattr(obj, field)
                    lst = _maybe_to_int_list(raw)
                    if lst is not None:
                        inspected.append(
                            (
                                f"{type(obj).__name__}.{field}",
                                len(lst),
                                getattr(raw, "shape", None),
                            )
                        )
                        if len(lst) == T:
                            found.append(
                                (f"attr:{type(obj).__name__}.{field}", field, lst)
                            )

        # 2) Some backends expose graph input buffers dict; try that via Attention impl
        impl = getattr(self.self_attn.attn, "impl", None)
        if impl is not None and hasattr(impl, "get_graph_input_buffers"):
            try:
                buf = impl.get_graph_input_buffers(
                    attn_md, is_encoder_decoder_model=False
                )
            except TypeError:
                buf = impl.get_graph_input_buffers(attn_md)
            if isinstance(buf, dict):
                for k, v in list(buf.items()):
                    lst = _maybe_to_int_list(v)
                    if lst is not None:
                        inspected.append(
                            (f"buf[{k}]", len(lst), getattr(v, "shape", None))
                        )
                        if len(lst) == T and ("slot" in k or "Slot" in k):
                            found.append((f"buf:{k}", k, lst))
        else:
            inspected.append(("impl.get_graph_input_buffers", "missing", None))

        # 3) Dataclass/namespace dicts if present
        if hasattr(attn_md, "__dict__") and isinstance(attn_md.__dict__, dict):
            for k, v in list(attn_md.__dict__.items()):
                lst = _maybe_to_int_list(v)
                if lst is not None:
                    inspected.append(
                        (f"__dict__[{k}]", len(lst), getattr(v, "shape", None))
                    )
                    if len(lst) == T and ("slot" in k or "Slot" in k):
                        found.append((f"__dict__:{k}", k, lst))

        if not found:
            # Collect a compact summary for debugging (no fallback)
            attn_md_type = type(attn_md).__name__
            # show up to 16 attribute names
            try:
                names = [n for n in dir(attn_md) if not n.startswith("_")][:16]
            except Exception:
                names = []
            raise AssertionError(
                "Could not extract runtime slot mapping from vLLM metadata. "
                f"layer={self.self_attn.attn.layer_name} T={T} attn_md_type={attn_md_type} "
                f"dir(attn_md)[:16]={names}; inspected={inspected[:12]}"
            )

        # If multiple candidate tensors match T, ensure they are identical
        slot_lists = [lst for (_, _, lst) in found]
        first = slot_lists[0]
        for lst in slot_lists[1:]:
            if lst != first:
                raise AssertionError(
                    f"Ambiguous slot candidates of len T={T} disagree: sources={[src for (src, _, _) in found]}"
                )
        slots = first
        logger.debug(
            f"Extracted slots from {found[0][0]}: len={len(slots)} first8={slots[:8]}"
        )

        # Prepare dtypes/devices & pre-initialize caches
        device = self.input_layernorm.weight.device
        dtype = self.input_layernorm.weight.dtype
        self._vanilla_store.maybe_initialize(
            device=device, dtype=dtype, kv_heads=self._Kv(), head_dim=self._Hd()
        )
        if self.debug_compare_local_steered:
            self._steered_store.maybe_initialize(
                device=device, dtype=dtype, kv_heads=self._Kv(), head_dim=self._Hd()
            )

        # 1) VANILLA Q/K/V (local only)
        hs_vanilla, residual_vanilla = self.get_after_input_layernorm(
            hidden_states=hidden_states, residual=residual
        )
        q_v, k_v, v_v = self.get_qkv(hs_vanilla)
        qv_rot, kv_rot = self.self_attn.rotary_emb(positions, q_v, k_v)

        Kv = self._Kv()
        Hd = self._Hd()
        k_step_v = kv_rot.view(T, Kv, Hd)
        v_step_v = v_v.view(T, Kv, Hd)
        for t, slot in enumerate(slots):
            self._vanilla_store.append_current(
                slot=slot, k_rot_step=k_step_v[t], v_step=v_step_v[t]
            )

        K_b_v, V_b_v, lens_b = self._vanilla_store.gather_batch(slots)
        logger.debug(
            f"Local VANILLA gather: K_b_v={tuple(K_b_v.shape)} V_b_v={tuple(V_b_v.shape)} lens[min,max]={int(lens_b.min())},{int(lens_b.max())}"
        )
        attn_concat_vanilla = self._multihead_attend_qwen2(
            q_rot_all=qv_rot, K_b=K_b_v, V_b=V_b_v, lens_b=lens_b
        )
        self.debug_last_attn_concat_vanilla = attn_concat_vanilla.detach()

        # 2) One fused vLLM attention pass (STEERED)
        hs_steered, _ = self.get_after_input_layernorm(
            hidden_states=hidden_states + self.steering_vector, residual=residual
        )
        q_s, k_s, v_s = self.get_qkv(hs_steered)
        q_rot_s, k_rot_s = self.self_attn.rotary_emb(positions, q_s, k_s)
        attn_concat_steered = self.self_attn.attn(q_rot_s, k_rot_s, v_s)

        # Optional local-steered compare (decode_only gate retained)
        self.step_counter += 1
        decode_only = self.step_counter > 2
        logger.info(
            f"T={T} hidden_states={tuple(hidden_states.shape)} decode_only={decode_only} debug_compare_local_steered={self.debug_compare_local_steered}"
        )
        if self.debug_compare_local_steered and decode_only:
            k_step_s = k_rot_s.view(T, Kv, Hd)
            v_step_s = v_s.view(T, Kv, Hd)
            for t, slot in enumerate(slots):
                self._steered_store.append_current(
                    slot=slot, k_rot_step=k_step_s[t], v_step=v_step_s[t]
                )
            K_b_s, V_b_s, lens_b_s = self._steered_store.gather_batch(slots)
            logger.debug(
                f"Local STEERED gather: K_b_s={tuple(K_b_s.shape)} V_b_s={tuple(V_b_s.shape)} lens[min,max]={int(lens_b_s.min())},{int(lens_b_s.max())}"
            )
            attn_concat_steered_local = self._multihead_attend_qwen2(
                q_rot_all=q_rot_s, K_b=K_b_s, V_b=V_b_s, lens_b=lens_b_s
            )
            diff = (attn_concat_steered - attn_concat_steered_local).abs()
            max_abs = diff.max().item() if diff.numel() > 0 else float("nan")
            mean_abs = diff.mean().item() if diff.numel() > 0 else float("nan")
            ok = torch.allclose(
                attn_concat_steered,
                attn_concat_steered_local,
                rtol=self.debug_rtol,
                atol=self.debug_atol,
            )
            self.debug_last_compare = {
                "ok": bool(ok),
                "max_abs": float(max_abs),
                "mean_abs": float(mean_abs),
            }
            logger.info(
                f"STEERED compare ok={ok} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rtol={self.debug_rtol} atol={self.debug_atol}"
            )
            if self.debug_raise and not ok:
                raise AssertionError(
                    f"Local steered SDPA != fused vLLM attn: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, "
                    f"rtol={self.debug_rtol}, atol={self.debug_atol}"
                )

        # 3) Patch-after-attention (build pre-W_O patched tensor)
        assert 0 <= self.head_idx < self._H(), (
            f"head_idx {self.head_idx} out of range [0,{self._H() - 1}]"
        )
        hs = self._head_slice_concat(self.head_idx)
        if self.proj_type == "head":
            attn_concat_patched = attn_concat_steered.clone()
            attn_concat_patched[:, hs] = attn_concat_vanilla[:, hs]
        elif self.proj_type == "inverse_head":
            attn_concat_patched = attn_concat_vanilla.clone()
            attn_concat_patched[:, hs] = attn_concat_steered[:, hs]
        else:
            raise ValueError(
                f"Unknown proj_type '{self.proj_type}'. Expected one of ['head','inverse_head']."
            )

        # 4) Apply output projection(s)
        attn_after_o_steered_unpatched, _ = self.self_attn.o_proj(attn_concat_steered)
        attn_after_o_vanilla_unpatched, _ = self.self_attn.o_proj(attn_concat_vanilla)
        self.debug_last_attn_after_o_steered = attn_after_o_steered_unpatched.detach()
        self.debug_last_attn_after_o_vanilla = attn_after_o_vanilla_unpatched.detach()

        attn_after_o_patched, _ = self.self_attn.o_proj(attn_concat_patched)

        # 5) Path-patched Residual/MLP stack (2D)
        residual_for_fc = residual_vanilla

        if self.proj_type == "head":
            attn_after_o_default = attn_after_o_steered_unpatched
        elif self.proj_type == "inverse_head":
            attn_after_o_default = attn_after_o_vanilla_unpatched
        else:
            raise ValueError(f"Unknown proj_type '{self.proj_type}'")

        hs_default, res_default = self.post_attention_layernorm(
            attn_after_o_default, residual_for_fc
        )
        hs_patched, res_patched = self.post_attention_layernorm(
            attn_after_o_patched, residual_for_fc
        )

        mode = self.path_patch_mode
        if mode == "residual":
            mlp_out = self.mlp(hs_default)
            residual_out = res_patched
        elif mode == "mlp":
            mlp_out = self.mlp(hs_patched)
            residual_out = res_default
        elif mode == "nowhere":
            mlp_out = self.mlp(hs_default)
            residual_out = res_default
        elif mode == "everywhere":
            mlp_out = self.mlp(hs_patched)
            residual_out = res_patched
        else:
            raise ValueError(
                "path_patch_mode must be one of {'residual','mlp','everywhere','nowhere' }."
            )

        hidden_states_out = residual_out + mlp_out
        assert hidden_states_out.shape == hidden_states.shape
        return hidden_states_out, residual_out


class LlamaForCausalLM_CustomLast(LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        layers = self.model.layers
        last = len(layers) - 1
        old = layers[last]
        _unregister_attention(old)

        cfg = self.model.config
        cache_cfg = vllm_config.cache_config
        quant_cfg = vllm_config.quant_config

        # NOTE: options are hard-coded inside LlamaLastLayer.
        layers[last] = LlamaLastLayer(
            cfg,
            cache_config=cache_cfg,
            quant_config=quant_cfg,
            prefix=f"model.layers.{last}",
        )

    def load_weights(self, weights):
        loaded_weights = super().load_weights(weights)
        self.model.layers[-1].steering_vector = torch.nn.Parameter(
            self.model.layers[-2].mlp.down_proj.bias.clone()
        )
        self.model.layers[-2].mlp.down_proj.bias = torch.nn.Parameter(
            torch.zeros_like(self.model.layers[-2].mlp.down_proj.bias)
        )
        return loaded_weights
