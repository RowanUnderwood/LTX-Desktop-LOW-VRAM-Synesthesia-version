"""Block swapping service for low-VRAM inference.

Keeps only N transformer blocks resident on GPU at any time, swapping
the rest to CPU RAM during the forward pass. Works with the LTX-2
dual-stream DiT (48 blocks: video + audio streams share the same
transformer block list).

Usage:
    service = BlockSwapService(blocks_on_gpu=20, device=torch.device("cuda:0"))
    service.install(transformer)   # call once after model load
    service.uninstall(transformer) # call to restore original behaviour
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_BLOCK_SWAP_ATTR = "_block_swap_original_forward"


class BlockSwapService:
    """Installs forward-pass hooks on a transformer to swap blocks on/off GPU.

    Args:
        blocks_on_gpu: How many blocks to keep on GPU simultaneously.
                       0 disables block swapping entirely.
        device:        The GPU device blocks run on during their forward pass.
    """

    def __init__(self, blocks_on_gpu: int, device: torch.device) -> None:
        self.blocks_on_gpu = blocks_on_gpu
        self.device = device
        self._installed_transformers: list[nn.Module] = []

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def install(self, transformer: nn.Module) -> None:
        """Patch transformer blocks with swap hooks. Idempotent."""
        if self.blocks_on_gpu == 0:
            logger.info("BlockSwap disabled (blocks_on_gpu=0)")
            return

        blocks = self._get_blocks(transformer)
        if not blocks:
            logger.warning("BlockSwap: could not find transformer blocks — skipping")
            return

        total = len(blocks)
        if self.blocks_on_gpu >= total:
            logger.info(
                "BlockSwap: blocks_on_gpu=%d >= total=%d — no swapping needed",
                self.blocks_on_gpu, total,
            )
            return

        logger.info(
            "BlockSwap: installing on %d blocks, keeping %d/%d on %s",
            total, self.blocks_on_gpu, total, self.device,
        )

        # Move all blocks to CPU initially.
        for block in blocks:
            block.to("cpu")

        # Patch each block with a swap-in / swap-out forward wrapper.
        for idx, block in enumerate(blocks):
            self._patch_block(block, idx, blocks)

        self._installed_transformers.append(transformer)

    def uninstall(self, transformer: nn.Module) -> None:
        """Remove swap hooks and move all blocks back to GPU."""
        blocks = self._get_blocks(transformer)
        if not blocks:
            return

        for block in blocks:
            orig = getattr(block, _BLOCK_SWAP_ATTR, None)
            if orig is not None:
                block.forward = orig  # type: ignore[method-assign]
                delattr(block, _BLOCK_SWAP_ATTR)
            block.to(self.device)

        if transformer in self._installed_transformers:
            self._installed_transformers.remove(transformer)

        logger.info("BlockSwap: uninstalled, all blocks moved to %s", self.device)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_blocks(self, transformer: nn.Module) -> list[nn.Module]:
        """Find the transformer block list by trying common attribute names.

        LTX model_ledger.transformer() returns X0Model which wraps LTXModel
        as self.velocity_model — so we traverse one level deeper if needed.
        """
        candidates = [transformer]
        # LTX-specific: X0Model/LegacyX0Model wrap the real model in velocity_model
        inner = getattr(transformer, "velocity_model", None)
        if isinstance(inner, nn.Module):
            candidates.append(inner)

        for module in candidates:
            for attr in ("transformer_blocks", "blocks", "layers", "model_blocks"):
                blocks = getattr(module, attr, None)
                if blocks is not None and len(blocks) > 0:
                    return list(blocks)

        # Fallback: find any ModuleList with >4 children.
        for module in candidates:
            for _, child in module.named_children():
                if isinstance(child, nn.ModuleList) and len(child) > 4:
                    return list(child)

        return []

    def _patch_block(
        self,
        block: nn.Module,
        idx: int,
        all_blocks: list[nn.Module],
    ) -> None:
        """Replace block.forward with a version that swaps neighbours."""
        original_forward = block.forward
        setattr(block, _BLOCK_SWAP_ATTR, original_forward)

        blocks_on_gpu = self.blocks_on_gpu
        device = self.device

        def swapped_forward(*args: Any, **kwargs: Any) -> Any:
            # Determine window: keep [idx, idx+blocks_on_gpu) on GPU.
            # Move blocks entering the window to GPU, evict those leaving.
            window_end = idx + blocks_on_gpu

            # Evict the block that just left the window (idx - 1).
            evict_idx = idx - 1
            if evict_idx >= 0:
                prev = all_blocks[evict_idx]
                if next(prev.parameters(), None) is not None:
                    if next(prev.parameters()).device.type != "cpu":
                        prev.to("cpu")

            # Load the next block about to enter the window.
            load_idx = window_end - 1
            if 0 <= load_idx < len(all_blocks):
                nxt = all_blocks[load_idx]
                if next(nxt.parameters(), None) is not None:
                    if next(nxt.parameters()).device.type == "cpu":
                        nxt.to(device)

            # Ensure this block itself is on GPU.
            block.to(device)

            # Run the original forward.
            return original_forward(*args, **kwargs)

        block.forward = swapped_forward  # type: ignore[method-assign]


def build_block_swap_service(
    blocks_on_gpu: int,
    device: torch.device,
) -> BlockSwapService | None:
    """Factory: returns None when swapping is disabled (blocks_on_gpu=0)."""
    if blocks_on_gpu <= 0:
        return None
    return BlockSwapService(blocks_on_gpu=blocks_on_gpu, device=device)