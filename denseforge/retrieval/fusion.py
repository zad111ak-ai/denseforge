"""Reciprocal Rank Fusion (RRF) for merging result lists from multiple retrieval channels."""
from collections import defaultdict
from typing import Any
from loguru import logger


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each result list is a list of dicts that must contain at least an ``id``
    (or ``doc_id``) key identifying the document.  Returns a single fused list
    sorted by descending RRF score.

    Args:
        result_lists: One ranked list per retrieval channel.
        k: RRF smoothing constant (default 60, standard in the literature).
        weights: Optional per-channel weights.  If *None*, equal weights are
            used (1 / number_of_channels).

    Returns:
        Fused and re-ranked list of dicts, each containing the original keys
        plus ``rrf_score`` and ``channels`` (list of channel indices that
        contributed).
    """
    if not result_lists:
        logger.warning("reciprocal_rank_fusion called with empty result_lists")
        return []

    num_channels = len(result_lists)
    if weights is None:
        weights = [1.0 / num_channels] * num_channels
    elif len(weights) != num_channels:
        raise ValueError(
            f"weights length ({len(weights)}) must match number of result lists ({num_channels})"
        )

    # Accumulate RRF scores per document
    doc_scores: dict[str, float] = defaultdict(float)
    doc_channels: dict[str, set[int]] = defaultdict(set)
    # Keep one copy of each doc's payload
    doc_payloads: dict[str, dict[str, Any]] = {}

    for channel_idx, results in enumerate(result_lists):
        weight = weights[channel_idx]
        for rank, result in enumerate(results, start=1):
            doc_key = str(result.get("id", result.get("doc_id", "")))
            if not doc_key:
                logger.debug(
                    "Skipping result without id/doc_id in channel {ch}: {r}",
                    ch=channel_idx, r=result,
                )
                continue

            rrf_contribution = weight / (k + rank)
            doc_scores[doc_key] += rrf_contribution
            doc_channels[doc_key].add(channel_idx)
            # Store payload (first occurrence wins for metadata)
            if doc_key not in doc_payloads:
                doc_payloads[doc_key] = dict(result)

    # Sort by fused score descending
    ranked_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for doc_key in ranked_keys:
        merged = dict(doc_payloads[doc_key])
        merged["rrf_score"] = doc_scores[doc_key]
        merged["channels"] = sorted(doc_channels[doc_key])
        fused_results.append(merged)

    logger.info(
        "RRF fused {n} unique documents from {c} channels (k={k})",
        n=len(fused_results), c=num_channels, k=k,
    )
    return fused_results
