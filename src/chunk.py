"""
Transcript chunker (two-tier, boundary-aware).

Input: list of transcript segments with timestamps.
Each segment:
{
  "start": 12.34,   # seconds
  "end": 18.90,
  "text": "...."
}

Output:
- fine chunks (retrieval)
- coarse chunks (summarization / context)

All chunks preserve timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import re
import math


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    seg_id: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert Segment to a dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    tier: str  # "fine" or "coarse"
    start: float
    end: float
    text: str
    segment_ids: List[int]


# ----------------------------
# Utilities
# ----------------------------

def format_ts(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def simple_token_count(text: str) -> int:
    """
    Approximate token count without external dependencies.
    """
    parts = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return len(parts)


# Common discourse / transition cues
_DISCOURSE_RE = re.compile(
    r"\b("
    r"anyway|so|now|next|okay|"
    r"let's|let us|"
    r"the key|important|"
    r"to summarize|in summary|"
    r"moving on|"
    r"that said|on the other hand"
    r")\b",
    flags=re.IGNORECASE,
)


def is_sentence_boundary(text: str) -> bool:
    text = text.strip()
    return bool(text) and text[-1] in ".!?…"


def normalize_segments(raw_segments: List[Dict[str, Any]]) -> List[Segment]:
    segs: List[Segment] = []
    for i, r in enumerate(raw_segments):
        text = (r.get("text") or "").strip()
        if not text:
            continue
        segs.append(
            Segment(
                start=float(r["start"]),
                end=float(r["end"]),
                text=text,
                seg_id=i,
            )
        )
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


# ----------------------------
# Boundary scoring (speaker-agnostic)
# ----------------------------

def boundary_score(prev_seg: Segment, next_seg: Segment) -> float:
    """
    Score a potential cut between prev_seg and next_seg.
    Higher = better boundary.
    """
    score = 0.0

    # Pause gap
    gap = max(0.0, next_seg.start - prev_seg.end)
    if gap >= 4.0:
        score += 2.5
    elif gap >= 2.0:
        score += 1.5
    elif gap >= 1.0:
        score += 0.8

    # Discourse marker at the start of next segment
    if _DISCOURSE_RE.search(next_seg.text):
        score += 1.2

    # Sentence-ending punctuation
    if is_sentence_boundary(prev_seg.text):
        score += 0.7

    return score


def find_best_cut_index(
    segs: List[Segment],
    start_idx: int,
    hard_end_idx: int,
    lookback_frac: float,
) -> int:
    """
    Choose best boundary near the hard_end_idx.
    Returns index where chunk should end (exclusive).
    """
    if hard_end_idx <= start_idx + 1:
        return hard_end_idx

    window_len = hard_end_idx - start_idx
    lookback = max(1, int(math.ceil(window_len * lookback_frac)))
    scan_from = max(start_idx + 1, hard_end_idx - lookback)

    best_idx = hard_end_idx
    best_score = -1e9

    for i in range(scan_from, hard_end_idx):
        score = boundary_score(segs[i - 1], segs[i])

        # Prefer cuts closer to hard_end_idx
        closeness = (i - scan_from) / max(1, (hard_end_idx - scan_from))
        score += 0.2 * closeness

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


# ----------------------------
# Core chunking logic
# ----------------------------

def build_chunks(
    segs: List[Segment],
    tier: str,
    target_tokens: int,
    overlap_frac: float,
    min_tokens: int,
    lookback_frac: float,
) -> List[Chunk]:
    assert 0.0 <= overlap_frac < 1.0
    assert tier in ("fine", "coarse")

    chunks: List[Chunk] = []
    n = len(segs)
    if n == 0:
        return chunks

    idx = 0
    chunk_num = 0
    seg_tokens = [simple_token_count(s.text) for s in segs]

    while idx < n:
        tok_sum = 0
        end_idx = idx

        while end_idx < n and tok_sum < target_tokens:
            tok_sum += seg_tokens[end_idx]
            end_idx += 1

        # Avoid tiny final chunk
        remaining = sum(seg_tokens[end_idx:]) if end_idx < n else 0
        if remaining < min_tokens and end_idx < n:
            end_idx = n

        cut_idx = find_best_cut_index(
            segs=segs,
            start_idx=idx,
            hard_end_idx=end_idx,
            lookback_frac=lookback_frac,
        )

        if cut_idx <= idx:
            cut_idx = end_idx

        chunk_segs = segs[idx:cut_idx]
        if not chunk_segs:
            break

        start_ts = chunk_segs[0].start
        end_ts = chunk_segs[-1].end
        text = " ".join(s.text for s in chunk_segs)

        chunks.append(
            Chunk(
                chunk_id=f"{tier}-{chunk_num:04d}-{format_ts(start_ts)}",
                tier=tier,
                start=start_ts,
                end=end_ts,
                text=text,
                segment_ids=[s.seg_id for s in chunk_segs],
            )
        )
        chunk_num += 1

        if cut_idx >= n:
            break

        # Step forward with overlap
        chunk_len = cut_idx - idx
        step = max(1, int(round(chunk_len * (1.0 - overlap_frac))))
        idx += step

    return chunks


def chunk_transcript_two_tier(
    raw_segments: List[Dict[str, Any]],
    fine_target_tokens: int = 380,
    fine_overlap: float = 0.20,
    coarse_target_tokens: int = 1200,
    coarse_overlap: float = 0.12,
) -> Dict[str, List[Dict[str, Any]]]:
    segs = normalize_segments(raw_segments)

    fine = build_chunks(
        segs,
        tier="fine",
        target_tokens=fine_target_tokens,
        overlap_frac=fine_overlap,
        min_tokens=140,
        lookback_frac=0.25,
    )

    coarse = build_chunks(
        segs,
        tier="coarse",
        target_tokens=coarse_target_tokens,
        overlap_frac=coarse_overlap,
        min_tokens=240,
        lookback_frac=0.20,
    )

    return {
        "fine": [asdict(c) for c in fine],
        "coarse": [asdict(c) for c in coarse],
    }


# ----------------------------
# Example
# ----------------------------

if __name__ == "__main__":
    demo = [
        {"start": 0.0, "end": 4.0, "text": "Welcome to the show. Today we're talking about pricing."},
        {"start": 4.2, "end": 10.0, "text": "Anyway, I think the first key point is segmentation."},
        {"start": 10.1, "end": 18.0, "text": "If you bundle features wrong, you confuse willingness to pay."},
        {"start": 18.5, "end": 26.0, "text": "Now let's move on to experiments and how to run them."},
        {"start": 26.2, "end": 34.0, "text": "A B tests fail when you don't define guardrails."},
        {"start": 34.3, "end": 42.0, "text": "To summarize, you need a clear hypothesis and stopping rule."},
    ]

    chunks = chunk_transcript_two_tier(demo)

    print("FINE CHUNKS")
    for c in chunks["fine"]:
        print(c["chunk_id"], format_ts(c["start"]), "-", format_ts(c["end"]))
        print(" ", c["text"])
        print()

    print("COARSE CHUNKS")
    for c in chunks["coarse"]:
        print(c["chunk_id"], format_ts(c["start"]), "-", format_ts(c["end"]))
        print(" ", c["text"])
        print()
