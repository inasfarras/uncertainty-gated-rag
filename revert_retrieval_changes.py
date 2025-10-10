"""Revert retrieval parameter changes back to original values."""

import re


def revert_config():
    """Revert src/agentic_rag/config.py to original retrieval settings."""

    config_path = "src/agentic_rag/config.py"

    with open(config_path, encoding="utf-8") as f:
        content = f.read()

    # Revert HYBRID_ALPHA: 0.4 → 0.6 (back to 60% vector, 40% BM25)
    content = re.sub(
        r"HYBRID_ALPHA: float = 0\.4.*",
        "HYBRID_ALPHA: float = 0.6  # Weight for vector vs BM25 (0.6 = 60% vector, 40% BM25)",
        content,
    )

    # Revert ANCHOR_BONUS: 0.20 → 0.07 (back to original)
    content = re.sub(
        r"ANCHOR_BONUS: float = 0\.20.*",
        "ANCHOR_BONUS: float = 0.07  # Score bonus for candidates containing question anchors",
        content,
    )

    # Revert MMR_LAMBDA: 0.2 → 0.45 (back to original)
    content = re.sub(r"MMR_LAMBDA: float = 0\.2.*", "MMR_LAMBDA: float = 0.45", content)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Reverted retrieval settings to original values:")
    print("   HYBRID_ALPHA: 0.4 → 0.6")
    print("   ANCHOR_BONUS: 0.20 → 0.07")
    print("   MMR_LAMBDA: 0.2 → 0.45")
    print()
    print("These changes made performance WORSE, so reverting.")


if __name__ == "__main__":
    revert_config()
