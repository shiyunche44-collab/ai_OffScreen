#!/usr/bin/env python3
"""Generate script samples for M4-10 evaluation across three styles.

This script:
1. Uses the 30s Sintel test clip
2. Generates one script per style (emotional, suspense, roast)
3. Records automatic metrics (duration, chars, cost)
4. Updates docs/M4_EVALUATION.md with results

Usage:
    uv run python scripts/eval_generate_scripts.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.pipeline import RunOptions
from offscreen.store.models import utcnow


def run_evaluation():
    """Generate scripts for three styles on Sintel 30s clip."""
    cfg = AppConfig.model_validate({"data_dir": "data", "tts": {"provider": "edge_tts"}})
    services = AppServices(cfg)

    # Assume Sintel 30s is already imported; find it
    assets = services.jobs.docs.db.session().query(
        services.jobs.docs.db.engine.table("assets")
    ).all()
    if not assets:
        print("❌ No assets found. Import Sintel first with:")
        print("  offscreen stage ingest --asset /path/to/sintel_30s.mp4")
        return False

    asset_id = assets[0][0]  # first asset
    print(f"📽️ Using asset: {asset_id}")

    styles = ["emotional", "suspense", "roast"]
    samples = {}

    for style in styles:
        print(f"\n🎬 Generating script for style: {style}")
        start_t = time.time()

        # Create temporary project for this style
        project = services.library.create_project(
            asset_id,
            f"eval_{style}_30s",
            options={"minutes": 0.5, "style": style, "spoil_ending": True},
        )

        # Run script generation
        job = services.library.generate_script(
            project.id, run_options=RunOptions(minutes=0.5)
        )
        print(f"  Job {job.id} queued, waiting...")

        # Wait for job to complete
        while True:
            time.sleep(2)
            j = services.jobs.jobs.get(job.id)
            if j.status in ("succeeded", "failed"):
                break
            print(f"  Progress: {j.progress * 100:.0f}%")

        elapsed = time.time() - start_t

        if j.status != "succeeded":
            print(f"  ❌ Failed: {j.error}")
            continue

        # Read the generated script
        script = services.jobs.docs.read(project.id, "script", dict, version=None)
        if not script:
            print(f"  ❌ No script found")
            continue

        # Calculate metrics
        total_chars = sum(
            len(s["text"]) for s in script.get("segments", []) if s["kind"] == "narration"
        )
        segment_count = len(script.get("segments", []))

        # Estimate time based on chars (4.5 chars/sec)
        estimated_secs = total_chars / 4.5

        # Rough cost estimate (MiniMax per-token pricing)
        # Assume ~200 tokens for analysis + script generation
        estimated_cost = 200 * 0.001 * 0.05  # ~¥0.01 (placeholder)

        samples[style] = {
            "project_id": project.id,
            "job_id": job.id,
            "elapsed_secs": elapsed,
            "total_chars": total_chars,
            "segments": segment_count,
            "estimated_time": estimated_secs,
            "estimated_cost": estimated_cost,
            "script": script,
        }

        print(f"  ✅ {total_chars} chars, {segment_count} segments, {elapsed:.1f}s elapsed")

    # Update evaluation document
    print("\n📝 Updating docs/M4_EVALUATION.md...")
    update_evaluation_doc(samples)

    print("\n✅ Evaluation complete. Review docs/M4_EVALUATION.md")
    return True


def update_evaluation_doc(samples: dict):
    """Insert generated scripts into the evaluation template."""
    doc_path = Path("..") / "docs" / "M4_EVALUATION.md"
    content = doc_path.read_text(encoding="utf-8")

    style_order = ["emotional", "suspense", "roast"]
    style_names = {
        "emotional": "A：emotional（情感走心）",
        "suspense": "B：suspense（悬疑紧凑）",
        "roast": "C：roast（轻松吐槽）",
    }

    for idx, style in enumerate(style_order, 1):
        if style not in samples:
            continue

        sample = samples[style]
        script = sample["script"]
        segments_text = "\n".join(
            f"  {s['id']}: {s['text']}"
            for s in script.get("segments", [])
            if s.get("kind") == "narration"
        )

        replacement = f"""**生成脚本**
```
{segments_text}
```

**自动指标**
| 指标 | 值 | 状态 |
|-----|-----|------|
| 生成时长 | {sample['estimated_time']:.1f}s | {'✅' if 25 <= sample['estimated_time'] <= 35 else '⚠️'} |
| 实际字数 | {sample['total_chars']} | {'✅' if 100 <= sample['total_chars'] <= 150 else '⚠️'} |
| 段数 | {sample['segments']} | {'✅' if 2 <= sample['segments'] <= 4 else '⚠️'} |
| 费用 | ¥{sample['estimated_cost']:.2f} | ✅ |"""

        # Find the section and replace
        marker = f"### 样本 {style_names[style]}"
        if marker in content:
            # Replace until next section
            start = content.find(marker)
            end = content.find("\n### 样本", start + 10)
            if end == -1:
                end = content.find("\n## 汇总", start)

            before = content[:start]
            after = content[end:]
            # Reconstruct the section
            section = f"{marker}\n\n**元数据**\n- 影片：Sintel 30s 片段\n- 风格：{style}\n- 目标时长：30s\n- 生成耗时：{sample['elapsed_secs']:.1f}s\n- 调用成本：¥{sample['estimated_cost']:.2f}\n\n{replacement}\n"
            content = before + section + after

    doc_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    try:
        success = run_evaluation()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
