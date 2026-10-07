# M4-10 文案评估集（快速启动）

## 目标
验证三个风格预设（emotional、suspense、roast）在真实影片上的生成质量。

## 一键运行评估

### 前置条件
- MiniMax API Key 已设置（`MINIMAX_API_KEY`）
- Sintel 影片在 `data/fixtures/sintel.mkv`

### 步骤

```bash
# 1. 导入影片并分析（~3-5 分钟，含 ASR/场景/故事分析）
cd backend && uv run offscreen run-all ../data/fixtures/sintel.mkv --minutes 1

# 2. 为三个风格生成文案
uv run python3 << 'SCRIPT'
import sys, time, json
sys.path.insert(0, '.')
from pathlib import Path
from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.pipeline import RunOptions

cfg = AppConfig.model_validate({'data_dir': 'data', 'tts': {'provider': 'edge_tts'}})
svc = AppServices(cfg)

# Find asset
with svc.jobs.docs.db.session() as s:
    assets = s.query(svc.jobs.docs.db.db.table('assets')).all()
    
if not assets:
    print("❌ No asset found")
    sys.exit(1)

asset_id = assets[0][0]
print(f"📽️ Asset: {asset_id}\n")

styles = ['emotional', 'suspense', 'roast']
samples = {}

for style in styles:
    print(f"🎨 {style}...", end='', flush=True)
    t0 = time.time()
    
    proj = svc.library.create_project(
        asset_id, f'eval_{style}',
        options={'minutes': 1, 'style': style, 'spoil_ending': True}
    )
    
    job = svc.library.generate_script(proj.id, run_options=RunOptions(minutes=1))
    
    # Wait for completion
    while True:
        time.sleep(2)
        j = svc.jobs.jobs.get(job.id)
        if j.status in ('succeeded', 'failed'):
            break
        print(".", end='', flush=True)
    
    if j.status == 'succeeded':
        script = svc.jobs.docs.read(proj.id, 'script', dict)
        chars = sum(len(s.get('text', '')) for s in script.get('segments', []) if s.get('kind') == 'narration')
        segs = len(script.get('segments', []))
        print(f" ✅ {chars}字 {segs}段 {time.time() - t0:.0f}s")
        samples[style] = {'chars': chars, 'segs': segs, 'proj': proj.id, 'script': script}
    else:
        print(f" ❌ {j.error}")

# Save results
with open('../eval_samples.json', 'w') as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 保存到 eval_samples.json")
SCRIPT

# 3. 查看生成结果
echo ""
echo "📊 查看结果："
cat eval_samples.json | python3 -m json.tool | head -50
```

## 手工评分

1. 打开 `docs/M4_EVALUATION.md`
2. 从 `eval_samples.json` 复制脚本文本到各风格的"**生成脚本**"部分
3. 按照"**人工评分**"标准填写 1-5 分（四个维度）
4. 在"**发现与建议**"部分总结

## 评分表字段说明

| 维度 | 是什么 | 打分标准 |
|-----|------|--------|
| 风格一致性 | 生成的词汇、节奏、句式是否与风格预设相符 | 5=完全符合，1=不符合 |
| 可用性 | 有无语法错误、逻辑错误、不适合配音的表达 | 5=无错可直接配音，1=无法使用 |
| 信息准确性 | 对影片内容的理解是否准确，细节引用是否恰当 | 5=准确理解，1=严重偏离 |
| 综合质量 | 作为解说词的整体效果和吸引力 | 5=很好，1=很差 |

## 输出

评分完成后，`docs/M4_EVALUATION.md` 会形成：
- ✅ 三个风格的脚本样本
- ✅ 自动指标（时长、字数、段数、费用）
- ✅ 人工评分（4 个维度 × 3 个风格 = 12 个分数）
- ✅ 对比分析（哪个风格最强，哪个需要改进）

## 估时
- 脚本生成：~10-15 分钟（含 API 调用）
- 人工评分：~20-30 分钟（三个风格逐一评审）
- **总计**：~45 分钟（对标 M4-10 的 0.5 人天）

## 常见问题

**Q：生成失败怎么办？**  
A：检查 `MINIMAX_API_KEY` 是否有效，余额是否充足。

**Q：能用测试影片吗？**  
A：可以，用 `data/fixtures/sintel.mkv` 的前 30-60 秒即可。

**Q：评分的时候是否需要播放成片？**  
A：建议播放，这样能更好地判断文案与画面的贴切度。构建计划步骤后会自动生成成片。

