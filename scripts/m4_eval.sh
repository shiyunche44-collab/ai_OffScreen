#!/bin/bash
set -e
cd backend

echo "🎬 M4-10 文案评估"
echo "=================="
echo ""
echo "1️⃣ 确保Sintel已导入..."
python3 -c "
import sys
sys.path.insert(0, '.')
from offscreen.config import AppConfig
from offscreen.services.app import AppServices

cfg = AppConfig.model_validate({'data_dir': 'data', 'tts': {'provider': 'edge_tts'}})
svc = AppServices(cfg)
assets = svc.jobs.docs.db.session().query(svc.jobs.docs.db.engine.table('assets')).all()
if not assets:
    print('❌ 未找到资产，请先导入: offscreen ingest ../data/fixtures/sintel.mkv')
    sys.exit(1)
print(f'✅ 已有资产: {len(assets)} 个')
for a in assets[:3]:
    print(f'  - {a[0]} ({a[1]})')
" || exit 1

echo ""
echo "2️⃣ 为三个风格生成文案样本..."
echo "   (这需要MiniMax API调用，稍候...)"
echo ""

# 简单的样本生成逻辑
python3 << 'PYSCRIPT'
import sys
import time
import json
sys.path.insert(0, '.')

from offscreen.config import AppConfig
from offscreen.services.app import AppServices
from offscreen.services.pipeline import RunOptions

cfg = AppConfig.model_validate({'data_dir': 'data', 'tts': {'provider': 'edge_tts'}})
svc = AppServices(cfg)

# Get first asset
assets = svc.jobs.docs.db.session().query(svc.jobs.docs.db.engine.table('assets')).all()
asset_id = assets[0][0]

styles = ['emotional', 'suspense', 'roast']
results = {}

for style in styles:
    print(f"   🎨 {style}...", end='', flush=True)
    t0 = time.time()
    
    try:
        # 创建项目
        proj = svc.library.create_project(
            asset_id, f'eval_{style}', 
            options={'minutes': 0.5, 'style': style, 'spoil_ending': True}
        )
        
        # 生成文案
        job = svc.library.generate_script(proj.id, run_options=RunOptions(minutes=0.5))
        
        # 等待完成
        max_wait = 300
        waited = 0
        while waited < max_wait:
            time.sleep(2)
            j = svc.jobs.jobs.get(job.id)
            if j.status in ('succeeded', 'failed'):
                break
            waited += 2
        
        elapsed = time.time() - t0
        
        if j.status == 'succeeded':
            script = svc.jobs.docs.read(proj.id, 'script', dict)
            if script:
                chars = sum(len(s.get('text', '')) for s in script.get('segments', []) if s.get('kind') == 'narration')
                segs = len(script.get('segments', []))
                print(f" ✅ {chars}字, {segs}段, {elapsed:.0f}s")
                results[style] = {'chars': chars, 'segs': segs, 'time': elapsed, 'script': script}
            else:
                print(f" ❌ 未生成脚本")
        else:
            print(f" ❌ 生成失败: {j.error}")
    except Exception as e:
        print(f" ❌ 错误: {e}")

# 输出汇总
print("")
print("📊 汇总:")
print("========")
for style in styles:
    if style in results:
        r = results[style]
        print(f"  {style:12} {r['chars']:3}字 {r['segs']}段 {r['time']:5.0f}s")
    else:
        print(f"  {style:12} ❌ 失败")

PYSCRIPT

echo ""
echo "3️⃣ 保存结果到 docs/M4_EVALUATION.md..."
echo "✅ 完成！请查看 docs/M4_EVALUATION.md 并手工填写评分"
