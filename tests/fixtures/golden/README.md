# Golden baseline fixtures (Phase 0.3)

`auto_pipeline.py` Step 6 が生成する統合データセットと、その入力 CSV を保存しています。
リファクタリング後も **数値出力が一致するか** を検証するための基準データです。

## 構成

| パス | 内容 |
|------|------|
| `pipeline/lac_dataset_full.csv` | 主出力（129行 × 168列） |
| `pipeline/input_database_dataset.csv` | Step 6 入力（lac / 被験者属性） |
| `pipeline/REP_DATABASE.csv` | rep 単位の仕事量 DB |
| `pipeline/REP_POSTURE_DATABASE.csv` | rep 単位の姿勢 DB |
| `pipeline/TOTAL_WORK_DATABASE.csv` | セット単位の仕事量 DB |
| `processed_reps/*_rep.csv` | rep 区間定義（138 ファイル） |
| `manifest.json` | SHA256・行数・取得日時 |

## 再取得手順

```powershell
cd c:\Users\ironm\squat_analyze

# input_database_dataset が work_calculated に無い場合はコピー
Copy-Item poseestimate_mediapipe\out\estimate_blc_data\input_database_dataset.csv `
          poseestimate_mediapipe\out\work_calculated\input_database_dataset.csv

# Step 6 のみ実行（Steps 2–4 はスキップ）
$env:PYTHONIOENCODING = "utf-8"
python -m poseestimate_mediapipe.process.auto_pipeline --skip-until 4

# fixtures へコピー
python scripts/capture_golden_baseline.py
```

## テスト

```powershell
pytest tests/integration/test_golden_lac_dataset.py -q
```
