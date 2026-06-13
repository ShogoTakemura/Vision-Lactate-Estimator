# Vision-Lactate-Estimator 🏋️‍♂️

## 使用技術一覧

![Python](https://img.shields.io/badge/Python-3776AB.svg?logo=python&style=for-the-badge&logoColor=white)
![Intel](https://img.shields.io/badge/Intel_RealSense-0071C5.svg?style=for-the-badge&logo=intel&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0769AD.svg?logo=google&style=for-the-badge&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&style=for-the-badge&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?logo=numpy&style=for-the-badge&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&style=for-the-badge&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&style=for-the-badge&logoColor=white)

## 目次

1. [概要](#概要)
2. [主な機能](#主な機能)
3. [ディレクトリ構成](#ディレクトリ構成)
4. [環境](#環境)
5. [開発環境構築](#開発環境構築)
6. [使用方法](#使用方法)

---

## 概要

Intel RealSense で計測した `.bag` ファイルから、スクワット動作の **姿勢推定・心拍推定・生体力学解析** を行い、**血中乳酸値推定**のためのデータセットを構築するパイプラインです。

カメラ映像だけを使ったノンコンタクト計測を実現しており、以下を一括処理します。

1. RealSense `.bag` → MediaPipe による 3D 姿勢推定
2. 顔 RGB 信号 → rPPG（非接触心拍推定）
3. 身体重心（CoM）軌跡・床反力・関節角度の算出
4. レップ（反復回数）ごとの姿勢品質評価
5. 仕事量・生理指標・姿勢特徴を統合した乳酸推定データセット生成

---

## 主な機能

| 機能 | モジュール |
| --- | --- |
| `.bag` ファイル再生・姿勢推定 | `poseestimate_mediapipe` |
| モデルベース姿勢補正 | `process/modelbasecorrect.py` |
| 身体重心・床反力・COP 計算 | `process/calccomprocess.py`, `squat_core/kinematics.py` |
| 関節角度・姿勢解析 | `process/pose_analyze_process.py` |
| rPPG 心拍推定（全セッション） | `frame_viewer_tool/hr_estimation.py` |
| rPPG 心拍推定（レップごと） | `frame_viewer_tool/hr_rep_analysis.py` |
| スクワット仕事量計算 | `process/calculate_work_process.py` |
| 乳酸推定データセット構築 | `process/build_lac_dataset.py` |
| フルパイプライン自動実行 | `process/auto_pipeline.py` |

---

## ディレクトリ構成

```text
squat_analyze/
├── poseestimate_mediapipe/        # メイン解析パッケージ
│   ├── __main__.py                # インタラクティブメニュー
│   ├── config/
│   │   ├── constants.py           # settings.toml ラッパー（GRAVITY, FPS 等）
│   │   ├── config.ini             # MediaPipe / RealSense 設定
│   │   └── *.csv                  # WORKSET, SUBJECTS_DATA 等
│   ├── process/                   # 各処理ステップ
│   │   ├── auto_pipeline.py       # フルパイプライン自動実行
│   │   ├── calccomprocess.py      # CoM・床反力・COP
│   │   ├── modelbasecorrect.py    # モデルベース補正
│   │   ├── pose_analyze_process.py
│   │   ├── calculate_work_process.py
│   │   └── build_lac_dataset.py
│   └── module/                    # 共通モジュール群
│       ├── com/                   # CoM 計算
│       ├── fp/                    # フォースプレート
│       ├── modelbased/            # モデルベース補正
│       └── movement.py, pose3d.py 等
│
├── squat_core/                    # 再利用可能な演算ライブラリ
│   ├── signal.py                  # rPPG 信号処理（POS, BPF, Welch, RRI）
│   └── kinematics.py              # 運動学（速度・加速度・床反力・荷重分配）
│
├── frame_viewer_tool/             # 心拍解析 CLI ツール
│   ├── hr_estimation.py           # 単一セッション心拍推定
│   └── hr_rep_analysis.py         # レップ別心拍推定
│
├── analyze_sensor/                # フォースプレート相関解析
├── scripts/                       # ユーティリティスクリプト
├── tests/                         # テストスイート
│   ├── unit/                      # 単体テスト
│   └── integration/               # 統合テスト（ゴールデンデータセット検証）
├── data/                          # 入力データ
├── results/                       # 出力結果
├── settings.toml                  # 物理定数・信号処理パラメータ設定
└── pyproject.toml
```

---

## 環境

| 項目 | バージョン |
| --- | --- |
| Python | 3.12 以上 |
| OS | Windows 10/11（RealSense SDK が必要） |
| ハードウェア | Intel RealSense D シリーズ（録画済み `.bag` ファイルがあれば不要） |

主要ライブラリ：

- `mediapipe` — 姿勢推定
- `pyrealsense2` — RealSense SDK（Windows/Linux のみ）
- `opencv-python` — 映像処理
- `scipy`, `numpy` — 信号処理・数値計算
- `pandas`, `matplotlib` — データ分析・可視化
- `scikit-learn` — 機械学習
- `questionary` — インタラクティブ CLI メニュー

---

## 開発環境構築

```bash
# 1. リポジトリをクローン
git clone <repository-url>
cd squat_analyze

# 2. 仮想環境の作成・有効化
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

# 3. パッケージのインストール
pip install -e ".[dev]"

# 4. テストの実行（動作確認）
pytest tests/ -q
```

> **Note**: `pyrealsense2` は Windows/Linux 向けです。macOS では `.bag` 再生機能は使えませんが、その他の解析機能は動作します。

---

## 使用方法

### インタラクティブメニュー（推奨）

```bash
python -m poseestimate_mediapipe
```

メニューから処理を選択します（14 種類）。

```text
? 処理を選択してください:
  > 1. 単一 bag ファイル 姿勢推定
    2. 複数 bag ファイル 姿勢推定
    3. モデルベース補正
    4. 身体重心（CoM）計算
    5. 姿勢角度解析
    ...
```

---

### フルパイプライン自動実行

```bash
python -m poseestimate_mediapipe.process.auto_pipeline

# 途中のステップから再開する場合
python -m poseestimate_mediapipe.process.auto_pipeline --skip-until 4
```

---

### 心拍推定（単一セッション）

```bash
python frame_viewer_tool/hr_estimation.py <csv_path> [オプション]

# 例
python frame_viewer_tool/hr_estimation.py data/rgb_20250121.csv \
    --rep_csv data/reps_20250121.csv \
    --save_dir results/hr/ \
    --trim_start 5 --trim_end 5 \
    --save_fig
```

| オプション | 説明 | デフォルト |
| --- | --- | --- |
| `csv_path` | 顔 RGB CSV ファイルのパス（必須） | — |
| `--rep_csv` | レップ境界 CSV | なし |
| `--save_dir` | 出力ディレクトリ | なし |
| `--trim_start` | 先頭からトリムする秒数 | 0 |
| `--trim_end` | 末尾からトリムする秒数 | 0 |
| `--no_plot` | グラフ表示を抑制 | False |
| `--save_fig` | グラフを画像として保存 | False |

---

### 心拍推定（レップ別）

```bash
python frame_viewer_tool/hr_rep_analysis.py [オプション]

# 例
python frame_viewer_tool/hr_rep_analysis.py \
    --input_dir frame_viewer_tool/roi_out/20241119 \
    --out_dir frame_viewer_tool/reps/processed \
    --prefix 20250121_subject1- \
    --n_sets 3
```

| オプション | 説明 | デフォルト |
| --- | --- | --- |
| `--input_dir` | RGB CSV ファイルのあるディレクトリ | — |
| `--out_dir` | 出力ディレクトリ | — |
| `--prefix` | ファイル名プレフィックス | — |
| `--n_sets` | セット数 | 3 |
| `--no_plot` | グラフ表示を抑制 | False |

---

### 設定ファイル

物理定数や信号処理パラメータは `settings.toml` で管理します。

```toml
[physics]
gravity = 9.80665   # 重力加速度 [m/s²]
fps     = 30.0      # フレームレート [Hz]

[signal]
hr_bpf_low      = 0.8   # 心拍帯域通過フィルタ下限 [Hz]
hr_bpf_high     = 2.0   # 心拍帯域通過フィルタ上限 [Hz]
hr_pos_window_s = 1.6   # POS 法ウィンドウ幅 [s]
```

---

### テスト

```bash
# 全テスト
pytest tests/ -q

# 単体テストのみ
pytest tests/unit/ -q

# 統合テスト（ゴールデンデータセット検証）
pytest tests/integration/ -q
```
