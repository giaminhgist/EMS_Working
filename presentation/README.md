# Presentation figure suite — EMS-Project

Bộ figures phục vụ presentation với supervisor về *Learned Stimulus-Conditioned
Normative Modeling* trên dataset EMS. Mọi số liệu được **tính lại từ run
artifacts** (predictions.csv, checkpoints, summaries) và được kiểm chứng khớp
với các nguồn đã commit; không có số liệu nào bịa.

## Nội dung

```
presentation/
├── figure/                  # 5 figures — PNG 300 dpi + SVG, đặt tên Figure_1..Figure_5
│   ├── Figure_1.png/.svg    # dataset overview (subjects, stimuli, official folds)
│   ├── Figure_2.png/.svg    # gaze signatures: HC vs SZ distributions + scanpaths
│   ├── Figure_3.png/.svg    # effect sizes của 45 features + rainclouds + category profiles
│   ├── Figure_4.png/.svg    # normative latent: PCA + heatmap + λ diagnostics
│   └── Figure_5.png/.svg    # feature + stimulus importance
├── scripts/                 # code tái lập (entry point: run_all.py)
├── tables/                  # CSV/JSON nguồn cho từng figure
├── cache/                   # intermediate tensors (fixations + latent exports)
├── figure_manifest.csv      # figure → nguồn, run/config, protocol, seed/fold, script
├── figure_index.md          # slide gợi ý, captions (EN), speaker notes (VI), giới hạn
└── missing_artifacts.md     # phần thiếu + lệnh bổ sung
```

## Chạy

```bash
cd /root/EMS-Project

# toàn bộ (lần đầu build cache fixations ~3 phút, latent exports ~1 phút)
.venv/bin/python presentation/scripts/run_all.py

# một nhóm
.venv/bin/python presentation/scripts/run_all.py --groups 05

# chạy trực tiếp một script
.venv/bin/python presentation/scripts/make_latent_figs.py

# regenerate manifest + gallery (contact sheet) sau khi đổi figures
.venv/bin/python presentation/scripts/make_manifest_gallery.py
```

Cache được bỏ qua nếu đã tồn tại; `--force-cache` để rebuild fixations.

## Dependencies

Đúng môi trường của repo (`.venv`, `uv sync`): python ≥3.12, numpy, pandas,
scipy, scikit-learn, matplotlib, seaborn, torch (CPU đủ — model 201k params),
openpyxl (đọc xlsx fixation files). Không cần pyarrow/tqdm (đã tránh).

## Quy ước chống sai số (đọc trước khi sửa figure)

- **Đơn vị thống kê**: suy diễn HC/SZ luôn ở mức *subject* (n=80/80), không
  dùng hàng nghìn fixation làm quan sát độc lập.
- **P1**: mean qua 5 seeds của mean 4 folds; fold-mean ≠ pooled AUC (ROC dùng
  pooled, bảng số dùng fold-mean — đều ghi nhãn rõ).
- **Checkpoint**: mọi inference dùng `best.pt` nguyên trạng, giữ nguyên bank
  buffers (không refresh), không retrain.
- **Verification**: script latent (Figure_4) dùng nguyên trạng các exports
  bank/z từ checkpoint `best.pt` trong `cache/latent`.
- **XAI**: ranking tính trên fold Set_1, đánh giá trên fold Set_0 (partition
  riêng); permutation hoán đổi cả feature trajectory giữa subjects.
- **Không impute giả**: NaN → 0 trong standardized space có nghĩa "= HC norm",
  mask loại pair thiếu; không trình bày như quan sát sinh học thật.

## Nguồn dữ liệu chính

- Run artifacts: `outputs/proposal/{ablation}__seed{s}__fold{f}__*` (best.pt,
  predictions.csv, metrics.jsonl, allfolds_summary.json),
  `outputs/proposal/test_heldout/*`, `outputs/evaluation/*.csv`.
- Dữ liệu: `original_dataset/EMS/`, `processed_dataset/` (features, metadata,
  quality_report).
- Docs đối chiếu: `docs/analysis.md`, `docs/model_spec.md`,
  `experiment_tracker.md`, `docs/EDA/README.md`.
