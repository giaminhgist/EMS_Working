# Presentation figure suite — EMS-Project

Bộ figures phục vụ presentation với supervisor về *Learned Stimulus-Conditioned
Normative Modeling* trên dataset EMS. Mọi số liệu được **tính lại từ run
artifacts** (predictions.csv, checkpoints, summaries) và được kiểm chứng khớp
với các nguồn đã commit; không có số liệu nào bịa.

## Nội dung

```
presentation/
├── figures/                 # PNG 300 dpi + SVG theo nhóm
│   ├── 01_dataset/          # 2 figures (overview, gaze signatures)
│   ├── 02_features/         # 2 figures (45 features; effect sizes + rainclouds + categories)
│   ├── 03_methodology/      # 2 figures (architecture, protocols)
│   ├── 04_ablations/        # 6 figures — mỗi figure một câu hỏi ablation
│   ├── 05_latent_distribution/  # 1 figure (normative latent: PCA + heatmap + λ diagnostics)
│   └── 06_importance_xai/   # 1 figure (feature+stimulus importance)
├── scripts/                 # code tái lập (entry point: run_all.py)
├── tables/                  # CSV/JSON nguồn cho từng figure
├── cache/                   # intermediate tensors (fixations + latent exports)
├── figure_manifest.csv      # figure ID → nguồn, run/config, protocol, seed/fold, script
├── figure_index.md          # slide gợi ý, captions (EN), speaker notes (VI), giới hạn
├── figures_gallery.pdf      # contact sheet toàn bộ figures
└── missing_artifacts.md     # phần thiếu + lệnh bổ sung
```

## Chạy

```bash
cd /root/EMS-Project

# toàn bộ (lần đầu build cache fixations ~3 phút, latent exports ~1 phút)
.venv/bin/python presentation/scripts/run_all.py

# một nhóm
.venv/bin/python presentation/scripts/run_all.py --groups 04,05

# chạy trực tiếp một script
.venv/bin/python presentation/scripts/make_ablation_figs.py

# regenerate manifest + gallery sau khi đổi figures
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
- **Baselines P1 chỉ có seed 42** → không vẽ error bar như thể có nhiều seed;
  error bars chỉ phản ánh n thực.
- **Config matching**: run được chọn theo `config.json` khớp với
  `ABLATION_META` (không chọn theo mtime hay kết quả đẹp).
- **Checkpoint**: mọi inference dùng `best.pt` nguyên trạng, giữ nguyên bank
  buffers (không refresh), không retrain.
- **Verification**: script 04 (ablations) tái lập đúng các paired p-values
  trong `docs/analysis.md`; script 05 (latent) dùng nguyên trạng các exports
  bank/z từ checkpoint `best.pt` trong `cache/latent`.
- **XAI**: ranking tính trên fold Set_1, đánh giá trên fold Set_0 (partition
  riêng); permutation hoán đổi cả feature trajectory giữa subjects.
- **Không impute giả**: NaN → 0 trong standardized space có nghĩa "= HC norm",
  mask loại pair thiếu; không trình bày như quan sát sinh học thật.

## Nguồn dữ liệu chính

- Run artifacts: `outputs/proposal/{ablation}__seed{s}__fold{f}__*` (best.pt,
  predictions.csv, metrics.jsonl, allfolds_summary.json),
  `outputs/proposal/test_heldout/*`, `outputs/evaluation/*.csv`.
- Baselines: `docs/baseline/results/summary.csv` + per-run dirs
  (P1 val_preds.csv không có label/fold → join theo subject_id với metadata).
- Dữ liệu: `original_dataset/EMS/`, `processed_dataset/` (features, metadata,
  quality_report).
- Docs đối chiếu: `docs/analysis.md`, `docs/model_spec.md`,
  `experiment_tracker.md`, `docs/EDA/README.md`.
