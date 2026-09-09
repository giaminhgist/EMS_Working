# Missing artifacts — presentation figure suite

Danh sách những gì chưa có / không thể có trong repo hiện tại, lý do và lệnh
bổ sung nếu cần. Các scripts figure hiện tại đã chạy hết trên artifacts sẵn có;
không có chỗ nào bịa dữ liệu để lấp chỗ trống.

## 1. Official test — không có labels (không thể có metric)

- **Thiếu**: AUC/Acc/BalAcc cho 48 subject official test. Labels bị tác giả
  dataset giữ kín; repo chỉ có `outputs/proposal/test_official/mlp_norm01/
  official_test_preds.csv` (48 probabilities, định dạng benchmark).
- **Ảnh hưởng**: Không thể vẽ ROC/confusion cho official test; "unseen
  subjects" chỉ đo được qua split 120/40 tự tạo (real labels).
- **Lệnh bổ sung** (nếu benchmark trả điểm): thêm script đối chiếu điểm chính
  thức, hoặc submit `official_test_preds.csv` cho tác giả.

## 2. Baselines P1 — chỉ có seed 42

- **Thiếu**: predictions 4-fold của các baseline (SVM-RBF, LogReg, RF, ...)
  cho seeds 1234/2024/2026/7. `docs/baseline/results/P1/*/seed42/` là thứ duy
  nhất được commit.
- **Ảnh hưởng**: Không thể paired-test proposal vs baseline qua seeds; các số
  baseline trong `docs/baseline/04_results.md` chỉ có 1 seed cho P1.
- **Lệnh bổ sung**: `cd src/baseline && python run_all.py --protocols P1` với
  seeds bổ sung (cần sửa `SEED` trong `src/baseline/common.py` và chạy lại).

## 3. Held-out P2 — chỉ 4 config proposal

- **Thiếu**: `test_heldout` cho các ablation còn lại (sub_attn, zsub_attn,
  mlp_deepset, diff_mean, mahal_mean) và official-test predictions của các
  config khác ngoài mlp_norm01.
- **Ảnh hưởng**: `experiment_tracker.md` Phase E chỉ có 4 config; không đo
  được P2 cho mọi ablation.
- **Lệnh bổ sung**: `python src/test_model/eval.py --ablation <abl>
  [--deviation/--comparator/--pool/--lambda_norm ...] --mode heldout` cho từng
  config, rồi cập nhật `experiment_tracker.md`.

## 4. Cross-stimulus (xstim) — retrain-per-subset, seed 42 only

- **Thiếu**: inference masking từ checkpoint train100 trên subset 25/50 (tức
  thí nghiệm "train100 → test với ít stimuli hơn" đúng nghĩa). Hiện tại
  `src/evaluation/cross_stimulus.py` retrain với cùng subset K cho cả train
  và val, seed 42.
- **Ảnh hưởng**: F04.06 được đặt tên "stimulus-budget/subset sensitivity" và
  caption ghi rõ đây KHÔNG phải generalization tới unseen stimuli.
- **Lệnh bổ sung** (nếu muốn thêm panel "train100→mask tại test"): dùng
  `model_utils.forward_full` với mask subset trên checkpoint `mlp_attn`/
  `mlp_norm01` seed 42 (cần chọn subset stimuli theo đúng `stim_seed=42` của
  run xstim — subset đang được sinh bên trong `NormativeDataset`, cần export
  thêm nếu muốn tái lập đúng tập con).

## 5. SHAP / Integrated Gradients — chưa chạy

- **Lý do**: `shap` không có trong environment; permutation importance (đã làm,
  F06.01) đủ trả lời câu hỏi feature-level và family-level mà không cần
  assumption về attributions đi qua từng tầng.
- **Lệnh bổ sung**: `uv add shap` rồi viết script SHAP cho head (ghi rõ
  background từ training data, target = logit).

## 6. Không có metadata lâm sàng

- **Thiếu**: tuổi, giới, PANSS/clinical scores của từng subject. Repo không
  commit các trường này; paper nói hai nhóm matched nhưng không có bảng số
  theo subject.
- **Ảnh hưởng**: Không vẽ/kiểm soát confound nhân khẩu; mọi kết luận HC/SZ
  là association, không causal.

## 7. UMAP / probing bổ sung — cố ý không làm

- F05.01 dùng PCA (fit trên training-HC reference) — đủ cho câu hỏi
  "SZ có lệch khỏi HC norm trong latent space không". UMAP chỉ bổ sung nếu
  PCA không tách được; không cần.
- `probing.py`/`category_probe.py` hiện có đã được tổng hợp (probe AUC,
  silhouette, per-category AUC) — xem `outputs/evaluation/*.csv`; OOF
  embeddings ghép qua các model khác nhau không phải common space nên không
  vẽ chung PCA từ embeddings.npz.
