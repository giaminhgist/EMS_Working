# Đặc tả kỹ thuật — Architecture · Tensor Shapes · Loss · Evaluations

Tài liệu này mô tả chi tiết mô hình chính (`src/proposal/model.py`), các
ablation của nó, loss, và ý nghĩa của từng evaluation. Mọi số liệu tham số
khớp với code hiện tại; mọi quy ước shape dùng `B` = batch size, `S` = số
stimuli (100), `D` = 45 features.

## 1. Tổng quan pipeline

```text
x_{i,s} ∈ R^45                              (45 hand-crafted eye-tracking features)
   │
   ├─ deviation = "learned" (MAIN) ─┐
   │   1. chuẩn hóa per-feature     │   deviation = "z"|"diff"|"mahal" (ABLATION)
   │   2. Encoder f_θ: 45→128→128   │   1. HC stats (μ_s, σ_s) per stimulus
   │   3. Latent bank (μ_s^z, σ_s^z)│   2. hard deviation cố định
   │   4. Comparator g_φ            │      z = (x−μ_s)/(σ_s+ε)
   │                                │      diff = x−μ_s
   │                                │      mahal = [z ‖ M_s]
   └──────────────┬─────────────────┘
                  ▼
        d_{i,s} ∈ R^64 (learned) | R^45/46 (hard)     × mask stimulus
                  ▼
        Stimulus pooling: attention | mean | deepset
                  ▼
        h_i ∈ R^64 (| R^128 deepset)
                  ▼
        Head: Linear→BN→ReLU→Dropout→Linear → logit
                  ▼
        p̂_i = sigmoid(logit) → HC/SZ tại threshold 0.5
```

Bảng configs (cờ CLI, khớp `experiment/matrix.json`):

| Config | deviation | comparator | pool | λ_norm | Vai trò |
|---|---|---|---|---|---|
| mlp_attn | learned | mlp | attention | 0 | **main proposal** |
| mlp_norm01 | learned | mlp | attention | **0.1** | proposal + chính quy hóa HC |
| sub_attn | learned | **sub** | attention | 0 | bỏ learned comparator |
| zsub_attn | learned | **zsub** | attention | 0 | bỏ comparator, giữ σ bank |
| mlp_mean | learned | mlp | **mean** | 0 | bỏ attention pooling |
| mlp_deepset | learned | mlp | **deepset** | 0 | pooling mean‖max |
| z_mean | **z** | — | mean | — | hard z-deviation |
| diff_mean | **diff** | — | mean | — | hard, không chuẩn hóa |
| mahal_mean | **mahal** | — | mean | — | hard + scalar Mahalanobis |

## 2. Dữ liệu & tiền xử lý (`src/data/tabular.py`, `proposal/model.py::NormativeDataset`)

- `subject_matrices(sids)` → `{sid: (X (S,45) float32, mask (S,) bool)}`; hàng
  thiếu stimulus = NaN, `mask` đánh dấu stimulus hợp lệ.
- **Input scaling (learned)**: `feature_norm_stats(train_ids)` — μ, σ per-feature
  từ **mọi subject của training fold** (unsupervised, không dùng nhãn):
  `X̃ = (X − μ)/σ` rồi `X̃ = X̃ * mask`.
- **HC normative stats (hard)**: `hc_normative_stats(train_ids)` — chỉ dùng
  **HC (id < 200) của training fold**:
  - μ_s (S,45), σ_s (S,45) per stimulus-feature (ε = 1e-6);
  - shrinkage: `σ_shrink = 0.9σ + 0.1·median_s(σ)` (Marquand-style) — dùng
    riêng cho scalar Mahalanobis.
- **Đảm bảo chống leakage**: mọi statistic chỉ fit trên training fold (HC-only
  nơi nêu rõ) rồi áp dụng lên val/test; official-test labels không bao giờ
  được dùng.
- Dataset contract (trainer dựa vào): `ds.subjects`, `ds.labels`,
  `ds.__getitem__(i) → ((X, mask), y, sid)`, `ds.collate(batch)`,
  `ds.hc_indices()`.
- `stim_subset` (cờ `--n_stim --stim_seed`): chọn ngẫu nhiên K stimuli cố định
  qua folds; với hard modes cắt cả (μ_s, σ_s), với learned mode cắt X và bank
  được đăng ký kích thước K.

## 3. Main model — `deviation="learned"` (201.602 tham số)

### 3.1 Encoder f_θ (22.656 tham số)

```python
encoder = Sequential(
    Linear(45, 128), LayerNorm(128), GELU(),
    Linear(128, 128))          # latent_dim = 128
```

Shapes: `(B, S, 45)` → view `(B·S, 45)` → `Linear+LN+GELU` → `(B·S, 128)`
→ `Linear` → `(B·S, 128)` → view `(B, S, 128)` = **z**.

### 3.2 HC latent normative bank (buffer, không train)

```python
register_buffer("bank_mu",    zeros(100, 128))
register_buffer("bank_sigma", ones (100, 128))
```

`refresh_bank(train_ds)` được trainer gọi **mỗi epoch trước khi train**:
forward encoder (eval mode, no-grad) trên toàn bộ HC của training fold, rồi
theo stimulus s:

```
μ_s^z = Σ_{i∈HC} z_{i,s}·mask / Σ mask        (S, 128)
σ_s^z = sqrt( Σ_{i∈HC} (z_{i,s}−μ_s^z)²·mask / Σ mask ) + ε
```

→ bank luôn bám theo encoder đang học, không cần gradient.

### 3.3 Comparator g_φ (172.480 tham số cho `mlp`)

```
feat = [z ‖ μ^z ‖ z−μ^z ‖ z⊙μ^z]      (B, S, 4·128 = 512)
mlp:  Linear(512, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 64)
      → d = comp(feat) ∈ (B, S, 64)
sub:  d = proj(z − μ^z),   proj = Linear(128, 64)
zsub: d = proj((z − μ^z)/(σ^z + ε))
d = d * mask[..., None]
```

### 3.4 Stimulus pooling

- `attention` (2.113 tham số): `score = Linear(64,32)→ReLU→Linear(32,1)`
  → `(B,S,1)`; cộng `(mask−1)·1e9` để stimulus không hợp lệ thành −∞;
  `a = softmax(score, dim=1)`; `h = Σ_s a·d·mask` → `(B, 64)`.
- `mean`: trung bình có mask → `(B, 64)`.
- `deepset`: `cat(mean, max)` → `(B, 128)` (max lấy trên stimulus hợp lệ,
  dùng −1e9 cho vị trí mask).

### 3.5 Head (4.353 tham số)

```python
head = Sequential(
    Linear(64 [·2 nếu deepset], 64), BatchNorm1d(64), ReLU(),
    Dropout(0.3),
    Linear(64, 1))
p̂ = sigmoid(head(h))        # (B, 1)
```

## 4. Hard-deviation ablations (4.642 / 4.738 tham số)

Dataset tự tính deviation cố định (`apply_deviation`):

```
z:    D = (X − μ_s)/(σ_s + ε)                              (S, 45)
diff: D = X − μ_s                                          (S, 45)
mahal:D = [z ‖ M_s],  M_s = sqrt(mean_k ((X−μ_s)/σ_shrink)²_k)   (S, 46)
D = D * mask
```

Model bỏ encoder/comparator/bank: `d = D` đi thẳng vào **cùng pooling + head**
như đường learned (`d_in = 45` hoặc `46` cho mahal) — khác biệt duy nhất với
learned path nằm ở cách tạo deviation, giữ ablation "sạch". `refresh_bank`
thành no-op. Tham số ~4,6k ≈ 2% của learned model — phần thắng của learned
không đến từ dung lượng mà từ biểu diễn học được.

## 5. Loss & huấn luyện

```
L = L_cls + λ_norm · L_norm
L_cls = BCE(p̂, y) = −[y log p̂ + (1−y) log(1−p̂)]          # nn.BCELoss
```

**L_norm** (chỉ khi `λ_norm > 0`, tính trong `_deviate`, graph-attached theo
batch):

```
sq_{i,s} = ‖z_{i,s} − μ_s^z‖²₂ · mask_{i,s}      (B, S)
L_norm = mean_{i: y_i=HC, s hợp lệ}( sq_{i,s} ) · λ_norm
```

→ kéo encoding của HC về tâm bank (dùng μ^z của đầu epoch). λ_norm = 0.1 là
giá trị đã chạy; batch nào không có HC thì không đóng góp.

**Huấn luyện** (`src/trainer/trainer.py`):

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW(lr=1e-3, weight_decay=1e-4) |
| Batch size | 16 subjects (mỗi sample = (100,45) + mask) |
| Epochs tối đa | 150 (matrix) |
| Early stopping | patience 30 trên **val AUC**, checkpoint = epoch AUC tốt nhất |
| Seed | `seed_all`: python/numpy/torch (+CUDA) mỗi fold |
| Metrics | Acc, AUC, BalAcc, Sen, Spe, F1 tại **threshold 0.5** |

Mỗi run ghi: `metrics.jsonl` (từng epoch), `best.pt`, `predictions.csv`,
`embeddings.npz` (h_i của val subjects), `summary.json`, `config.json`.

## 6. Ý nghĩa từng evaluation

### 6.1 Official 4-fold CV (Phase B — protocol đánh giá chính)

**Câu hỏi**: mô hình tốt đến đâu trên đúng protocol của dataset (Set_0..3,
40 subjects/fold, 120 train / 40 val mỗi fold), so với baseline suite.
**Cách đọc**: val AUC/Acc là trung bình 4 fold; chạy 5 seeds để có mean±std
và paired t-test giữa các ablation — đây là nguồn số liệu cho mọi kết luận
thành phần trong `docs/analysis.md`. Threshold 0.5 cố định để so trực tiếp
với `docs/baseline/results/`.

### 6.2 Held-out test 120/40 (Phase E)

**Câu hỏi**: mô hình có tổng quát hóa sang **subject chưa từng thấy** không
(trong 4-fold, các folds luân phiên val nhưng mọi subject đều từng được train
ở fold khác). Split stratified 120/40 với seeds 42/2024/2026, trong đó 120
lại chia 90 train / 30 inner-val (early stopping). **Cách đọc**: gap
val→held-out nhỏ = không overfit subject; đây là số so với LogReg-L2 0.9075,
SVM-RBF 0.9058, MSNet 0.8854.

### 6.3 Official test (Phase E)

**Câu hỏi**: dự đoán cho 48 subjects mà **nhãn bị tác giả giữ kín** — không
thể tính metric, chỉ xuất `Test_000..047 → prob` theo định dạng benchmark.
Train lại trên toàn bộ 160 labeled subjects (inner 75/25 early stopping).
**Cách đọc**: file `outputs/proposal/test_official/{ablation}/official_test_preds.csv`
gửi cho benchmark; là phép đo "sạch" nhất vì model không bao giờ thấy nhãn.

### 6.4 Probing — embedding separability (Phase F-1)

**Câu hỏi**: biểu diễn deviation có **tách biệt** HC/SZ không, độc lập với
head phân loại. Pool embedding out-of-fold của 4 fold (mỗi subject được embed
bởi model chưa từng thấy nó), rồi (a) **linear probe**: LogisticRegression
với StratifiedKFold nội bộ → probe AUC; (b) **silhouette score** theo nhãn.
**Cách đọc**: probe AUC cao = thông tin bệnh nằm trong biểu diễn, không phải
do head "giỏi"; mlp_norm01 0.9356 vs z_mean 0.8688 → latent learned deviation
mang thông tin tách biệt hơn hard z ~+0.07.

### 6.5 Calibration — ECE/Brier (Phase F-2)

**Câu hỏi**: xác suất dự đoán có "nói thật" không — khi model nói p̂=0.8 thì
tần suất SZ thật trong nhóm đó có ~80% không. **ECE** chia [0,1] thành 10
bin, mỗi bin tính |p̂̄ − ȳ| rồi trung bình trọng số; **Brier** = MSE(p̂, y).
**Cách đọc**: ECE 0.089 (mlp_norm01) = lệch trung bình ~9 điểm phần trăm —
tốt; 0.1515 (mlp_attn) = overconfident. Kèm temperature scaling (fit 1 hệ số
T: p_T = sigmoid(logit/T), out-of-fold): mlp_norm01 giảm ECE về 0.0647,
mlp_attn/mlp_mean không sửa được (T dao động qua folds).

### 6.6 Category probe (Phase F-3)

**Câu hỏi**: tín hiệu bệnh có nằm ở **từng stimulus đơn lẻ** không, và có tập
trung ở category nào (social / natural / synthetic / manipulated — EMS thiết
kế để stress SZ). Mỗi stimulus: logistic probe per-subject trên z-deviation
(của fold ngoài) → per-stimulus AUC, gộp theo category. **Cách đọc**:
raw 0.727 ≈ z 0.725 và các category ngang nhau (0.71–0.73) → tín hiệu không
nằm ở stimulus đơn lẻ mà ở **subject aggregation** — xác nhận thiết kế
"deviate per stimulus → set-pooling".

### 6.7 Cross-stimulus generalization (Phase F-4)

**Câu hỏi**: khi số stimuli giảm (100→50→25), model nào giữ được chất lượng —
deviation cố định hay learned? Train với subset stimuli cố định (stim_seed
42) trên cả 4 fold, so AUC drop 100→25. **Cách đọc**: hard z drop 0.0125 <
learned 0.0190 → z-deviation là đại lượng transferable hơn; learned cần nhiều
stimulus để xây bank/trọng số ổn định. Đồng thời kiểm chứng nguyên tắc thiết
kế "giữ trục stimulus đến pooling".

## 7. Tham chiếu cờ CLI

```
# common
--seed --fold(all|Set_0..3) --epochs --patience --lr --weight_decay --batch_size --dropout
# proposal
--ablation <tag>          # tên run, nằm trong run-dir
--deviation learned|z|diff|mahal
--comparator mlp|sub|zsub # chỉ dùng khi learned
--pool attention|mean|deepset
--lambda_norm 0.0|0.1
--n_stim K --stim_seed S  # subset stimuli (cross-stimulus eval)
```

Run-dir: `outputs/proposal/{ablation}__seed{s}__fold{f}__{ts}/` + file tổng
`{ablation}__seed{s}__allfolds_summary.json` (mean±std 4 fold).
