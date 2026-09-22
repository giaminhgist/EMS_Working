# Missing artifacts — presentation figure suite

Danh sách những gì chưa có / không thể có trong repo hiện tại, lý do và lệnh
bổ sung nếu cần. Các scripts figure hiện tại đã chạy hết trên artifacts sẵn có;
không có chỗ nào bịa dữ liệu để lấp chỗ trống.

## 1. SHAP / Integrated Gradients — chưa chạy

- **Lý do**: `shap` không có trong environment; permutation importance (đã làm,
  Figure_5) đủ trả lời câu hỏi feature-level và family-level mà không cần
  assumption về attributions đi qua từng tầng.
- **Lệnh bổ sung**: `uv add shap` rồi viết script SHAP cho head (ghi rõ
  background từ training data, target = logit).

## 2. Không có metadata lâm sàng

- **Thiếu**: tuổi, giới, PANSS/clinical scores của từng subject. Repo không
  commit các trường này; paper nói hai nhóm matched nhưng không có bảng số
  theo subject.
- **Ảnh hưởng**: Không vẽ/kiểm soát confound nhân khẩu (giới hạn của
  Figure_1/Figure_2); mọi kết luận HC/SZ là association, không causal.

## 3. UMAP / probing bổ sung — cố ý không làm

- Figure_4 dùng PCA (fit trên training-HC reference) — đủ cho câu hỏi
  "SZ có lệch khỏi HC norm trong latent space không". UMAP chỉ bổ sung nếu
  PCA không tách được; không cần.
- `probing.py`/`category_probe.py` hiện có đã được tổng hợp (probe AUC,
  silhouette, per-category AUC) — xem `outputs/evaluation/*.csv`; OOF
  embeddings ghép qua các model khác nhau không phải common space nên không
  vẽ chung PCA từ embeddings.npz.
