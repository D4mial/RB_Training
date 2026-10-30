# RB_Training: robot mê cung, học heuristic từ dataset ảnh

Mỗi mê cung là **một ảnh PNG** trong `dataset/images/`. Mô hình học dự đoán khoảng cách thật
từ mỗi ô tới đích, dùng làm heuristic h(n) cho GBFS / A* và so sánh với Manhattan.

## Dataset
| Thư mục | Loại | Số ảnh |
|---|---|---|
| `dataset/images/K1` | Mê cung mở 20x20 (cố định) | 1 |
| `dataset/images/K2` | Bẫy chữ U 15x15 (cố định) | 1 |
| `dataset/images/K3_10`, `K3_20`, `K3_30` | Tường ngẫu nhiên 30x30, mật độ 10/20/30% | 1000 mỗi loại |
| `dataset/images/K4` | Mê cung hoàn hảo 31x31 | 1000 |

K1 và K2 là kịch bản cố định nên chỉ có 1 ảnh. Mã hóa màu: trắng = trống, đen = tường,
xanh lá = xuất phát, cam = đích; mỗi ô lưới là 10x10 điểm ảnh. Seed cố định (2026).

## Chạy
    pip install -r requirements.txt
    python train.py              # đọc ảnh -> huấn luyện -> đánh giá

Sinh lại dataset (hoặc đổi số lượng): `python build_dataset.py --n 1000`

## File
- `robot_me_cung.py`: bộ giải và bộ sinh mê cung gốc
- `build_dataset.py`: sinh và đọc ảnh mê cung
- `train.py`: đặc trưng, huấn luyện, đánh giá (chia train/test theo ảnh: 75% đầu / 25% cuối mỗi loại)
