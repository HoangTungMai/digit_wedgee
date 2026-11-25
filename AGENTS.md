# Bối cảnh (đọc trước khi làm)

Dự án: tái tạo depth map cho cảm biến xúc giác Meta DIGIT theo phương pháp GelSight Wedge
(MLP màu→gradient + Poisson). Hướng dẫn đầy đủ trong README.md; phương pháp trong docs/wedge_method.md.

## File
- `wedge.py`: toàn bộ pipeline (synth/capture/measure/check/label/review/train/eval/recon/live/export).
- `sensor_digit.json`: cấu hình sensor; `mm_per_px` do người đo bằng lệnh `measure`.
- `data_digit/`: dữ liệu gốc (ref.png, ref_end.png, ball_*.png, capture_log.json, labels.json). KHÔNG sửa/xoá ảnh gốc.
- `docs/`: tóm tắt phương pháp + paper.

## Quy tắc
- Các bước có GUI (capture, measure, label) do NGƯỜI làm; agent không tự tạo hay sửa labels.json
  ngoài yêu cầu rõ ràng. Không bao giờ tự đoán mm/px hay bán kính bi.
- Không sửa hàm lõi (`sphere_grad`, `poisson_dst`, `features`, `build_samples`, `split_names`) khi chưa hỏi.
  Nghi có bug → báo kèm bằng chứng (số liệu, ảnh).
- Giữ nguyên seed và split train/test.
- Lỗi do phiên bản thư viện: sửa tối thiểu, ghi lại đã sửa gì.
- Không thay torch bằng framework khác; không tải dữ liệu hay model từ ngoài.
- Script mới đặt ở thư mục gốc; kết quả ở `results/`.
- Khi yêu cầu có CHECKPOINT: dừng hẳn và báo cáo.
