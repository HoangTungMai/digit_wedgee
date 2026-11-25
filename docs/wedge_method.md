# GelSight Wedge – tóm tắt phương pháp (Wang et al., ICRA 2021)

Tham chiếu đầy đủ: `docs/ICRA2021_Wedge.pdf`.

## Pipeline
1. Tiền xử lý: unwarp ảnh về hình chữ nhật → trừ ảnh nền (ref, không tiếp xúc) → ảnh diff.
2. Color-to-gradient: MLP 5-32-32-32-2, tanh. Input (R, G, B, X, Y), output (Gx, Gy).
   X, Y được thêm vào để bù suy giảm ánh sáng LED theo khoảng cách.
3. Fast Poisson solver 2D (DST), điều kiện biên = 0 → depth map.

## Calibration
- Ấn bi cầu bán kính đã biết lên gel; tìm tâm + bán kính vết (Hough + chỉnh tay).
- mm/px đo bằng thước kẹp ấn lên gel.
- Nhãn Gx, Gy tính giải tích từ mặt cầu cho từng pixel trong vết.
- Thêm pixel ngoài vết với nhãn 0, số lượng = 5% số pixel tiếp xúc.
- Paper dùng 40 ảnh: 32 train / 8 test.

## Kết quả tham chiếu (Table I, test set)
| Cấu hình đèn | Gx err | Gy err | θx (°) | θy (°) |
|---|---|---|---|---|
| RGB (3 đèn) | 0.043 | 0.042 | 2.25 | 2.19 |
| RG (2 vuông góc) | 0.045 | 0.044 | 2.33 | 2.29 |
| RB (2 đối diện), không NN | 0.106 | 0.045 | 5.71 | 2.35 |

Thứ tự tốt → kém: RGB > RG > RB > R. Thiếu đèn ở một trục thì gradient trục đó hỏng;
paper bù bằng U-Net train trên 10k depth map tổng hợp (suy Gx từ Gy).

## Lưu ý phần cứng từ paper
- Tắt AWB, khoá gain từng kênh và exposure (nếu không ảnh nhấp nháy, màu trôi).
- Gray filter dán lên cạnh acrylic: giảm phản xạ nội → tăng tương phản.
- Diffuser: tránh vùng tối ở mép làm mất gradient.
- Marker trên gel gây gợn nhỏ trong depth → nội suy gradient (nearest, ~10 ms).

## Mốc so sánh trên DIGIT
- 3D Cal (Kota et al., arXiv 2511.03078): CNN direct-depth, train bằng dữ liệu ấn tự động từ máy in 3D.
  Sai số depth trung bình trên vật thể chưa thấy: ~156 µm (DIGIT), ~205 µm (GelSight Mini).
  Phương pháp khác (cần nhiều dữ liệu hơn), nhưng là mốc tham khảo hợp lý cho cùng loại cảm biến.
