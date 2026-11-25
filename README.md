# Wedge-style 3D reconstruction trên Meta DIGIT

Mục tiêu: tái tạo depth map từ ảnh DIGIT theo phương pháp GelSight Wedge, đo độ chính xác.
Pipeline: `diff RGB + (X,Y) → MLP 5-32-32-32-2 → (Gx,Gy) → Poisson (DST) → depth [mm]`.
Đọc trước: `docs/wedge_method.md` (tóm tắt, ~2 trang). Paper gốc: `docs/ICRA2021_Wedge.pdf`.

## 0. Cài đặt + chạy thử không cần phần cứng (≈15 phút)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
echo '{"width":320,"height":240,"mm_per_px":0.06,"corners":null}' > sensor_synth.json
python wedge.py --sensor sensor_synth.json synth --out data_synth --n 40 --ball-r 1.5 2.5
cp data_synth/gt_synth.json data_synth/labels.json      # synth có sẵn nhãn đúng
python wedge.py --sensor sensor_synth.json train --data data_synth --out synth.pt
python wedge.py --sensor sensor_synth.json eval  --data data_synth --model synth.pt
```
Mục đích: chắc môi trường chạy được và hiểu output trước khi đụng vào sensor thật.

## 1. Chuẩn bị DIGIT
- Tìm serial: `python -c "from digit_interface import DigitHandler; print(DigitHandler.list_digits())"`.
  Không dùng được digit-interface thì DIGIT vẫn là camera UVC: dùng `--cam N` thay cho `--digit SERIAL`.
- **Khoá auto-exposure / auto white balance** nếu camera cho phép (Linux: `v4l2-ctl -d /dev/videoX --list-ctrls`).
  Màu trôi giữa các frame sẽ bị MLP hiểu thành độ dốc.
- Bật DIGIT ~5 phút cho LED ổn định. Làm việc tránh đèn chiếu thẳng vào sensor.
- Gel không được hỏng, rách, bám bụi; lau nhẹ trước khi chụp.

## 2. Đo độ phân giải không gian (mm/px), BẮT BUỘC
Phải đo trên chính ảnh làm việc (cùng resolution với lúc chụp dữ liệu).

Cách khuyến nghị: ấn nhẹ một vật có khoảng cách đã biết lên gel, chụp, rồi click 2 điểm và nhập khoảng cách:
- **Header pin 2.54 mm** trên PCB: click tâm hai chân xa nhau nhất trong ảnh, nhập n × 2.54. Baseline dài → chính xác.
- **Thước kẹp**: đặt khe hàm ở giá trị biết trước (vd 4.00 / 6.00 / 8.00 mm), ấn 2 đầu hàm lên gel, click 2 mép trong.

```bash
# (lần đầu) điền width/height trong sensor_digit.json = kích thước frame thật (capture sẽ in ra), hoặc để null
python wedge.py --sensor sensor_digit.json measure --digit DXXXXX --write      # chụp từ camera
python wedge.py --sensor sensor_digit.json measure --img gauge.png --write      # hoặc từ ảnh có sẵn
```
Phím (khi chụp từ camera): cửa sổ chạy live → `space` = đứng ảnh (trung bình 5 frame) → click A, click B, gõ mm ở terminal
→ `space` = về live để đặt thước ở vị trí khác · `q` = kết thúc (tất cả phép đo trong phiên được gộp).
Yêu cầu: **≥ 3 phép đo theo trục ngang + ≥ 3 theo trục dọc**, ở cả giữa và gần mép vùng cảm biến.
Đạt khi độ lệch chuẩn < 2 % và trung bình trục x so với trục y lệch < 2 %. Không đạt → báo lại (có thể do méo ống kính, cần unwarp).
Kết quả lưu trong `measure_log.json`; `--write` ghi trung bình vào sensor json.

## 3. Thu dữ liệu ấn bi
Bi thép/bi vòng bi bất kỳ, **không cố định bán kính**: mỗi phiên khai báo bán kính bi đang dùng (đo bằng thước kẹp: đường kính / 2).
```bash
python wedge.py --sensor sensor_digit.json capture --digit DXXXXX --out data_digit --ball-r 1.5
# đổi bi: thoát (q), chạy lại với --ball-r mới, cùng --out
```
Phím: `r` = ref (trung bình 10 frame, **chụp đầu tiên**, không chạm) · `space` = ảnh ấn bi · `e` = ref cuối phiên · `q` = thoát.

Quy tắc chụp:
- Tổng **40–60 ảnh**, nên dùng **≥ 2 cỡ bi**; độ sâu ấn từ nông đến sâu.
- Rải đều **khắp** mặt gel (MLP học cả suy giảm ánh sáng theo vị trí X,Y).
- Mỗi ảnh đúng **một** vết; vết cách mép ảnh ≥ bán kính vết + 5 px.
- Giữ bi đứng yên lúc chụp (không mờ do rung).
- Chụp `e` (ref cuối) trước khi thoát mỗi phiên.

Kiểm tra trôi / lọt sáng ngay sau khi chụp:
```bash
python wedge.py --sensor sensor_digit.json check --data data_digit
```
Lệch nền > ~2 DN hoặc vùng thay đổi lớn bất thường → tìm nguyên nhân (đèn phòng, auto-exposure) và chụp lại. Chụp lại rẻ hơn sửa dữ liệu hỏng.

## 4. Label tay
```bash
python wedge.py --sensor sensor_digit.json label --data data_digit
```
- Chuột trái = đặt tâm · chuột phải = đặt bán kính qua điểm click (click lên mép vết).
- `w/a/s/d` dịch tâm 0.25 px, `m/n` tăng/giảm bán kính 0.25 px (chữ HOA = ×4). Ô zoom ×6 ở góc trái trên.
- `Enter` lưu · `x` bỏ ảnh · `u` hoàn tác ảnh trước · `Esc` thoát (đã lưu sau từng ảnh, chạy lại sẽ tiếp tục).
- Vòng tròn khớp **mép vùng đổi màu** của vết. Đây là bước quyết định chất lượng: lệch 1 px bán kính
  với vết nhỏ có thể làm depth đỉnh lệch hàng chục %. Không chắc thì bỏ ảnh.
- Bán kính bi lấy tự động từ `capture_log.json`; ảnh không có log thì thêm `--ball-r`.

Kiểm tra lại toàn bộ:
```bash
python wedge.py --sensor sensor_digit.json review --data data_digit
```
→ `labels_review.png` (từng vết + vòng label) và `labels_coverage.png` (độ phủ). Sửa: `label --redo`.

## 5. Train + đánh giá
```bash
python wedge.py --sensor sensor_digit.json train --data data_digit --out digit_mlp.pt
python wedge.py --sensor sensor_digit.json eval  --data data_digit --model digit_mlp.pt
```
- Train in ra `Gx_err, Gy_err, thx_deg, thy_deg` (so với Table I: RGB ≈ 0.043 / 2.2°).
- Eval: `rmse_contact_mm`, `peak_pred_mm` vs `peak_gt_mm`, `min_depth_mm` (âm = artifact Poisson). Lưu `results/eval.json`.
- Peak lệch có hệ thống (luôn cao hoặc luôn thấp) → nghi label bán kính hoặc mm/px.

## 6. Thử vật chưa thấy
```bash
python wedge.py --sensor sensor_digit.json recon --model digit_mlp.pt --ref data_digit/ref.png --img screw.png --out results/screw
python wedge.py --sensor sensor_digit.json live  --model digit_mlp.pt --digit DXXXXX
```
Tối thiểu 3 vật: ốc vít nhỏ, Lego, 1 vật có cạnh sắc (indenter vuông hoặc đồng xu).

## 7. Bàn giao
Điền `REPORT_TEMPLATE.md` → `results/summary.md`. Kèm: `sensor_digit.json`, `measure_log.json`,
`data_digit/` (đủ ảnh + `capture_log.json` + `labels.json` + 2 ảnh review), `digit_mlp.pt`, `results/`.

## Lệnh tham khảo
`python wedge.py -h` · `python wedge.py <lệnh> -h`. Không sửa `sphere_grad`, `poisson_dst`, `features`,
`build_samples`, `split_names` khi chưa trao đổi; giữ nguyên seed để kết quả các lần chạy so sánh được.
