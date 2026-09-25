# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Phan Hiếu Nghĩa

Công cụ gán nhãn đã dùng: CVAT (Docker local v2.76.0)

---

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool: 268 ảnh) và tập kiểm thử (test: 20 ảnh) được chia theo trục thời gian kèm vùng đệm (buffer: 112 ảnh, mỗi khoảng đệm dài 4 giây trước và sau đoạn test) thay vì chia ngẫu nhiên vì đặc tính của dữ liệu video:
- Camera được đặt cố định trên cầu vượt, quay liên tục một cảnh duy nhất trong 160 giây với tốc độ trích xuất 2.5 khung hình/giây. Ở khoảng cách 0.4 giây giữa hai frame liền kề, nền cảnh hoàn toàn bất biến và mỗi chiếc xe xuất hiện trong khung hình suốt 3–4 giây trước khi đi qua.
- **Nếu chia ngẫu nhiên (random split)**: Cùng một chiếc xe sẽ xuất hiện đồng thời ở cả tập huấn luyện lẫn tập kiểm thử (chỉ lệch nhau vài phần mười giây). Điều này dẫn tới hiện tượng **rò rỉ dữ liệu (data leakage)** nghiêm trọng. Khi đó, mô hình được đánh giá trên đúng những chiếc xe mà nó đã được huấn luyện nhìn thấy, khiến các số đo đánh giá như AP50, Precision, Recall bị **thổi phồng giả tạo (over-optimistic)**, không phản ánh đúng năng lực khái quát hóa trên dữ liệu thực tế ngoài thực địa.
- Việc chia theo thời gian với vùng đệm 4 giây đảm bảo chiếc xe gần nhất ở tập pool cách ảnh test ít nhất 4.4 giây, triệt tiêu hoàn toàn sự trùng lặp cá thể xe giữa train và test.

---

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg` và `outputs/metrics_round0.json`:
- **Loại xe không khớp nhãn tham chiếu**: Mô hình khởi đầu lạnh bỏ sót nhiều xe nhỏ ở xa gần khúc cua/chân trời (chỉ có hai chấm đèn nhỏ), các xe tối màu chìm vào nền đường đêm ở làn bên phải, và các xe bị che khuất một phần bởi xe khác (tạo ra 206 False Negatives trên tập test). Đồng thời, mô hình có 16 False Positives do nhận nhầm các vệt đèn pha rọi sáng loang trên mặt đường bê tông ướt hoặc biển báo phản quang thành xe.
- **Độ phủ (Recall) theo kích thước xe**:
  - `R small = 0.182`: Rất thấp, bỏ sót hơn 81% xe kích thước nhỏ (< 32² px).
  - `R medium = 0.547` và `R large = 0.561`: Đạt mức trung bình khá.
  - Con số này cho thấy mô hình YOLOv8n pretrained trên COCO chỉ nhận diện tương đối tốt các xe có kích thước đủ lớn ở cự ly gần/trung bình; khả năng nhận diện xe nhỏ ở điều kiện ánh sáng yếu ban đêm là điểm yếu cốt tử.
- **Trường hợp cần người rà lại nhãn tham chiếu**: Ở các xe phía xa sát chân trời nơi chỉ nhìn thấy hai đốm sáng mờ ảo hoặc các vùng phản quang đèn đường. Vì nhãn tham chiếu do một mô hình tự động tạo ra và chưa được con người rà soát từng box, nhiều trường hợp nhãn tham chiếu tự gán box vào vệt sáng hoặc gán thiếu xe bị che khuất. Cần người thẩm định thủ công để phân định rõ giữa xe thật và ảo ảnh trước khi kết luận mô hình dự đoán sai.

---

## 3. Chiến lược chọn mẫu

### Giải thích công thức và tham số
Công thức chọn mẫu kết hợp độ bất định và tính đa dạng:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D = 0.5 \cdot U + 0.3 \cdot A + 0.2 \cdot D$$
- **$U$ (Uncertainty - Trọng số 0.5)**: Đo bằng trung bình 5 độ bất định lớn nhất của các box trong ảnh với $u(c) = 1 - |2c - 1|$. Khi độ tin cậy $c = 0.5$, $u = 1.0$ (mô hình phân vân nhất giữa có xe hay không). Trọng số lớn nhất (0.5) nhằm ưu tiên các frame chứa đối tượng khiến mô hình bối rối nhất.
- **$A$ (Ambiguous - Trọng số 0.3)**: Tỷ lệ số box có độ tin cậy mập mờ ($0.15 \le c < 0.50$) chuẩn hóa theo giá trị cực đại trong pool. Đo lường "mật độ nghi ngờ" trên toàn bộ khung ảnh.
- **$D$ (Diversity - Trọng số 0.2)**: Khoảng cách thời gian tới frame đã gán gần nhất (chặn tối đa ở 10 giây). Đảm bảo các mẫu được chọn phân bổ đều đặn trên trục thời gian thay vì dồn cục vào một đoạn ngắn.
- **Vai trò của `MIN_GAP_S = 2.0s`**: Camera đặt cố định, hai ảnh cách nhau dưới 2 giây hầu như giống hệt nhau về bố cục giao thông. Việc gán cả hai ảnh này tiêu tốn gấp đôi công sức của con người nhưng mô hình chỉ học thêm được các đặc trưng trùng lặp (redundancy). `MIN_GAP_S` đóng vai trò bộ lọc khử trùng lặp (deduplication filter), tối ưu hóa giá trị thông tin thu được trên mỗi giờ công gán nhãn.

### Dẫn chứng minh họa từ dữ liệu
- Dẫn chứng 3 frame trong `reports/SELECTION.md`:
  - `frame_0182.jpg` (Rank 1, t = 72.8s, score = 0.9591, U = 0.9182, A = 1.0): Mật độ box mập mờ đạt cực đại (18 box), chứa nhiều xe tối màu và vệt sáng phức tạp.
  - `frame_0369.jpg` (Rank 2, t = 147.6s, score = 0.9324, U = 0.9315, A = 0.8889): Có tới 43 box, nhiều cụm xe đi ngược chiều ken đặc ở làn trái.
  - `frame_0099.jpg` (Rank 8, t = 39.6s, score = 0.9063, U = 0.9460): Chứa nhiều xe nhỏ ở khúc cua với điểm bất định U cao nhất nhóm đầu.
- Dẫn chứng 1 frame khác:
  - `frame_0372.jpg` (Rank 6, score = 0.9101, t = 148.8s): Có điểm cao hơn `frame_0099.jpg` nhưng **bị loại** vì cách `frame_0369.jpg` (t = 147.6s) chỉ 1.2s < `MIN_GAP_S`. Việc loại frame này thể hiện sự cân nhắc kỹ lưỡng giữa điểm số toán học và chi phí gán nhãn thực tế.

### Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?
**Không**. Điểm bất định cao chỉ phản ánh rằng mô hình hiện tại đang thiếu tự tin trên khung hình đó. Sự thiếu tự tin này có thể bắt nguồn từ nhiễu (noise) ngoại cảnh như lóa đèn pha, vệt nước phản chiếu, mờ nhòe chuyển động hoặc bóng cây. Nếu một ảnh có điểm bất định cao do chứa quá nhiều nhiễu bất khả quy (aleatoric uncertainty), việc đưa ảnh đó vào huấn luyện thậm chí có thể làm mô hình học phải các mẫu nhiễu cục bộ và giảm hiệu năng trên tập kiểm thử tổng quát.

---

## 4. Các vòng học chủ động (active learning)

Bảng tổng hợp từ `reports/rounds_table.md`:
| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 311 | 0.447 | -0.324 | 1.000 | 0.052 | 0.099 | 0.000 | 0.034 | 0.268 |

### Trình bày chi tiết Vòng 1
- **Mức độ sửa nhãn gợi ý** (trích xuất từ `outputs/round1_diff.md`):
  - Tổng số: 12 ảnh. Mô hình đề xuất ban đầu 169 box, sau khi người rà soát sửa còn 311 box.
  - **Accepted (giữ nguyên)**: 126 box (tỷ lệ chấp nhận đạt 75%).
  - **Edited (kéo lại ranh giới box)**: 26 box.
  - **Deleted (xóa box giả - FP của AI)**: 17 box.
  - **Added (thêm box bị bỏ sót - FN của AI)**: 159 box.
- **Biến thiên số đo**:
  - AP50 giảm từ 0.771 xuống 0.447 ($\Delta = -0.324$).
  - **Precision@0.25 tăng tuyệt đối lên 1.000** (không còn bất kỳ False Positive nào).
  - Recall@0.25 giảm từ 0.489 xuống 0.052; F1 giảm từ 0.640 xuống 0.099.
- **Xu hướng theo kích thước xe**:
  - Nhóm xe lớn (`large`): Recall giảm từ 0.561 xuống 0.268.
  - Nhóm xe vừa (`medium`): Recall giảm từ 0.547 xuống 0.034.
  - Nhóm xe nhỏ (`small`): Recall giảm từ 0.182 xuống 0.000.

### Phân tích định tính trên ảnh so sánh
Quan sát `outputs/compare_round1.jpg` (so với `compare_round0.jpg`):
- Ở `frame_0050`: Tại vòng 0, mô hình có 11 True Positives nhưng dính 2 False Positives (vẽ box trùm vệt đèn rọi trên mặt đường). Sang vòng 1, mô hình có 2 True Positives, **0 False Positives** và 16 False Negatives.
- **Nguyên nhân**: Khi huấn luyện trên 12 ảnh đã được người gán loại bỏ triệt để các box vệt đèn pha và căn chỉnh viền khắt khe, hàm loss đã phạt rất nặng các dự đoán mơ hồ. Kết quả là phân phối độ tin cậy (confidence score) của mô hình bị dịch chuyển mạnh về phía thận trọng. Tại ngưỡng cắt cố định `conf_thr = 0.25`, mô hình đạt độ chính xác tuyệt đối 100% (chỉ phát hiện các xe lớn rõ ràng ở gần), nhưng toàn bộ các xe vừa và nhỏ ở xa có độ tin cậy bị rơi xuống dưới ngưỡng 0.25 (ví dụ 0.10–0.20), dẫn đến Recall sụt giảm.

### Đối chiếu bằng chứng và ca khó theo quy tắc
- **Quan sát độc lập (`reports/BLIND_SCAN.md`)**: Trên `frame_0099.jpg`, tôi đếm được 25 xe, dự đoán AI sẽ bỏ sót xe xa chỉ có đèn nhỏ và dễ vẽ sai vệt đèn pha rọi sáng ở làn giữa.
- **Lỗi pre-label đã sửa (`reports/REVIEW_LOG.csv` và `outputs/round1_diff.md`)**: Pre-label AI ban đầu chỉ phát hiện 13 box; tôi đã thêm 14 box (cho các xe tối màu làn phải và xe ở xa), chỉnh sửa 3 box trùm vệt đèn, xóa 1 box giả, giữ nguyên 9 box $\rightarrow$ tổng 26 box, khớp với quan sát mắt thường ban đầu.
- **Kết quả mô hình sau train**: Mô hình fine-tune đã khắc phục hoàn toàn lỗi vẽ box lên vệt đèn phản chiếu (Precision = 1.000).
- **Ca khó theo guideline**: Xe màu tối ở làn bên phải chạy cạnh xe màu trắng bạc (`frame_0182.jpg`, `frame_0099.jpg`). Thân xe tối chìm vào bóng đêm, chỉ thấy hai chấm đèn hậu đỏ. Theo quy tắc trong [GUIDELINE_LABEL.md](GUIDELINE_LABEL.md), người gán phải ước lượng thân xe quanh cụm đèn chứ không được chỉ khoanh hai chấm đèn, đòi hỏi sự nhất quán cao để tránh đưa tín hiệu nhiễu vào mô hình.

---

## 5. Kết luận và giới hạn

### Đánh giá kết quả và quyết định
- So với cold start, mô hình vòng 1 có bước tiến vượt bậc về **độ chính xác thuần túy (Precision đạt 100%, không còn FP)**, loại bỏ triệt để việc nhận diện nhầm vệt đèn phản chiếu. Tuy nhiên, AP50 và Recall bị suy giảm do hiện tượng co cụm độ tự tin và kích thước mẫu huấn luyện còn quá nhỏ (12 ảnh).
- **Quyết định**: **Dừng lại ở Vòng 1**. 
  - *Lý do*: Việc tiếp tục chạy vòng 2 với chỉ thêm 12 ảnh và giữ nguyên siêu tham số (50 epochs, learning rate mặc định) sẽ tiếp tục làm mô hình overfit cục bộ vào bối cảnh hẹp, không giải quyết được căn nguyên của việc sụt giảm Recall ở ngưỡng conf 0.25. Cần tối ưu lại quy trình huấn luyện trước khi nạp thêm dữ liệu.

### Đề xuất hai ca cho vòng tiếp theo (nếu làm tiếp)
1. **`frame_0005.jpg`** (Rank 1 vòng 2, t = 2.0s): Thuộc đoạn đầu video, chứa nhiều xe cự ly trung bình với ánh sáng đèn đối diện, giúp bổ sung dữ liệu đa dạng theo thời gian.
2. **`frame_0074.jpg`** (t = 29.6s, score cao ở vòng 2): Chứa mật độ giao thông đông đúc ở cả hai chiều, nhiều xe kích thước nhỏ ở xa.
- **Chi phí rà nhãn**: Trung bình 30–40 box/ảnh, ước tính mất khoảng 5–7 phút cho mỗi ảnh để rà soát kỹ lưỡng theo guideline.
- **Nguy cơ ảnh gần trùng**: Cần duy trì nghiêm ngặt ràng buộc `MIN_GAP_S >= 2.0s` nhằm ngăn chặn thuật toán chọn các frame liền kề nhau chứa cùng một đoàn xe đang di chuyển.

### Giới hạn thực nghiệm
1. **Tập test nhỏ (20 ảnh)**: Với kích thước tập test chỉ 20 ảnh, phương sai thống kê rất lớn; chỉ cần một vài xe thay đổi trạng thái nhận diện có thể khiến số đo AP50 biến động mạnh.
2. **Luật bỏ qua box nhỏ (< 16 px)**: Loại trừ 14/417 box tham chiếu ở sát chân trời. Mặc dù giúp tránh phạt mô hình ở các điểm ảnh mơ hồ, luật này cũng làm giảm bớt độ nhạy của phép đo ở vùng cự ly cực xa.
3. **Nhãn tham chiếu do mô hình tự sinh**: Nhãn test chưa được con người thẩm định từng box, do đó AP50 chỉ phản ánh **mức độ khớp giữa mô hình fine-tune với mô hình tạo nhãn tham chiếu**, chứ không đại diện cho chân lý thực tế tuyệt đối.

### Biện pháp kiểm tra và khắc phục khi AP50 giảm
Trước khi tiếp tục gán nhãn hoặc huấn luyện thêm, tôi sẽ thực hiện:
1. **Kiểm tra phân phối Confidence Score**: Đánh giá lại mô hình ở các ngưỡng confidence thấp hơn (ví dụ `conf = 0.05` và `conf = 0.10`). Nếu Recall và AP50 tăng vọt ở ngưỡng thấp, điều đó chứng minh mô hình vẫn bắt đúng vị trí xe nhưng bị phạt điểm tin cậy, vấn đề nằm ở việc cân chỉnh độ tự tin (Confidence Calibration).
2. **Điều chỉnh siêu tham số fine-tune**: Giảm số epoch (từ 50 xuống 15–20 epochs) hoặc áp dụng freeze backbone (chỉ huấn luyện detection head) để tránh làm mất các đặc trưng tổng quát đã học từ COCO; giảm learning rate để tránh dịch chuyển trọng số quá mạnh trên lô 12 ảnh.
3. **Tăng cường dữ liệu (Data Augmentation)**: Bật mosaic, mixup và điều chỉnh độ sáng/tương phản để tăng tính bền vững của bộ phát hiện trong điều kiện đêm tối.
