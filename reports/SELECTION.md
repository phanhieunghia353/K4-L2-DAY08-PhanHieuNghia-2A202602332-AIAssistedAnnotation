# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét ảnh gần trùng hoặc trường hợp model không dự đoán được box:
1. **`frame_0182.jpg`** (Thứ tự: Rank 1, Điểm: 0.9591, Thời điểm: 72.8s): Ảnh có điểm số cao nhất toàn bộ tập pool; độ bất định của 5 box khó nhất rất cao (U = 0.9182) và số lượng box mập mờ đạt tối đa (A = 1.0, 18 box trong dải conf 0.15–0.50). Khung cảnh có mật độ xe dày ở cả hai chiều di chuyển, nhiều xe tối màu và vệt đèn pha phản chiếu trên mặt đường ướt.
2. **`frame_0369.jpg`** (Thứ tự: Rank 2, Điểm: 0.9324, Thời điểm: 147.6s): Mật độ xe lớn (43 box), độ bất định cao (U = 0.9315, A = 0.8889), nằm ở cuối trục thời gian của video, giúp mô hình bao quát phân phối giao thông ở các thời điểm khác nhau.
3. **`frame_0326.jpg`** (Thứ tự: Rank 4, Điểm: 0.9155, Thời điểm: 130.4s): Điểm bất định cao (U = 0.9310), có 39 box và 15 box mập mờ; cách xa các frame trên hơn 17 giây đảm bảo đa dạng ngữ cảnh giao thông.
4. **`frame_0099.jpg`** (Thứ tự: Rank 8, Điểm: 0.9063, Thời điểm: 39.6s): Điểm U cao vượt trội (0.9460), nằm ở đoạn đầu video (giây 39.6) giúp bổ sung dữ liệu đa dạng theo thời gian cho các frame ở nửa đầu video vốn ít được chọn trong top 5 đầu.
5. **`frame_0227.jpg`** (Thứ tự: Rank 11, Điểm: 0.8915, Thời điểm: 90.8s): Được ưu tiên thay vì các frame có rank cao hơn như `frame_0331.jpg` (Rank 5, t = 132.4s) hay `frame_0372.jpg` (Rank 6, t = 148.8s) vì hai frame đó gần trùng về mặt thời gian (chỉ cách `frame_0326.jpg` 2.0s và cách `frame_0369.jpg` 1.2s). `frame_0227.jpg` ở giây 90.8 cách xa các frame đã chọn hơn 16 giây, giúp tối ưu hóa giá trị thông tin trên mỗi ảnh được gán nhãn với ngân sách hạn hẹp.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
- **`frame_0182.jpg`** (Rank 1, CSV: score = 0.9591, U = 0.9182, A = 1.0, 18 box mơ hồ; contact sheet hàng 1, ô 3): Nhiều xe bị chìm vào vùng tối ở làn phải và các vệt sáng phản chiếu gây nhầm lẫn cao.
- **`frame_0369.jpg`** (Rank 2, CSV: score = 0.9324, U = 0.9315, A = 0.8889, 43 box; contact sheet hàng 3, ô 2): Cụm xe đi ngược chiều dồn cục ở làn trái, nhiều vệt đèn pha rọi sáng chéo nhau.
- **`frame_0099.jpg`** (Rank 8, CSV: score = 0.9063, U = 0.9460, A = 0.7778, 29 box; contact sheet hàng 1, ô 1): Thể hiện rõ các xe nhỏ ở khúc cua phía xa chỉ có đèn li ti và vệt đèn pha kéo dài ở làn giữa.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- **`frame_0372.jpg`** (Rank 6, Điểm = 0.9101, t = 148.8s): Có điểm số rất cao thuộc top 6 toàn pool, nhưng **bị loại không chọn vào lô** bởi thuật toán chọn tham lam có ràng buộc `MIN_GAP_S = 2.0s`. Do frame này chỉ cách `frame_0369.jpg` (Rank 2, t = 147.6s) đúng 1.2 giây, camera cố định khiến hai ảnh gần như trùng lặp nhau về vị trí các xe. Việc gán cả hai frame này sẽ gây lãng phí ngân sách gán nhãn mà mô hình không học thêm được nhiều đặc trưng mới.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Phép chọn mẫu theo độ bất định (Uncertainty Sampling) chỉ phản ánh mức độ phân vân của mô hình hiện tại trên các dự đoán ở tập pool (các box có confidence xấp xỉ 0.5 hoặc nhiều box mơ hồ). Nó **chưa chứng minh được rằng sau khi gán nhãn và huấn luyện trên các ảnh này thì hiệu năng của mô hình trên tập kiểm thử (test set) sẽ chắc chắn tăng**. Độ bất định cao có thể xuất phát từ nhiễu hình ảnh (lóa sáng, phản chiếu mặt đường, nhòe chuyển động) thay vì các đặc trưng hữu ích cho việc phát hiện xe. Ngoài ra, việc cải thiện còn phụ thuộc vào kích thước tập dữ liệu huấn luyện, siêu tham số fine-tune và mức độ tương thích với nhãn tham chiếu của tập test.
