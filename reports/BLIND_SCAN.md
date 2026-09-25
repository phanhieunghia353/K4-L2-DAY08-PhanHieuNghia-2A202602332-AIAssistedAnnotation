# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 25 xe

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Các xe ở xa sát khúc cua phía trên (chỉ thấy hai chấm đèn pha nhỏ hoặc đèn hậu đỏ mờ, thân xe tối chìm vào nền đêm): AI rất dễ bỏ sót không nhận diện được xe (false negative).
2. Vệt đèn pha phản chiếu sáng rực trên mặt đường bê tông phía trước các xe đi ngược chiều ở làn trái (đặc biệt là chiếc xe to ở hàng đầu và xe sedan ở giữa làn): AI dễ vẽ bounding box quá rộng ôm trùm cả vệt ánh sáng rọi trên mặt đường thay vì chỉ ôm sát ranh giới thân xe.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
