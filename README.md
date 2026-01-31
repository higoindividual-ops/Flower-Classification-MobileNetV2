# 🌸 Flower Classification System: MobileNetV2 & Smart Padding

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)
![Philosophy](https://img.shields.io/badge/Philosophy-AI_Assisted_Dev-blueviolet?style=for-the-badge)

> **"Product-Oriented Engineering: Focus on Architecture, Efficiency, and User Experience."**

## 📖 Introduction (Giới thiệu)

Dự án xây dựng hệ thống phân loại 6 loài hoa phổ biến sử dụng kiến trúc **MobileNetV2** (Transfer Learning). 

Điểm khác biệt của dự án này không nằm ở việc xây dựng một model khổng lồ, mà nằm ở **Tư duy Giải quyết Vấn đề (Problem Solving)**: Làm thế nào để model hoạt động chính xác trên dữ liệu thực tế (ảnh méo, ảnh kém chất lượng) và tối ưu hóa quy trình huấn luyện trên môi trường giới hạn (Google Colab).

## 🤖 Development Philosophy (Triết lý phát triển)

Dự án này được thực hiện theo quy trình **AI-Assisted Development** (Phát triển với sự hỗ trợ của AI).
* **Tôi không tập trung vào việc gõ thủ công từng dòng code.**
* **Tôi tập trung vào việc:**
    1.  **Kiến trúc hệ thống:** Lựa chọn MobileNetV2 vì sự cân bằng giữa tốc độ và độ chính xác.
    2.  **Quản lý công nghệ:** Phối hợp các công cụ (TensorFlow, OpenCV, Gdown) để tạo ra pipeline mượt mà.
    3.  **Xử lý điểm mù (Edge Cases):** Phát hiện và xử lý vấn đề ảnh đầu vào bị méo tỷ lệ (Aspect Ratio Distortion).

## 💡 Key Features & Problem Solving (Điểm nhấn Kỹ thuật)

### 1. 🖼️ Smart Padding Strategy (Giải quyết vấn đề méo ảnh)
* **Vấn đề (The Pain Point):** Các model CNN yêu cầu input vuông (224x224). Phương pháp `resize` thông thường sẽ "ép" ảnh chữ nhật thành hình vuông, làm biến dạng đặc trưng của hoa (ví dụ: hoa hướng dương bị bóp dẹt).
* **Giải pháp của tôi:** Viết thuật toán **Smart Padding**:
    * Tính toán tỷ lệ khung hình gốc.
    * Resize giữ nguyên tỷ lệ (Aspect Ratio Preservation).
    * Tự động thêm viền đen (Padding) vào phần thừa.
* **Kết quả:** Model nhận diện tốt cả những ảnh Panorama cực đoan (Tỷ lệ 5:1, kích thước 2576x517).

### 2. ⚡ I/O Optimization Pipeline (Tối ưu hóa hiệu năng)
* **Vấn đề:** Huấn luyện trên Google Colab thường bị nghẽn cổ chai (Bottleneck) khi đọc hàng nghìn file ảnh nhỏ trực tiếp từ Google Drive.
* **Giải pháp:**
    * Sử dụng cơ chế **Cloud-to-Local**: Tự động tải file nén `.rar` từ Cloud về ổ cứng SSD cục bộ của Colab.
    * Giải nén tại chỗ (Local Unzip) để loại bỏ độ trễ mạng.
    * Sử dụng `tf.data.Dataset` với `.cache()` và `.prefetch()` để nạp dữ liệu song song vào GPU.
* **Hiệu quả:** Tăng tốc độ huấn luyện gấp **~40 lần** so với đọc trực tiếp từ Drive.

### 3. ☁️ "One-Click" Deployment
* Code được thiết kế để **bất kỳ ai cũng chạy được ngay** mà không cần cấu hình phức tạp.
* Tự động tải Model và Data thông qua `gdown`, loại bỏ bước Mount Drive thủ công phiền phức.

## 📊 Results (Kết quả thực nghiệm)

### 1. Training Performance (Hiệu suất huấn luyện)
Biểu đồ dưới đây cho thấy quá trình hội tụ của mô hình qua 10 epochs.

<img width="864" height="396" alt="Biểu đồ Độ chính xác   Hàm mất mát" src="https://github.com/user-attachments/assets/00911ec8-9ddc-4f7a-8774-bef18bc9aacb" />

* **Training Accuracy:** ~93%
* **Validation Accuracy:** ~91%
* **Nhận xét:** Đường biểu diễn Accuracy của tập Validation bám rất sát tập Training, và Loss giảm đều xuống mức thấp (~0.25). Điều này chứng minh mô hình **không bị Overfitting** và có khả năng tổng quát hóa tốt trên dữ liệu lạ.

### 2. Case Study: Xử lý ảnh Panorama (Tỉ lệ cực đoan 5:1)
Đây là bài kiểm tra áp lực (Stress Test) để minh chứng hiệu quả của Smart Padding.

<img width="1644" height="1079" alt="Hình 4 5 Phân tích so sánh sai số dự đoán trên trường hợp ảnh có tỷ lệ khung hình cực đoan" src="https://github.com/user-attachments/assets/ea7be57e-a83b-4483-a1dd-bf63c30ce940" />

> **📝 Technical Analysis:**
> * **Ảnh gốc:** Kích thước 2576 x 517 pixel (Tỷ lệ ~5:1).
> * **Resize thông thường (Trái):** Ảnh bị biến dạng nặng, mất đặc trưng hình học của hoa.
> * **Smart Padding (Phải):** Giữ nguyên tỷ lệ, thêm viền đen.
> * **Kết quả:** Dù vùng thông tin hữu ích chỉ chiếm ~20% (còn lại là 80% viền đen), Model vẫn dự đoán **CHÍNH XÁC** loài hoa (Black-eyed Susan).
> * **Độ tin cậy (Confidence):** Đạt **~55%**. Con số này cao gấp 3 lần so với xác suất ngẫu nhiên (16%), chứng tỏ độ bền bỉ của thuật toán trong điều kiện nhiễu cao.

## 📂 Project Structure

```text
Flower-Classification-MobileNetV2/
├── Sample_images/                        # Ảnh mẫu dùng để test (Đã phân loại sẵn)
│   ├── black_eyed_susan/                 # Bao gồm cả ảnh Panorama
│   ├── calendula/
│   └── ...
├── 01_Flower_Training_MobileNetV2.ipynb  # Code huấn luyện (Full Pipeline)
├── 02_Flower_Predictor.ipynb             # Code dự đoán (Demo với Smart Padding)
├── LICENSE                               # Giấy phép MIT
└── README.md                             # Tài liệu dự án
```
## 🚀 How to Run (Hướng dẫn chạy)

Không cần cài đặt môi trường phức tạp hay tải dữ liệu thủ công. Chỉ cần làm theo các bước đơn giản sau:

1.  Mở file `02_Flower_Predictor.ipynb` trên GitHub này.
2.  Nhấn nút **Open in Colab** (hoặc tải về và upload lên Google Colab).
3.  Chọn menu **Runtime -> Run all**.
    * Hệ thống sẽ tự động tải Model đã huấn luyện.
    * Tải các ảnh trong thư mục `Sample_images` về máy để upload lên kiểm thử ngay lập tức (bao gồm cả các trường hợp khó).

## 📢 Acknowledgements & Data Source (Nguồn dữ liệu)

Dự án sử dụng bộ dữ liệu hoa từ Kaggle. Xin gửi lời cảm ơn chân thành đến tác giả vì đã chia sẻ nguồn dữ liệu quý giá này cho cộng đồng nghiên cứu.

* **Dataset:** [Flower Images by Zahra Aghapour](https://www.kaggle.com/datasets/zahraaghapour/flowers-images)
* **Original Source:** Kaggle
* **Tools:** TensorFlow, Keras, OpenCV.

---
### 👤 Author
* **Developer:** Bui Tien Phat
* **Contact:** higo.individual@gmail.com
* **Role:** AI Engineer / Computer Vision Researcher
