# Machine Learning Visualization Project

This repository contains visualizations for different machine learning algorithms. The images demonstrate clustering, image impression, linear regression, and gradient descent applied to parabolic shapes.

## Visualizations

- Dùng houghline detect các ô đậu xe

![Parking lot](assets/parking.jpg)

- Các bước xử lí: 

### 1. Đọc ảnh và chuyển ảnh qua ảnh xám. 
![Buoc 1](assets/B1.png)

### 2. Dùng bộ lọc Gaussian.
Sử dụng Gaussian với kernel kích thước 5x5,  với cả 2 phương x, y của ảnh. Mục đích của bước làm này để loại bỏ một số đường thẳng không mong muốn. 
![Buoc 2](assets/B2.png)

### 3. Tìm cạnh
Sử dụng Canny với ngưỡng dưới và trên lần lượt là 100 và 200. 
This visualization uses linear regression to model a parabolic function.
![Buoc 3](assets/B3.png)

### 4. Tìm các đường thẳng sử dụng Hough line. 
![Buoc 4](assets/B4.png)
Trong trường hợp không sử dụng bộ lọc nhiễu Gaussian trước khi tìm cạnh Canny. Kết quả Hough line như hình dưới. Có rất nhiều đường thẳng không phải vạch đỗ xe.
![Buoc 4 - 1](assets/B4-1.png)
Hough lines trả về giá trị  và , từ đó ta xác định phương trình đường thẳng như sau:
##TODO

### 5. Loại bỏ các đường thẳng liền kề nhau. 
Nếu các đường thẳng có điểm trên biên cách nhau nhỏ hơn hoặc bằng min_distance thì chỉ giữ lại đường thẳng có tọa độ trên biên bé nhất trong số các đường thẳng liền kề đó. Hình dưới là kết quả khi thực hiện với min_distance=15.
![Buoc 5](assets/B5.png)

### 6. Xác định giao điểm các đường thẳng với nhau và giao điểm đường thẳng với biên, từ đó vẽ ra các ô đậu xe. 
![Buoc 6](assets/B6.png)
![Buoc 6](assets/B6-1.png)
![Buoc 6](assets/B6-2.png)
![Buoc 6](assets/B6-3.png)
![Buoc 6](assets/B6-4.png)
![Buoc 6](assets/B6-5.png)
![Buoc 6](assets/B6-6.png)
![Buoc 6](assets/B6-7.png)
![Buoc 6](assets/B6-8.png)

