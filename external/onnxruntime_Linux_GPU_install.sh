curl -L -O -J https://github.com/Remdox/Yolo11-Card-Detection-cpp/releases/latest/download/onnxruntime-linux-x64-gpu-1.21.0.zip

unzip onnxruntime-linux-x64-gpu-1.21.0.zip

sudo cp -r onnxruntime-linux-x64-gpu-1.21.0/lib/libonnxruntime* /usr/local/lib64/.

sudo mkdir -p /usr/local/lib64/cmake/
sudo cp -r onnxruntime-linux-x64-gpu-1.21.0/lib/cmake/onnxruntime/ /usr/local/lib64/cmake/.

sudo mkdir -p /usr/local/include/onnxruntime/
sudo cp -r onnxruntime-linux-x64-gpu-1.21.0/include/* /usr/local/include/onnxruntime

sudo ldconfig
