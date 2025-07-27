tar -zxvf onnxruntime-linux-x64-1.21.0.tgz
sudo cp -r onnxruntime-linux-x64-1.21.0/lib/libonnxruntime* /usr/local/lib64/.

sudo mkdir /usr/local/lib64/cmake/
sudo cp -r onnxruntime-linux-x64-1.21.0/lib/cmake/onnxruntime/ /usr/local/lib64/cmake/.

sudo cp -r onnxruntime-linux-x64-1.21.0/include/onnxruntime/ /usr/local/include/.

sudo ldconfig
