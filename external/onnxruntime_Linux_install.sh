tar -zxvf onnxruntime-linux-x64-1.21.0.tgz
sudo cp -r onnxruntime-linux-x64-1.21.0/lib/libonnxruntime* /usr/local/lib64/.

sudo mkdir -p /usr/local/lib64/cmake/
sudo cp -r onnxruntime-linux-x64-1.21.0/lib/cmake/onnxruntime/ /usr/local/lib64/cmake/.

if [ -d "onnxruntime-linux-x64-1.21.0/include/onnxruntime/" ]; then
    sudo cp -r onnxruntime-linux-x64-1.21.0/include/onnxruntime/ /usr/local/include/.
else
    sudo cp -r onnxruntime-linux-x64-1.21.0/include/* /usr/local/include/.
fi

sudo ldconfig
