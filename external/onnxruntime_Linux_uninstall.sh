sudo rm /usr/local/lib64/libonnxruntime*

sudo rm -r /usr/local/lib64/cmake/onnxruntime

if [ -d "/usr/local/include/onnxruntime" ]; then
    sudo rm -rf /usr/local/include/onnxruntime
else
    sudo rm -f /usr/local/include/cpu_provider_factory.h \
               /usr/local/include/onnxruntime_c_api.h \
               /usr/local/include/onnxruntime_cxx_api.h \
               /usr/local/include/onnxruntime_cxx_inline.h \
               /usr/local/include/onnxruntime_float16.h \
               /usr/local/include/onnxruntime_lite_custom_op.h \
               /usr/local/include/onnxruntime_run_options_config_keys.h \
               /usr/local/include/onnxruntime_session_options_config_keys.h \
               /usr/local/include/provider_options.h

    if [ -f "/usr/local/include/core/providers/resource.h" ]; then
        sudo rm -rf /usr/local/include/core/
    fi
fi
