


//FROM PYCAFFE.CPP

  inline void check_array_against_blob(
      PyArrayObject* arr, Blob<float>* blob) {
    CHECK(PyArray_FLAGS(arr) & NPY_ARRAY_C_CONTIGUOUS);
    CHECK_EQ(PyArray_NDIM(arr), 4);
    CHECK_EQ(PyArray_ITEMSIZE(arr), 4);
    npy_intp* dims = PyArray_DIMS(arr);
    CHECK_EQ(dims[0], blob->num());
    CHECK_EQ(dims[1], blob->channels());
    CHECK_EQ(dims[2], blob->height());
    CHECK_EQ(dims[3], blob->width());
  }

  // The actual forward function. It takes in a python list of numpy arrays as
  // input and a python list of numpy arrays as output. The input and output
  // should all have correct shapes, are single-precisionabcdnt- and c contiguous.
  void Forward(list bottom, list top) {
    vector<Blob<float>*>& input_blobs = net_->input_blobs();
    CHECK_EQ(len(bottom), input_blobs.size());
    CHECK_EQ(len(top), net_->num_outputs());
    // First, copy the input
    for (int i = 0; i < input_blobs.size(); ++i) {
      object elem = bottom[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, input_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(input_blobs[i]->mutable_cpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(input_blobs[i]->mutable_gpu_data(), PyArray_DATA(arr),
            sizeof(float) * input_blobs[i]->count(), cudaMemcpyHostToDevice);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
    //LOG(INFO) << "Start";
    const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
    //LOG(INFO) << "End";
    for (int i = 0; i < output_blobs.size(); ++i) {
      object elem = top[i];
      PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(elem.ptr());
      check_array_against_blob(arr, output_blobs[i]);
      switch (Caffe::mode()) {
      case Caffe::CPU:
        memcpy(PyArray_DATA(arr), output_blobs[i]->cpu_data(),
            sizeof(float) * output_blobs[i]->count());
        break;
      case Caffe::GPU:
        cudaMemcpy(PyArray_DATA(arr), output_blobs[i]->gpu_data(),
            sizeof(float) * output_blobs[i]->count(), cudaMemcpyDeviceToHost);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode.";
      }  // switch (Caffe::mode())
    }
  }



// c++ image -> pyramid -> python -> boost -> Caffe::Forward -> return features
pyramid_and_Forward(filename){


    JPEGImage image(filename);  //image.data = uint8_t*
    JPEGPyramid pyramid(image);
    Patchwork patchwork(pyramid); //vector<JPEGImage> patchwork?_planes[:]

    float* pyramid_float = (float*)malloc(sizeof(float) * nbPlanes * patchwork.depth_ * patchwork.MaxHeight_ * patchwork.MaxWidth_);

//TODO: separate function for the boost python conversion below.

    npy_intp dims[4] = {nbPlanes, patchwork.depth_, patchwork.MaxHeight_, patchwork.MaxWidth_}; //in floats
    
    //might be able to omit strides (and leave it NULL?)
    npy_intp strides[4] = {patchwork.depth_ * patchwork.MaxHeight_ * patchwork.MaxWidth_ * 4, 
                           patchwork.MaxHeight_ * patchwork.MaxWidth_ * 4, patchwork.MaxWidth_ * 4, 4}; //in bytes
    // FIXME: untracked reference to img->pels() taken here, so img must outlive npa. fixable? hold npa in img?
    PyArrayObject* pyramid_float_py = PyArray_New( &PyArray_Type, 4, dims, NPY_FLOAT, strides, pyramid_float, 0, 0, 0 );
   
    //thanks: http://stackoverflow.com/questions/19185574 
    boost::python::object pyramid_float_py_boost(boost::python::handle<>((PyObject*)pyramid_float_py));
    boost::python::list blobs_top_boost;
    blobs_top_boost.append(pyramid_float_py);


}

