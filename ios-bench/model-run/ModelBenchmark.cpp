//
//  ModelBenchmark.cpp
//  model-run
//
//  Created by Apple on 2021/7/10.
//

#include "ModelBenchmark.hpp"


#include <sys/time.h>
#include <time.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "paddle_use_kernels.h"  // NOLINT
#include "paddle_use_ops.h"      // NOLINT

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

std::string ShapePrint(const std::vector<shape_t>& shapes) {
  std::string shapes_str{""};
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    auto shape = shapes[shape_idx];
    std::string shape_str;
    for (auto i : shape) {
      shape_str += std::to_string(i) + ",";
    }
    shapes_str += shape_str;
    shapes_str +=
        (shape_idx != 0 && shape_idx == shapes.size() - 1) ? "" : " : ";
  }
  return shapes_str;
}

std::string ShapePrint(const shape_t& shape) {
  std::string shape_str{""};
  for (auto i : shape) {
    shape_str += std::to_string(i) + " ";
  }
  return shape_str;
}

std::vector<std::string> split_string(const std::string& str_in) {
  std::vector<std::string> str_out;
  std::string tmp_str = str_in;
  while (!tmp_str.empty()) {
    size_t next_offset = tmp_str.find(":");
    str_out.push_back(tmp_str.substr(0, next_offset));
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return str_out;
}

std::vector<int64_t> get_shape(const std::string& str_shape) {
  std::vector<int64_t> shape;
  std::string tmp_str = str_shape;
  while (!tmp_str.empty()) {
    int dim = atoi(tmp_str.data());
    shape.push_back(dim);
    size_t next_offset = tmp_str.find(",");
    if (next_offset == std::string::npos) {
      break;
    } else {
      tmp_str = tmp_str.substr(next_offset + 1);
    }
  }
  return shape;
}

template <typename T>
double compute_mean(const T* in, const size_t length) {
  double sum = 0.;
  for (size_t i = 0; i < length; ++i) {
    sum += in[i];
  }
  return sum / length;
}

template <typename T>
double compute_standard_deviation(const T* in,
                                  const size_t length,
                                  bool has_mean = false,
                                  double mean = 10000) {
  if (!has_mean) {
    mean = compute_mean<T>(in, length);
  }

  double variance = 0.;
  for (size_t i = 0; i < length; ++i) {
    variance += pow((in[i] - mean), 2);
  }
  variance /= length;
  return sqrt(variance);
}

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}


std::shared_ptr<PaddlePredictor> CreatorPredictor(const std::string& model_file, const std::string& metal_lib, int thread_num, int power_mode) {
  // set config
  MobileConfig config;
  config.set_model_from_file(model_file);
  // config.set_power_mode(static_cast<paddle::lite_api::PowerMode>(power_mode));
  config.set_threads(thread_num);
  config.set_metal_lib_path(metal_lib);
  // 2. Create PaddlePredictor by MobileConfig
  std::shared_ptr<PaddlePredictor> predictor =
    CreatePaddlePredictor<MobileConfig>(config);
  return predictor;
}

void ModelBenchmark::debug() {
    // 5. Get output
    std::cout << "\n====== output summary ====== " << std::endl;
    if(predictor_ == nullptr) return;
    size_t output_tensor_num = predictor_->GetOutputNames().size();
    std::cout << "output tensor num:" << output_tensor_num << std::endl;

    for (size_t tidx = 0; tidx < output_tensor_num; ++tidx) {
      std::unique_ptr<const paddle::lite_api::Tensor> output_tensor =
          predictor_->GetOutput(tidx);
      std::cout << "\n--- output tensor " << tidx << " ---" << std::endl;
      auto out_shape = output_tensor->shape();
      auto out_data = output_tensor->data<float>();
      auto out_mean = compute_mean<float>(out_data, ShapeProduction(out_shape));
      auto out_std_dev = compute_standard_deviation<float>(
          out_data, ShapeProduction(out_shape), true, out_mean);

      std::cout << "output shape(NCHW):" << ShapePrint(out_shape) << std::endl;
      std::cout << "output tensor " << tidx
                << " elem num:" << ShapeProduction(out_shape) << std::endl;
      std::cout << "output tensor " << tidx
                << " standard deviation:" << out_std_dev << std::endl;
      std::cout << "output tensor " << tidx << " mean value:" << out_mean
                << std::endl;
    }

}
              
void ModelBenchmark::init() {
    // init
    std::vector<std::string> str_input_shapes;
    std::vector<shape_t> input_shapes{};  // shape_t ==> std::vector<int64_t>
    str_input_shapes = split_string(shapes_);
    for (size_t i = 0; i < str_input_shapes.size(); ++i) {
      input_shapes.push_back(get_shape(str_input_shapes[i]));
    }
    // create predictor
    predictor_ = CreatorPredictor(model_file_, metal_lib_, num_threads_, 0);
    
    // 3. Prepare input data
    std::cout << "input_shapes.size():" << input_shapes.size() << std::endl;
    for (int j = 0; j < input_shapes.size(); ++j) {
      auto input_tensor = predictor_->GetInput(j);
      input_tensor->Resize(input_shapes[j]);
      auto input_data = input_tensor->mutable_data<float>();
      int input_num = 1;
      for (int i = 0; i < input_shapes[j].size(); ++i) {
        input_num *= input_shapes[j][i];
      }

      for (int i = 0; i < input_num; ++i) {
        input_data[i] = 1.f;
      }
    }
    
    std::cout << "[version] " << predictor_ -> GetVersion() << std::endl;
    // debug
    // predictor_ -> Run();
    // debug();
}

BenchResult ModelBenchmark::get_result() {
  BenchResult result;
  
  // warmup
  for(int i = 0; i < warmup_times_; ++ i) {
      timeval tv_begin, tv_end;
      gettimeofday(&tv_begin, NULL);
      
      predictor_ -> Run();
      
      gettimeofday(&tv_end, NULL);
      double elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
      result.addTime(elapsed);
  }
  // run loop
  for(int i = 0; i < repeat_times_; ++ i) {
      timeval tv_begin, tv_end;
      gettimeofday(&tv_begin, NULL);
      
      predictor_ -> Run();
      
      gettimeofday(&tv_end, NULL);
      double elapsed = (tv_end.tv_sec - tv_begin.tv_sec) * 1000.0 + (tv_end.tv_usec - tv_begin.tv_usec) / 1000.0;
      result.addTime(elapsed);
  }
  debug();
    
  return result;
}
