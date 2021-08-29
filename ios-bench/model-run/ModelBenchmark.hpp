//
//  ModelBenchmark.hpp
//  model-run
//
//  Created by Apple on 2021/7/10.
//

#ifndef ModelBenchmark_hpp
#define ModelBenchmark_hpp

#include <stdio.h>
#include <string>
#include <float.h>
#include <sstream>

#include "paddle_api.h"  // NOLINT
/////////////////////////////////////////////////////////////////////////
// If this demo is linked to static library:libpaddle_api_light_bundled.a
// , you should include `paddle_use_ops.h` and `paddle_use_kernels.h` to
// avoid linking errors such as `unsupport ops or kernels`.
/////////////////////////////////////////////////////////////////////////

using namespace paddle::lite_api;  // NOLINT

struct BenchResult {
    //time
    float min = FLT_MAX;
    float max = FLT_MIN;
    float first = -1;
    float avg = 0;
    float total = 0;
    int count = 0;
    
    float diff = 0;
    
    int addTime(float time){
        count++;
        if(count == 1) {
            first = time;
            return 0;
        }
        total += time;
        min = std::min(min, time);
        max = std::max(max, time);
        avg = total/(count - 1);
        return 0;
    };

    void reset() {}
    
    std::string description() {
        std::ostringstream ostr;
        ostr << "first = " << first << "  min = " << min << "  max = " << max << "  avg = " << avg << std::endl;
        return ostr.str();
    };
};

class ModelBenchmark {
public:
    ModelBenchmark(const std::string& model_file, const std::string& metal_lib, const std::string& shapes, int num_threads = 1, int warmup_times = 10, int repeat_times = 10):
    model_file_(model_file),
    metal_lib_(metal_lib),
    shapes_(shapes),
    num_threads_(num_threads),
    warmup_times_(warmup_times),
    repeat_times_(repeat_times) {}

  void init();
  void debug();
  BenchResult get_result();
    
private:
  std::string model_file_;
  std::string metal_lib_;
  std::string shapes_;
  int num_threads_;
  int warmup_times_;
  int repeat_times_;
  std::shared_ptr<PaddlePredictor> predictor_;
};

#endif /* ModelBenchmark_hpp */
