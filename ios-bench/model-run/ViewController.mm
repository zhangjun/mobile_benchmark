//
//  ViewController.m
//  model-run
//
//  Created by Apple on 2021/7/9.
//

#import "ViewController.h"
#include "image_utils.h"
#include <string>
#include <vector>
#include <sstream>
#include "ModelBenchmark.hpp"

struct Model;

struct Model {
    std::string name;
    std::string model_file;
    std::string shape;
    int num_threads = 1;
    int repeat_runs = 10;
    int warmup_runs = 10;
    bool nb_model;
    bool use_mps;
    bool use_gpu;
};


NSString* FilePathForResourceName(NSString* filename) {
  NSString* name = [filename stringByDeletingPathExtension];
  NSString* extension = [filename pathExtension];
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    NSLog(@"Couldn't find '%@.%@' in bundle.", name, extension);
    exit(-1);
  }
  return file_path;
}

NSDictionary* ParseJson(NSString* json_config) {
  NSData* data = [NSData dataWithContentsOfFile:json_config];
  return [NSJSONSerialization JSONObjectWithData:data options:kNilOptions error:nil];
}

void ParseConfig(NSString* json_config, Model* model) {
  NSDictionary* param_dict = ParseJson(json_config);
  for (NSString* key in param_dict) {
    NSString* value = param_dict[key];
    if ([key isEqualToString:@"graph"]) {
      value = FilePathForResourceName(value);
    }
    
    if ([key isEqualToString:@"name"]) {
      model -> name = [value UTF8String];
    }
    
    if ([key isEqualToString:@"num_threads"]) {
      model -> num_threads = [value intValue];
    }

    if ([key isEqualToString:@"repeat_runs"]) {
      model -> repeat_runs = [value intValue];
    }

    if ([key isEqualToString:@"warmup_runs"]) {
      model -> warmup_runs = [value intValue];
    }

    if ([key isEqualToString:@"input_shapes"]) {
      model -> shape = [value UTF8String];
    }
    
    if ([key isEqualToString:@"use_gpu"]) {
      // model -> use_gpu = [[param_dict objectForKey:key] boolValue];
      model -> use_gpu = [value boolValue];
      NSLog(@"use_gpu: %@", model -> use_gpu?@"YES":@"NO");
    }
    
    if ([key isEqualToString:@"use_mps"]) {
      model -> use_mps = [[param_dict objectForKey:key] boolValue];
    }
    //params->push_back(FormatCommandLineParam(key, value));
  }
}

void image_process(NSString* image_path) {
    // Read the Grace Hopper image.
    //NSString* image_path = FilePathForResourceName(@"grace_hopper", @"jpg");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<uint8_t> image_data =
        LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    const int wanted_width = 224;
    const int wanted_height = 224;
    const int wanted_channels = 3;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    assert(image_channels >= wanted_channels);
    uint8_t* in = image_data.data();
    float* out;      // input tensor
    for (int y = 0; y < wanted_height; ++y) {
      const int in_y = (y * image_height) / wanted_height;
      uint8_t* in_row = in + (in_y * image_width * image_channels);
      float* out_row = out + (y * wanted_width * wanted_channels);
      for (int x = 0; x < wanted_width; ++x) {
        const int in_x = (x * image_width) / wanted_width;
        uint8_t* in_pixel = in_row + (in_x * image_channels);
        float* out_pixel = out_row + (x * wanted_channels);
        for (int c = 0; c < wanted_channels; ++c) {
          out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
        }
      }
    }
}

struct BenchOption {
    int warm_count = 10;
    int forward_count = 20;
    int create_count = 1;
    
    std::string description() {
        std::ostringstream ostr;
        ostr << "create_count = " << create_count
        << "  warm_count = " << warm_count
        << "  forward_count = " << forward_count;
        
        ostr << "\n";
        return ostr.str();
    };
};


std::vector<Model> getAllModels() {
    NSString *modelZone = [[NSBundle mainBundle] pathForResource:@"model"
                                                          ofType:nil];
    NSArray *modelList = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:modelZone error:nil];
    NSLog(@"%@", modelZone);
    
    std::vector<Model> models;
    
    NSPredicate *modelFormat = [NSPredicate predicateWithFormat:@"self ENDSWITH 'nb'"];
    NSPredicate *configFormat = [NSPredicate predicateWithFormat:@"self ENDSWITH 'json'"];
    for(NSString *modelDir in modelList) {
        NSString *modelDirPath = [modelZone stringByAppendingPathComponent:modelDir];
        BOOL isDirectory = NO;
        if ([[NSFileManager defaultManager] fileExistsAtPath:modelDirPath
                                                 isDirectory:&isDirectory]) {
            if (!isDirectory) {
                continue;
            }
            NSLog(@"%@", modelDirPath);
            Model model;
            model.name = modelDir.UTF8String;
            
            NSArray *dirFiles = [[NSFileManager defaultManager] contentsOfDirectoryAtPath:modelDirPath error:nil];
            
            // find nb model
            NSArray<NSString *> *nbFiles = [dirFiles filteredArrayUsingPredicate:modelFormat];
            if(nbFiles.count > 0) {
                model.model_file = [modelDirPath stringByAppendingPathComponent:nbFiles[0]].UTF8String;
                NSLog(@"model_file: %s", model.model_file.c_str());
            }
            
            // find json config
            NSArray<NSString *> *configFiles = [dirFiles filteredArrayUsingPredicate:configFormat];
            if(nbFiles.count > 0) {
//                std::string config_file = [modelDirPath stringByAppendingPathComponent:configFiles[0]].UTF8String;
                ParseConfig([modelDirPath stringByAppendingPathComponent:configFiles[0]], &model);
                //NSLog(@"config: %s", config_file.c_str());
            } else {
                continue;
            }
            
            models.push_back(model);
        }
    }
    
    return models;
}

BenchResult runBenchmark(const Model &model, const std::string& metal_lib) {
    BenchResult result;
    ModelBenchmark benchmark(model.model_file, metal_lib, model.shape, model.num_threads, model.warmup_runs, model.repeat_runs);
    benchmark.init();
    result = benchmark.get_result();
    return result;
}

std::string Result() {
    // prepare models
    auto allModels = getAllModels();
    
    NSString *libPath = [[NSBundle mainBundle] pathForResource:@"lib" ofType:nil];
    NSString *metalLib = [libPath stringByAppendingPathComponent:@"lite.metallib"];
    
    NSString *allResult = [NSString string];
    // run benchmark
    for(auto model: allModels) {
      allResult = [allResult stringByAppendingFormat:@"model: %s\n", model.name.c_str()];
      
      auto result = runBenchmark(model, [metalLib UTF8String]);
      // NSLog(@"result: %s\n", result.description().c_str());
      allResult = [allResult stringByAppendingFormat:@"result:\n %s\n",
                     result.description().c_str()];
    }
    // process result
    return [allResult UTF8String];
}

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
    [self initPredictor];
}

- (void)initPredictor {
    
}
- (IBAction)runPredict:(id)sender {
    // run benchmark
    std::string result = Result();
    NSString* data = [NSString stringWithFormat:@"%s", result.c_str()];
    NSLog(@"results: %@", data);
    self.resultsView.text = data;
    
//    if (@available(iOS 13.0, *)) {
//        self.resultsView.accessibilityTextualContext = data;
//    } else {
//        // Fallback on earlier versions
//    }
}


@end
