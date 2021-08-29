//
//  ViewController.m
//  MNN-bench
//
//  Created by Apple on 2021/8/28.
//

#import "ViewController.h"
#include <string>
#import "Executor.hpp"
//#import "Executor.hpp"
#import "benchmark.h"

@interface ViewController ()

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view.
}

- (IBAction)Run:(UIButton *)sender {
#define BENCHMARK
#ifdef BENCHMARK
    // benchmark
    {
        auto bundle = CFBundleGetMainBundle();
        auto url    = CFBundleCopyBundleURL(bundle);
        auto string = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
        CFRelease(url);
        auto cstring = CFStringGetCStringPtr(string, kCFStringEncodingUTF8);
        auto res     = std::string(cstring) + "/models";
        NSLog(@"path: %s", res.c_str());
        CFRelease(string);
        iosBenchAll(res.c_str());
    }
#endif
    
    // run benchmark
    std::string result = "benchmark test";
    NSString* data = [NSString stringWithFormat:@"%s", result.c_str()];
    NSLog(@"results: %@", data);
    self.ResultArea.text = data;
//    [ResultArea setText:[NSString stringWithUTF8String:results.c_str()]];
}

@end
