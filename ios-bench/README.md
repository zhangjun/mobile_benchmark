# ios-bench

# Compile Paddle-Lite for iOS GPU
```
git clone https://github.com/PaddlePaddle/Paddle-Lite
cd Paddle-Lite
./lite/tools/build_ios.sh --with_metal=ON
```
详细参见Paddle-Lite [iOS GPU预测库编译](https://paddle-lite.readthedocs.io/zh/develop/source_compile/compile_ios.html)

# Run Benchmark
* 打开model-run.xcodeproj, 替换include，预测库lib和metallib
* 将待测模型以目录形式放置在model目录下，修改config.json中配置信息
* 连接手机，XCode点击Run按钮进行测试