#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "features.h"

#undef OLD_METHOD

// Raw Hex Data (Ready on Edge Impulse -> Live Classification part)


#ifndef OLD_METHOD
__global__ void listExtract(unsigned int* d_raw_in, float* d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        unsigned int hex_val = d_raw_in[idx];
        float r = static_cast<float>((hex_val >> 16) & 0xFF);
        float g = static_cast<float>((hex_val >> 8) & 0xFF);
        float b = static_cast<float>(hex_val & 0xFF);

        d_out[idx * 3 + 0] = r;
        d_out[idx * 3 + 1] = g;
        d_out[idx * 3 + 2] = b;
    }

}
#endif // !OLD_METHOD


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

int main(int argc, char** argv) {
    std::cout << "Jetson Orin Nano TensorRT Inferencing... \n";

    // 1. FP32 / FP16 / int8 Engine Loading
    std::ifstream file("model_int8_calibrated.engine", std::ios::binary);
    if (!file.good()) {
        std::cerr << "Engine Error!\n";
        return -1;
    }
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    /* 
        Parsing HEX values as R, G, B values
        OLD_METHOD indicates that the parsing process will be done on CPU, 
        while the other will utilize from GPU to acceleration.
    */
    size_t total_pixels = sizeof(raw_features) / sizeof(raw_features[0]);
    std::vector<float> h_input(total_pixels * 3);
    #ifdef OLD_METHOD
    auto start_old = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < total_pixels; i++) {
        unsigned int hex_val = raw_features[i];
        float r = static_cast<float>((hex_val >> 16) & 0xFF);
        float g = static_cast<float>((hex_val >> 8) & 0xFF);
        float b = static_cast<float>(hex_val & 0xFF);

        h_input[i * 3 + 0] = r;
        h_input[i * 3 + 1] = g;
        h_input[i * 3 + 2] = b;
    }
    auto stop_old = std::chrono::high_resolution_clock::now();
    #endif // OLD_METHOD

    size_t input_size = total_pixels * sizeof(float) * 3;
    void* d_input = nullptr;                    // Holds the result of listExtract method.
    cudaMalloc(&d_input, input_size);           // This field will remain both the array created by the ListExtract method and an input for the inference phase.

#ifndef OLD_METHOD
    void* d_raw_in = nullptr;                   // Represents the raw_feature input array on GPU Memory.
    cudaMalloc(&d_raw_in, sizeof(raw_features));
    cudaMemcpy(d_raw_in, &raw_features[0], total_pixels * sizeof(unsigned int), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    // Warm-up to calculate exact process correctly
    for (int i = 0; i < 10; i++) {
        listExtract << <blocksPerGrid, threadsPerBlock >> > (
            static_cast<unsigned int*>(d_raw_in),
            static_cast<float*>(d_input),
            total_pixels
            );
    }
    cudaDeviceSynchronize();

    
    int k_iters = 100;
    cudaEventRecord(start1);
    for (int i = 0; i < k_iters; i++) {
        listExtract << <blocksPerGrid, threadsPerBlock >> > (
            static_cast<unsigned int*>(d_raw_in),
            static_cast<float*>(d_input),
            total_pixels
            );
    }
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    float ms = 0;
    cudaEventElapsedTime(&ms, start1, stop1);
    std::cout << "GPU List Parsing: " << (ms / k_iters) << " ms" << std::endl;

    // Clean Up
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
#endif // !OLD_METHOD

    
    size_t output_size = 1 * 11 * sizeof(float); // 11 of output class

    void* d_output = nullptr;
    cudaMalloc(&d_output, output_size);

#ifdef OLD_METHOD
    // 3. Data Transmitting into GPU.
    cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice);
    // It is not used because the data is already available in the GPU as d_input.
#endif // OLD_METHOD


    void* bindings[] = { d_input, d_output };

    // Inferencing Part
    // Warm-up
    for (int i = 0; i < 5; i++) {
        context->executeV2(bindings);
    }

    int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        context->executeV2(bindings);
    }
    auto end = std::chrono::high_resolution_clock::now();

    float duration_ms = std::chrono::duration<float, std::milli>(end - start).count() / iterations;

    // Get the results back into CPU
    std::vector<float> h_output(11);
    cudaMemcpy(h_output.data(), d_output, output_size, cudaMemcpyDeviceToHost);

    int predicted_class = 0;
    float max_score = h_output[0];

    std::cout << "\n--- Results ---\n";
    for (int i = 0; i < 11; i++) {
        std::cout  << class_list.at(i+1) <<  ": " << h_output[i] << "\n";
        if (h_output[i] > max_score) {
            max_score = h_output[i];
            predicted_class = i;
        }
    }

    std::cout << "-----------------------------------\n";
    std::cout << "Inferencing Time: " << duration_ms << " ms\n";
    std::cout << "=> Predicted Class: " << predicted_class << " (Score: " << max_score << ")\n";
    std::cout << "-----------------------------------\n";

    // Clean Up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_raw_in);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}