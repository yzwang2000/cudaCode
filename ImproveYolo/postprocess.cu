#include <cuda_runtime.h>
#include <cmath>
#include <cuda.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace
{
    __device__ float sigmoid(float x)
    {
        return 1.0 / (1.0 + exp(-x));
    }

    __global__ void gpuYoloLayer_nc_has_sigmoid(
        const float *input, int *num_detections, float *detection_boxes, float *detection_scores, int *detection_classes,
        const float scoreThreshold, const uint netWidth, const uint netHeight, const uint gridSizeX, const uint gridSizeY,
        const uint numOutputClasses, const uint numBBoxes, const float scaleXY, const float *anchors)
    {
        // x, y, z 方向的坐标
        uint x_id = threadIdx.x + blockDim.x * blockIdx.x;
        uint y_id = threadIdx.y + blockDim.y * blockIdx.y;
        uint z_id = threadIdx.z + blockDim.z * blockIdx.z;

        if (x_id >= gridSizeX || y_id >= gridSizeY || z_id >= numBBoxes)
            return;

        const int numGridCells = gridSizeX * gridSizeY;
        const int bbindx = y_id * gridSizeX + x_id;

        const float objectness = sigmoid(input[numGridCells * ((numOutputClasses + 5) * z_id + 4) + bbindx]);

        if (objectness < scoreThreshold)
            return;

        // 原子计算
        int count = (int)atomicAdd(num_detections, 1);

        const float alpha = scaleXY;
        const float beta = -0.5 * (scaleXY - 1);

        float x = (sigmoid(input[numGridCells * ((numOutputClasses + 5) * z_id + 0) + bbindx]) * alpha + beta + x_id);
        float y = (sigmoid(input[numGridCells * ((numOutputClasses + 5) * z_id + 1) + bbindx]) * alpha + beta + y_id);

        float w = powf(sigmoid(input[numGridCells * ((numOutputClasses + 5) * z_id + 2) + bbindx]) * alpha, 2) * anchors[z_id * 2] * gridSizeX / netWidth;
        float h = powf(sigmoid(input[numGridCells * ((numOutputClasses + 5) * z_id + 3) + bbindx]) * alpha, 2) * anchors[z_id * 2 + 1] * gridSizeY / netHeight;

        float maxProb = -6.0f;
        int maxIndex = -1;
        for (uint i = 0; i < numOutputClasses; i++)
        {
            float prob = input[numGridCells * ((numOutputClasses + 5) * z_id + (i + 5)) + bbindx];
            // float prob = input[numGridCells*((numOutputClasses+5)*z_id + (i+5)) + bbindx];
            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = i;
            }
        }

        detection_boxes[count * 4 + 0] = (x - 0.5 * w) / gridSizeX;
        detection_boxes[count * 4 + 1] = (y - 0.5 * h) / gridSizeY;
        detection_boxes[count * 4 + 2] = (x + 0.5 * w) / gridSizeX;
        detection_boxes[count * 4 + 3] = (y + 0.5 * h) / gridSizeY;
        detection_scores[count] = objectness * sigmoid(maxProb);
        detection_classes[count] = maxIndex;
    }

    __global__ void gpuYoloLayer_nc_unroll(
        const float *input, int *num_detections, float *detection_boxes, float *detection_scores, int *detection_classes,
        const float scoreThreshold, const uint netWidth, const uint netHeight, const uint gridSizeX, const uint gridSizeY,
        const uint numOutputClasses, const uint numBBoxes, const float scaleXY, const float *anchors)
    {

        // 实现网格跨步访问, 就不用再判断是否超边界了
        for (uint x_id = threadIdx.x + blockDim.x * blockIdx.x,
                  y_id = threadIdx.y + blockDim.y * blockIdx.y,
                  z_id = threadIdx.z + blockDim.z * blockIdx.z;
             x_id < gridSizeX && y_id < gridSizeY && z_id < numBBoxes;
             x_id += blockDim.x * gridDim.x, y_id += blockDim.y * gridDim.y, z_id += blockDim.z * gridDim.z)
        {

            const int numGridCells = gridSizeX * gridSizeY;
            const int bbindx = y_id * gridSizeX + x_id;

            const float objectness = input[numGridCells * ((numOutputClasses + 5) * z_id + 4) + bbindx];

            if (objectness < scoreThreshold)
                return;

            // 原子计算, 这个原子计算来进行规约
            int count = (int)atomicAdd(num_detections, 1);

            const float alpha = scaleXY;
            const float beta = -0.5 * (scaleXY - 1);

            float x = (input[numGridCells * ((numOutputClasses + 5) * z_id + 0) + bbindx] * alpha + beta + x_id);
            float y = (input[numGridCells * ((numOutputClasses + 5) * z_id + 1) + bbindx] * alpha + beta + y_id);

            float w = powf(input[numGridCells * ((numOutputClasses + 5) * z_id + 2) + bbindx] * alpha, 2) * anchors[z_id * 2] * gridSizeX / netWidth;
            float h = powf(input[numGridCells * ((numOutputClasses + 5) * z_id + 3) + bbindx] * alpha, 2) * anchors[z_id * 2 + 1] * gridSizeY / netHeight;

            // 得到预测的概率
            float maxProb = 0.0f;
            int maxIndex = -1;
            #pragma unroll // 提示编译器进行循环展开
            for (uint i = 0; i < numOutputClasses; i++)
            {
                float prob = input[numGridCells * ((numOutputClasses + 5) * z_id + (i + 5)) + bbindx];
                if (prob > maxProb)
                {
                    maxProb = prob;
                    maxIndex = i;
                }
            }

            detection_boxes[count * 4 + 0] = (x - 0.5 * w) / gridSizeX;
            detection_boxes[count * 4 + 1] = (y - 0.5 * h) / gridSizeY;
            detection_boxes[count * 4 + 2] = (x + 0.5 * w) / gridSizeX;
            detection_boxes[count * 4 + 3] = (y + 0.5 * h) / gridSizeY;
            detection_scores[count] = objectness * maxProb;
            detection_classes[count] = maxIndex;
        }
    }

    __global__ void gpuYoloLayer_nc(
        const float *input, int *num_detections, float *detection_boxes, float *detection_scores, int *detection_classes,
        const float scoreThreshold, const uint netWidth, const uint netHeight, const uint gridSizeX, const uint gridSizeY,
        const uint numOutputClasses, const uint numBBoxes, const float scaleXY, const float *anchors)
    {

        // x, y, z 方向的坐标
        uint x_id = threadIdx.x + blockDim.x * blockIdx.x;
        uint y_id = threadIdx.y + blockDim.y * blockIdx.y;
        uint z_id = threadIdx.z + blockDim.z * blockIdx.z;

        if (x_id >= gridSizeX || y_id >= gridSizeY || z_id >= numBBoxes)
            return;

        const int numGridCells = gridSizeX * gridSizeY;
        const int bbindx = y_id * gridSizeX + x_id;

        const float objectness = input[numGridCells * ((numOutputClasses + 5) * z_id + 4) + bbindx];

        if (objectness < scoreThreshold)
            return;

        // 原子计算
        int count = (int)atomicAdd(num_detections, 1);

        const float alpha = scaleXY;
        const float beta = -0.5 * (scaleXY - 1);

        float x = (input[numGridCells * ((numOutputClasses + 5) * z_id + 0) + bbindx] * alpha + beta + x_id);
        float y = (input[numGridCells * ((numOutputClasses + 5) * z_id + 1) + bbindx] * alpha + beta + y_id);

        float w = powf(input[numGridCells * ((numOutputClasses + 5) * z_id + 2) + bbindx] * alpha, 2) * anchors[z_id * 2] * gridSizeX / netWidth;
        float h = powf(input[numGridCells * ((numOutputClasses + 5) * z_id + 3) + bbindx] * alpha, 2) * anchors[z_id * 2 + 1] * gridSizeY / netHeight;

        float maxProb = 0.0f;
        int maxIndex = -1;
        #pragma unroll // 提示编译器进行循环展开
        for (uint i = 0; i < numOutputClasses; i++)
        {
            float prob = input[numGridCells * ((numOutputClasses + 5) * z_id + (i + 5)) + bbindx];
            if (prob > maxProb)
            {
                maxProb = prob;
                maxIndex = i;
            }
        }

        detection_boxes[count * 4 + 0] = (x - 0.5 * w) / gridSizeX;
        detection_boxes[count * 4 + 1] = (y - 0.5 * h) / gridSizeY;
        detection_boxes[count * 4 + 2] = (x + 0.5 * w) / gridSizeX;
        detection_boxes[count * 4 + 3] = (y + 0.5 * h) / gridSizeY;
        detection_scores[count] = objectness * maxProb;
        detection_classes[count] = maxIndex;
    }

}

cudaError_t cudaYoloLayer_nc(
    const void *input, int *num_detections, float *detection_boxes, float *detection_scores, int *detection_classes,
    const uint &batchSize, uint64_t &inputSize, uint64_t &outputSize, const float &scoreThreshold, const uint &netWidth,
    const uint &netHeight, const nvinfer1::DimsHW &gridSize, const uint &numOutputClasses, const uint &numBBoxes,
    const float &scaleXY, const float *anchors, cudaStream_t stream)
{
    dim3 block_dim(16, 16, 3);
    dim3 grid_dim((gridSize.w() + block_dim.x - 1) / block_dim.x,
                    (gridSize.h() + block_dim.y - 1) / block_dim.y,
                    (numBBoxes + block_dim.z - 1) / block_dim.z);  // 处理每一个 anchor

    for (uint batch = 0; batch < batchSize; batch++)
    {
        gpuYoloLayer_nc_unroll<<<grid_dim, block_dim>>>(
            static_cast<const float *>(input) + (batch * inputSize),
            num_detections + (batch),                    // batch_size, int
            detection_boxes + (batch * 4 * outputSize),  // batch_size*4*(featureH*featureW*3)
            detection_scores + (batch * outputSize),     // batch_size*(featureH*featureW*3)
            detection_classes + (batch * outputSize),    // batch_size*(featureH*featureW*3)
            scoreThreshold,                              // 置信度阈值
            netWidth, netHeight,                         // 输入模型的图片大小
            gridSize.w(), gridSize.h(),                  // 特征图的宽和长
            numOutputClasses,                            // 分类的类别 80
            numBBoxes,                                   // 每个一个特征图对应的 anchor 个数(3)
            scaleXY,                                     // 缩放偏移用的
            static_cast<const float *>(anchors));
    }

    return cudaGetLastError();
}

