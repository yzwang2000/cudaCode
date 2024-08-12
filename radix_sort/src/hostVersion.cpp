#include <iostream>
#include <vector>

// 基数排序是从最低位看起, 把这一位数字值相同的按照扫描顺序放入同一个桶里面, 值小的桶在前面。
// 当所有数字都扫描完，再使用高一位，循环上述步骤，直至达到所有数字所有的最高位数，最后输出的就是排序后的答案。
void radixSort(std::vector<unsigned int>& inputVals, int numBits = 2) {
    int n = inputVals.size();    // 输入元素的个数
    int numBins = 1 << numBits;  // numBins = 2^numBits, 桶的个数
    int maxBits = 32;  // Assuming unsigned int (32-bit)

    std::vector<unsigned int> outputVals(n);  // 存放排序后的结果
    for (int i = 0; i < maxBits; i += numBits){
        std::vector<int> binHistogram(numBins, 0);
        std::vector<int> binScan(numBins, 0);

        // Step 1: 计算统计图
        for (int j = 0; j < n; ++j) {  // 遍历所有元素, 统计每个桶中元素个数
            int bin = (inputVals[j] >> i) & (numBins - 1);
            binHistogram[bin]++;
        }

        // Step 2: 计算前缀和
        for (int j = 1; j < numBins; ++j) {  // 遍历所有桶, 统计元素个数的前缀和
            binScan[j] = binScan[j - 1] + binHistogram[j - 1];
        }

        // Step 3: 基于当前数字来排序
        for (int j = 0; j < n; ++j) {  // 遍历所有元素, 将输入元素赋值到输出对应的位置
            int bin = (inputVals[j] >> i) & (numBins - 1);  // 判断当前数字属于那个 bin, 调整其位置
            outputVals[binScan[bin]] = inputVals[j];
            binScan[bin]++;
        }

        // Step 4: 下一轮的排序基于当前排序的结果
        inputVals.swap(outputVals);
    }
}

int main() {
    std::vector<unsigned int> inputVals = {170, 45, 75, 90, 802, 24, 2, 66};
    radixSort(inputVals, 2);

    for (unsigned int val : inputVals) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}