#include <vector>
#include <unordered_map>
#include <random>
#include <torch/torch.h>
#include <torch/extension.h>

constexpr int64_t block_size = 256;                            // 每个 block 存放的 token 个数
constexpr int64_t n_kv_heads = 32;                             // kv head 的个数
constexpr int64_t head_dim = 128;                              // 每个 head 的维度
constexpr int64_t max_seq_len = 8192;                          // token 的最大个数 (prompt + generate)
constexpr int64_t num_blocks = (max_seq_len / block_size) * 5; // 提前分配的活跃的线程块个数, 为了保持动态的分配和释放

class CacheManager {
public:
    CacheManager(int64_t batch_size) : batch_size(batch_size) {
        // 缓存 k 和 v 用的, 是实际内存空间, 每个 block 中存放的是 block_size * n_kv_heads * head_dim 个数, 这个其实是个显存池
        k_cache_paged = torch::randn({num_blocks, block_size, n_kv_heads, head_dim}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));
        v_cache_paged = torch::randn({num_blocks, block_size, n_kv_heads, head_dim}, torch::dtype(torch::kBFloat16).device(torch::kCUDA));

        // 初始化有那些是空闲的 block, 与 k_cache_paged, v_cache_paged 是对应的
        for (int64_t i = 0; i < num_blocks; ++i) {
            free_blocks.insert(i);
        }

        // Initialize block table
        block_table.resize(batch_size);  // 每个 batch 对应一个 table
    }

    torch::Tensor get_block_table() {
        // 从所有 batch 中得到最大的 block 个数
        int64_t max_len = 0;
        for (const auto& blocks : block_table) {
            max_len = std::max(max_len, static_cast<int64_t>(blocks.size()));
        }

        auto table = torch::full({batch_size, max_len}, -1, torch::dtype(torch::kInt32).device(torch::kCUDA));
        for (int64_t i = 0; i < batch_size; ++i) {
            for (int64_t j = 0; j < block_table[i].size(); ++j) {
                table[i][j] = block_table[i][j].first;
            }
        }

        return table;
    }

    // 得到 k_cache_paged, v_cache_paged
    std::tuple<torch::Tensor, torch::Tensor> get_kv_cache() {
        return std::make_tuple(k_cache_paged, v_cache_paged);
    }

    // 统计每个 batch 缓存的 token 个数
    torch::Tensor get_last_pos() {
        std::vector<int64_t> last_pos(batch_size, 0);  // 最后一个 pos
        for (int64_t i = 0; i < batch_size; ++i) {
            const auto& blocks = block_table[i];
            if (!blocks.empty()) {
                last_pos[i] = (blocks.size() - 1) * block_size + blocks.back().second - 1;
            }
        }

        return torch::tensor(last_pos, torch::dtype(torch::kInt32).device(torch::kCUDA));
    }

    // eos_reached 是 batch_size 大小, input_text_mask 是 batch_size 大小
    void update(const torch::Tensor& eos_reached, const torch::Tensor& input_text_mask) {
        auto eos_cpu = eos_reached.to(torch::kCPU);
        auto mask_cpu = input_text_mask.to(torch::kCPU);

        // 一个 batch, 一个 batch 的处理
        for (int64_t i = 0; i < batch_size; ++i) {
            // 是填充的词, 遇到填充的词, 就不分配 KVCache
            if (mask_cpu[i].item<bool>() == true) continue;  // Skip if it's part of the original prompt

            // 这个 batch 已经结束了, 将这个 batch KV Cache 占用的 block 归还给 freeBlock
            if (eos_cpu[i].item<bool>() == false) {
                free_memory(i);
                continue;
            }

            auto& blocks = block_table[i];  // 将这个 batch 的拿出来
            // 如果 block 满了, 就得到一个新的 free_block 的索引
            if (blocks.empty() || blocks.back().second == block_size) {
                int64_t new_index = get_free_block();
                blocks.emplace_back(new_index, 1);
            } else {
                blocks.back().second++;
            }
        }
    }

    double get_fragmented_memory_size() {
        double size = 0;
        for (const auto& blocks : block_table) {
            if (!blocks.empty()) {
                size += (block_size - blocks.back().second) * n_kv_heads * head_dim * 2 * sizeof(float);
            }
        }
        return size;
    }

private:
    int64_t batch_size;         // batch 大小
    // key 是 batch(相当于每个batch 都有一个自己的映射表), value 是占用的内存块的序号数 和 占用的这个块的大小
    std::unordered_map<int64_t, std::vector<std::pair<int64_t, int64_t>>> block_table;
    std::set<int64_t> free_blocks;  // 存放空闲 block 的索引
    torch::Tensor k_cache_paged, v_cache_paged;  // 显存池

    // 找到空闲的 block
    int64_t get_free_block() {
        if (free_blocks.empty()) {
            throw std::runtime_error("No more free blocks available");
        }

        int64_t index = *free_blocks.begin();
        free_blocks.erase(index);
        return index;
    }

    // 释放某个 batch 所占用的显存
    void free_memory(int64_t index) {       // 这里 index 表示是第几个 batch
        auto& blocks = block_table[index];  // 引用出来
        if (blocks.size() > 1) {            // 还占用着 index
            // 归还给 free_blocks
            for (size_t i = 1; i < blocks.size(); ++i) {
                free_blocks.insert(blocks[i].first);
            }
            // 保留一个 block
            blocks.resize(1); // Keep only the first block
            blocks.back().second = 0;
        }
    }
};

int main() {
    int64_t batch_size = 4; // 示例值
    CacheManager cache_manager(batch_size);

    torch::Tensor tokens = torch::full({batch_size, max_seq_len}, 0, torch::dtype(torch::kInt64).device(torch::kCUDA));

    for (int64_t cur_pos = 0; cur_pos < max_seq_len; ++cur_pos) {
        // Placeholder for the `forward` function that generates the next token, 生成下一个 token
        torch::Tensor next_token = torch::randint(0, 32000, {batch_size}, torch::dtype(torch::kInt64).device(torch::kCUDA));

        // Update CacheManager
        // eos_reached 是 batch_size 个,  当某个 batch 为 true 的时候, 就是这个句子结束了。
        // 记录那个是真正的 token, 而不是填充词。
        torch::Tensor eos_reached = torch::zeros({batch_size}, torch::dtype(torch::kBool).device(torch::kCUDA));
        torch::Tensor input_text_mask = (tokens != 0);

        cache_manager.update(eos_reached, input_text_mask);

        // Check if all EOS tokens have been reached, 如果所有句子都结束了, 就提前结束
        if (eos_reached.all().item<bool>()) {
            break;
        }
    }

    // Calculate fragmented memory
    double fragmented_memory_size = cache_manager.get_fragmented_memory_size();
    double fragmented_ratio = fragmented_memory_size / torch::cuda::getDeviceProperties(0)->totalGlobalMem;

    std::cout << "Fragmented Memory: " << fragmented_memory_size / 1e9 << " GB (" << fragmented_ratio * 100.0 << "%)" << std::endl;

    return 0;
}