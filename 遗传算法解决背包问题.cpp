/*
 * 遗传算法求解固定的 0-1 背包问题（C++17）。
 * 数据和适应度规则来自同目录的《遗传算法实验报告.docx》。
 *
 * 编译（在本文件所在目录执行，MinGW-w64 / MSYS2）：
 * g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -static -finput-charset=UTF-8 -fexec-charset=UTF-8 GeneticKnapsack.cpp -o GeneticKnapsack.exe
 * 运行：.\GeneticKnapsack.exe
 *
 * 输入示例（每个数输入一行）：
 *   6          种群规模；注意它与固定为 6 的“物品种类数”不同
 *   0.8        一对父代发生交叉的概率
 *   0.05       一个基因位发生翻转的概率
 *   1000       最多更新 1000 代；初始种群另外记为第 0 轮
 *   10         连续 10 轮最好适应度等于精确最优值，才确认稳定
 *   20261103   随机种子；同一程序、相同输入可复现，-1 表示自动生成
 *
 * 算法：轮盘赌选择 -> 单点交叉 -> 按位变异 -> 重新评价 -> 更新种群。
 * 每代原样保留 1 个精英，其余 N-1 个位置按子代生成顺序填充。
 *
 * “稳定”只指最好值连续 K 轮等于精确最优值，不代表全体个体收敛。
 * 有精英保留时，首次找到最优后它就不会丢失，所以通常在首次命中后的
 * 第 K-1 轮确认稳定。首次命中的这一轮也计入 K 轮。
 * 穷举求解独立进行，不把精确最优染色体放入遗传算法的初始种群。
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace knapsack {

// 以下是报告固定的数据，不在运行时输入，也不随实验参数改变。
constexpr std::size_t kItemCount = 6;  // 共 6 种物品，每种只有一件，不能分割。
constexpr int kCapacity = 15;         // 背包容量：总重量等于 15 也算可行。
constexpr std::array<int, kItemCount> kWeights = {2, 3, 4, 5, 7, 9};
constexpr std::array<int, kItemCount> kValues = {6, 5, 8, 9, 13, 17};
// 下标 0 对应物品 1，下标 5 对应物品 6。输出时不反转基因顺序。
using Chromosome = std::array<int, kItemCount>;

struct Individual {
    Chromosome genes{};  // 每一位只能为 0 或 1；全 0 表示空背包。
    int weight = 0;
    int value = 0;       // 超重时仍记录物品价值之和，便于与适应度区分。
    int fitness = 0;     // 可行：等于 value；超重：严格清零。

    bool feasible() const { return weight <= kCapacity; }
};

struct Settings {
    int populationSize = 0;
    double crossoverProbability = 0.0;
    double mutationProbability = 0.0;
    int maxGenerations = 0;
    int stableRounds = 0;
    std::int64_t seedInput = -1;  // -1 自动生成；0 也是有效的固定种子。
};

// 每次交叉或变异后，都通过这个函数重新计算，不能沿用父代得分。
Individual evaluate(const Chromosome& genes) {
    Individual result;
    result.genes = genes;
    for (std::size_t i = 0; i < kItemCount; ++i) {
        result.weight += genes[i] * kWeights[i];
        result.value += genes[i] * kValues[i];
    }
    result.fitness = result.feasible() ? result.value : 0;
    return result;
}

// 统一确定“谁更好”，用于精英、每轮显示和历来最好方案。
// 先比较适应度；同分时优先可行方案，再取字典序较小的染色体。
// 因而空背包（可行、0 分）优于任何超重且同为 0 分的方案。
bool better(const Individual& a, const Individual& b) {
    if (a.fitness != b.fitness) return a.fitness > b.fitness;
    if (a.feasible() != b.feasible()) return a.feasible();
    return a.genes < b.genes;
}

std::string chromosomeString(const Individual& individual) {
    std::string text;
    for (int bit : individual.genes) text += static_cast<char>('0' + bit);
    return text;
}

std::string selectedItems(const Individual& individual) {
    std::ostringstream text;
    text << '{';
    bool first = true;
    for (std::size_t i = 0; i < kItemCount; ++i) {
        if (individual.genes[i] == 1) {
            if (!first) text << ',';
            text << i + 1;
            first = false;
        }
    }
    text << '}';  // 空背包输出 {}。
    return text.str();
}

struct ExactResult {
    Individual best;  // 默认是可行的空背包。
    int feasibleCount = 0;
    int optimalCount = 0;
};

// 精确基准：独立枚举全部 2^6 = 64 个组合，而非把报告答案写成常量。
// mask 的最高有效位对应 genes[0]，保证字符串左端始终代表物品 1。
ExactResult solveExactly() {
    ExactResult result;
    constexpr unsigned int combinationCount = 1U << kItemCount;
    for (unsigned int mask = 0; mask < combinationCount; ++mask) {
        Chromosome genes{};
        for (std::size_t i = 0; i < kItemCount; ++i) {
            genes[i] = static_cast<int>((mask >> (kItemCount - 1 - i)) & 1U);
        }
        const Individual candidate = evaluate(genes);
        if (!candidate.feasible()) continue;
        ++result.feasibleCount;
        if (candidate.value > result.best.value) {
            result.best = candidate;
            result.optimalCount = 1;
        } else if (candidate.value == result.best.value) {
            ++result.optimalCount;
            if (better(candidate, result.best)) result.best = candidate;
        }
    }
    return result;
}

// 每个输入占一行：拒绝空行、越界数字以及“0.8abc”等尾随内容。
// 输入流结束（例如 Ctrl+Z 后回车，或文件读完）时返回 false，避免死循环。
template <typename T, typename Validator>
bool readValue(const std::string& prompt, T& destination, Validator valid,
               const std::string& requirement) {
    for (;;) {
        std::cout << prompt << std::flush;
        std::string line;
        if (!std::getline(std::cin, line)) return false;
        std::istringstream input(line);
        T value{};
        if (input >> value) {
            input >> std::ws;
            if (input.eof() && valid(value)) {
                destination = value;
                return true;
            }
        }
        std::cout << "输入无效，" << requirement << "，请重新输入。\n";
    }
}

bool readSettings(Settings& settings) {
    const auto positive = [](int x) { return x >= 1; };
    const auto probability = [](double x) {
        return std::isfinite(x) && x >= 0.0 && x <= 1.0;
    };
    return readValue("请输入种群规模 N（至少 2，建议 6）：", settings.populationSize,
                     [](int x) { return x >= 2; }, "必须是至少为 2 的整数")
        && readValue("请输入交叉概率（0~1，建议 0.8）：", settings.crossoverProbability,
                     probability, "必须是 0 到 1 之间的有限数字")
        && readValue("请输入按位变异概率（0~1，建议 0.05）：", settings.mutationProbability,
                     probability, "必须是 0 到 1 之间的有限数字")
        && readValue("请输入最大演化轮数 G（正整数，建议 1000）：", settings.maxGenerations,
                     positive, "必须是正整数")
        && readValue("请输入确认稳定的连续最优轮数 K（正整数，建议 10）：", settings.stableRounds,
                     positive, "必须是正整数")
        && readValue("请输入随机种子（-1 自动，0~4294967295 固定，建议 20261103）：",
                     settings.seedInput,
                     [](std::int64_t x) {
                         return x >= -1 && x <= std::numeric_limits<std::uint32_t>::max();
                     }, "必须是 -1 或 0 到 4294967295 之间的整数");
}

std::uint32_t resolveSeed(std::int64_t input) {
    if (input >= 0) return static_cast<std::uint32_t>(input);
    try {
        std::random_device device;
        return static_cast<std::uint32_t>(device());
    } catch (const std::exception&) {
        // 个别环境不支持 random_device，退回到当前时钟生成种子。
        const auto ticks = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        return static_cast<std::uint32_t>(ticks);
    }
}

std::vector<Individual> initializePopulation(const Settings& settings, std::mt19937& rng) {
    std::vector<Individual> population;
    population.reserve(static_cast<std::size_t>(settings.populationSize));
    std::bernoulli_distribution randomBit(0.5);
    bool hasFeasible = false;
    for (int i = 0; i < settings.populationSize; ++i) {
        Chromosome genes{};
        for (int& bit : genes) bit = randomBit(rng) ? 1 : 0;
        population.push_back(evaluate(genes));
        hasFeasible = hasFeasible || population.back().feasible();
    }
    // 仅在全部超重时兜底，不把精确最优解或报告的演示个体注入种群。
    // 以后每轮保留一个精英，所以始终至少存在一个可行方案。
    if (!hasFeasible) population.front() = evaluate(Chromosome{});
    return population;
}

struct GenerationStats {
    std::size_t bestIndex = 0;
    std::uint64_t totalFitness = 0;  // 用宽整数累加，避免大种群使 int 溢出。
    std::size_t feasibleCount = 0;
};

GenerationStats summarize(const std::vector<Individual>& population) {
    GenerationStats stats;
    for (std::size_t i = 0; i < population.size(); ++i) {
        if (better(population[i], population[stats.bestIndex])) stats.bestIndex = i;
        stats.totalFitness += static_cast<std::uint64_t>(population[i].fitness);
        if (population[i].feasible()) ++stats.feasibleCount;
    }
    return stats;
}

// 累计适应度相当于轮盘赌区间长度；权重为 0 的个体区间为空。
std::vector<std::uint64_t> cumulativeFitness(const std::vector<Individual>& population) {
    std::vector<std::uint64_t> cumulative;
    cumulative.reserve(population.size());
    std::uint64_t sum = 0;
    for (const Individual& individual : population) {
        sum += static_cast<std::uint64_t>(individual.fitness);
        cumulative.push_back(sum);
    }
    return cumulative;
}

std::size_t selectParent(const std::vector<std::uint64_t>& cumulative, std::mt19937& rng) {
    const std::uint64_t total = cumulative.back();
    if (total == 0) {
        // 所有适应度为 0 时没有可用的概率比例，改为等概率抽样。
        std::uniform_int_distribution<std::size_t> uniform(0, cumulative.size() - 1);
        return uniform(rng);
    }
    // 本例适应度都是整数，直接在 [0,total-1] 抽整数票，避免概率舍入。
    // 例如权重 23、22 对应票号 [0,22]、[23,44]，机会正好为 23:22。
    std::uniform_int_distribution<std::uint64_t> ticket(0, total - 1);
    const auto position = std::upper_bound(cumulative.begin(), cumulative.end(), ticket(rng));
    return static_cast<std::size_t>(position - cumulative.begin());
}

// cut 表示前 cut 位不交换，后面的位互换；合法切分点为 1~5。
void singlePointCrossover(Chromosome& a, Chromosome& b, std::size_t cut) {
    for (std::size_t i = cut; i < kItemCount; ++i) std::swap(a[i], b[i]);
}

void mutate(Chromosome& genes, double probability, std::mt19937& rng) {
    std::bernoulli_distribution flip(probability);
    // 每一位独立判断，允许一位也不变，或同一染色体中多位同时变异。
    for (int& bit : genes) {
        if (flip(rng)) bit = 1 - bit;
    }
}

std::vector<Individual> nextGeneration(const std::vector<Individual>& population,
                                     const Settings& settings, std::mt19937& rng) {
    std::vector<Individual> next;
    next.reserve(population.size());
    next.push_back(population[summarize(population).bestIndex]);  // 唯一精英直接保留。
    const auto cumulative = cumulativeFitness(population);
    std::bernoulli_distribution doCrossover(settings.crossoverProbability);
    std::uniform_int_distribution<std::size_t> cutPoint(1, kItemCount - 1);

    while (next.size() < population.size()) {
        // 有放回抽样：父代可以重复选中，同一个个体也可以与自身配对。
        Chromosome a = population[selectParent(cumulative, rng)].genes;
        Chromosome b = population[selectParent(cumulative, rng)].genes;
        if (doCrossover(rng)) singlePointCrossover(a, b, cutPoint(rng));

        mutate(a, settings.mutationProbability, rng);
        next.push_back(evaluate(a));
        if (next.size() < population.size()) {
            mutate(b, settings.mutationProbability, rng);
            next.push_back(evaluate(b));
        }
        // 只剩一个空位时，第二个子代不进入下一代，也不再对它变异。
        // 这由填充顺序决定，不比较两个子代的得分后临时挑选。
    }
    return next;
}

void printGeneration(std::ostream& out, std::int64_t generation,
                     const std::vector<Individual>& population, const GenerationStats& stats) {
    const Individual& best = population[stats.bestIndex];
    const double mean = static_cast<double>(stats.totalFitness)
                      / static_cast<double>(population.size());
    out << "第" << generation << "轮：得到最好结果字符串是" << chromosomeString(best)
        << "，选择物品是" << selectedItems(best)
        << "，最好适应度是" << best.fitness
        << "，平均适应度是" << std::fixed << std::setprecision(2) << mean
        << "，可行个体数量为" << stats.feasibleCount << '/' << population.size()
        << std::endl;  // 每轮刷新，运行过程中即可看到这一行。
}

struct RunResult {
    Individual bestEver;
    std::int64_t lastGeneration = 0;
    std::int64_t firstOptimalGeneration = -1;  // -1 表示尚未命中。
    std::int64_t stableGeneration = -1;        // -1 表示尚未确认稳定。
    std::int64_t consecutiveOptimal = 0;
};

// 搜索只接收精确最优“数值”作为判定基准，不接收精确最优染色体。
// population 按值传入，方便独立验证初始种群已最优等边界情况。
RunResult evolve(std::vector<Individual> population, const Settings& settings,
                 int optimalFitness, std::mt19937& rng, std::ostream& out) {
    RunResult result;
    for (std::int64_t generation = 0;; ++generation) {
        const GenerationStats stats = summarize(population);
        const Individual& best = population[stats.bestIndex];
        if (best.feasible() && better(best, result.bestEver)) result.bestEver = best;
        result.lastGeneration = generation;

        if (best.feasible() && best.fitness == optimalFitness) {
            if (result.firstOptimalGeneration < 0) result.firstOptimalGeneration = generation;
            ++result.consecutiveOptimal;
        } else {
            result.consecutiveOptimal = 0;
        }
        printGeneration(out, generation, population, stats);

        // 先检查成功，再检查上限：最后一轮刚好满足稳定条件，也算成功。
        if (result.consecutiveOptimal >= settings.stableRounds) {
            result.stableGeneration = generation;
            break;
        }
        if (generation >= settings.maxGenerations) break;
        population = nextGeneration(population, settings, rng);
    }
    return result;
}

void printSummary(const RunResult& result, const Settings& settings, int optimalFitness) {
    std::cout << "\n===== 实验结束 =====\n";
    if (result.stableGeneration >= 0) {
        std::cout << "结束原因：连续 " << settings.stableRounds << " 轮最好适应度等于精确最优值。\n"
                  << "首次找到最优解：第" << result.firstOptimalGeneration << "轮。\n"
                  << "确认稳定得到最优解：第" << result.stableGeneration << "轮。\n";
    } else if (result.firstOptimalGeneration >= 0) {
        std::cout << "结束原因：达到最大演化轮数，已经找到最优解，但尚未累计满 "
                  << settings.stableRounds << " 轮。\n"
                  << "首次找到最优解：第" << result.firstOptimalGeneration << "轮。\n"
                  << "当前连续最优轮数：" << result.consecutiveOptimal << "。\n";
    } else {
        std::cout << "结束原因：达到最大演化轮数，尚未找到精确最优解。\n";
    }
    std::cout << "实际演化轮数：" << result.lastGeneration << "（另含第0轮初始评价）。\n"
              << "历来最好可行方案：" << chromosomeString(result.bestEver) << '\n'
              << "选择物品：" << selectedItems(result.bestEver) << '\n'
              << "总重量：" << result.bestEver.weight << '/' << kCapacity << '\n'
              << "总价值 / 适应度：" << result.bestEver.value << '\n'
              << "与精确最优值的差距：" << optimalFitness - result.bestEver.value << '\n';
}

}  // namespace knapsack

// 在 Windows 中启用 UTF-8。只有输入和输出都连着真实控制台时才等待回车，
// 便于双击 exe 后查看结果；用管道或文件输入做测试时不会多等一次输入。
void pauseIfInteractive() {
#ifdef _WIN32
    DWORD inputMode = 0;
    DWORD outputMode = 0;
    if (GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &inputMode)
        && GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &outputMode)) {
        std::cout << "\n按回车键退出……" << std::flush;
        std::string ignored;
        std::getline(std::cin, ignored);
    }
#endif
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(&std::cout);
    try {
        using namespace knapsack;
        std::cout << "===== 遗传算法：固定的 0-1 背包问题 =====\n"
                  << "固定物品数：6；固定容量：15；精英保留：1。\n"
                  << "物品编号：1  2  3  4  5  6\n"
                  << "物品重量：2  3  4  5  7  9\n"
                  << "物品价值：6  5  8  9  13 17\n\n";

        const ExactResult exact = solveExactly();
        std::cout << "独立穷举：检查64种组合，可行组合为" << exact.feasibleCount
                  << "种，最优组合为" << exact.optimalCount << "种。\n"
                  << "精确最优字符串：" << chromosomeString(exact.best)
                  << "，选择物品：" << selectedItems(exact.best)
                  << "，总重量：" << exact.best.weight
                  << "，最优价值：" << exact.best.value << "。\n\n";

        Settings settings;
        if (!readSettings(settings)) {
            std::cout << "\n输入结束，程序退出。\n";
            return 0;
        }
        const std::uint32_t seed = resolveSeed(settings.seedInput);
        std::mt19937 rng(seed);
        std::cout << "\n实际随机种子：" << seed << "（下次输入此值可复现）。\n"
                  << "第0轮为初始种群；最多更新到第" << settings.maxGenerations << "轮。\n"
                  << "稳定标准：最好适应度连续" << settings.stableRounds
                  << "轮等于" << exact.best.value << "，不要求全体个体收敛。\n";
        if (static_cast<std::int64_t>(settings.stableRounds)
            > static_cast<std::int64_t>(settings.maxGenerations) + 1) {
            std::cout << "提示：含初始种群的总评价轮数少于K，本次最多能找到最优，无法确认稳定。\n";
        }
        std::cout << "\n===== 每轮种群结果 =====\n";
        const RunResult result = evolve(initializePopulation(settings, rng), settings,
                                        exact.best.value, rng, std::cout);
        printSummary(result, settings, exact.best.value);
        pauseIfInteractive();
        return 0;
    } catch (const std::bad_alloc&) {
        std::cerr << "\n内存不足，请减小种群规模后重新运行。\n";
    } catch (const std::exception& error) {
        std::cerr << "\n程序无法完成本次实验：" << error.what() << '\n';
    }
    pauseIfInteractive();
    return 1;
}
