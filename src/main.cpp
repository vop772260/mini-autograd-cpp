#include <iostream>
#include <chrono>
#include "Autograd.hpp"

void run_simple_test() {
    std::cout << "=== 1. Test Tinh Đao Ham Nguoc (Autograd Engine) ===\n";
    // Biểu thức: L = (a * b) + ReLU(c)
    auto a = Value::create(2.0);
    auto b = Value::create(-3.0);
    auto c = Value::create(4.0);

    auto ab = a->mul(b);
    auto relu_c = c->relu();
    auto L = ab->add(relu_c);

    L->backward();

    std::cout << "Gia tri L = " << L->data << "\n";
    std::cout << "dL/da = " << a->grad << " (Expected: -3.0)\n";
    std::cout << "dL/db = " << b->grad << " (Expected: 2.0)\n";
    std::cout << "dL/dc = " << c->grad << " (Expected: 1.0)\n\n";
}

void run_performance_benchmark() {
    std::cout << "=== 2. Empirical Benchmark: Topological Traversal & Gradient Computation ===\n";
    int depths[] = {1000, 5000, 10000, 50000};

    for (int depth : depths) {
        auto start = std::chrono::high_resolution_clock::now();
        
        auto root = Value::create(1.0);
        auto curr = root;
        for (int i = 0; i < depth; ++i) {
            auto param = Value::create(1.001);
            curr = curr->mul(param);
        }

        curr->backward();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Graph Depth = " << depth 
                  << " nodes | Time: " << duration.count() << " ms\n";
    }
}

int main() {
    run_simple_test();
    run_performance_benchmark();
    return 0;
}