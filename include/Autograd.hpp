#ifndef AUTOGRAD_HPP
#define AUTOGRAD_HPP

#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <iostream>

class Value : public std::enable_shared_from_this<Value> {
public:
    double data;
    double grad = 0.0;
    std::vector<std::shared_ptr<Value>> prev;
    std::function<void()> _backward = []() {};

    explicit Value(double val, std::vector<std::shared_ptr<Value>> children = {})
        : data(val), prev(children) {}

    static std::shared_ptr<Value> create(double val) {
        return std::make_shared<Value>(val);
    }

    std::shared_ptr<Value> add(std::shared_ptr<Value> other);
    std::shared_ptr<Value> mul(std::shared_ptr<Value> other);
    std::shared_ptr<Value> relu();

    void backward();
};

#endif