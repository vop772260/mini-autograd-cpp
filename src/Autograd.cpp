#include "Autograd.hpp"

std::shared_ptr<Value> Value::add(std::shared_ptr<Value> other) {
    auto out = std::make_shared<Value>(this->data + other->data, std::vector<std::shared_ptr<Value>>{shared_from_this(), other});
    out->_backward = [this, other, out]() {
        this->grad += 1.0 * out->grad;
        other->grad += 1.0 * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::mul(std::shared_ptr<Value> other) {
    auto out = std::make_shared<Value>(this->data * other->data, std::vector<std::shared_ptr<Value>>{shared_from_this(), other});
    out->_backward = [this, other, out]() {
        this->grad += other->data * out->grad;
        other->grad += this->data * out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::relu() {
    auto out = std::make_shared<Value>(this->data > 0 ? this->data : 0.0, std::vector<std::shared_ptr<Value>>{shared_from_this()});
    out->_backward = [this, out]() {
        this->grad += (this->data > 0 ? 1.0 : 0.0) * out->grad;
    };
    return out;
}

void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<Value*> visited;

    std::function<void(std::shared_ptr<Value>)> build_topo = [&](std::shared_ptr<Value> v) {
        if (visited.find(v.get()) == visited.end()) {
            visited.insert(v.get());
            for (auto& child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this());

    this->grad = 1.0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}