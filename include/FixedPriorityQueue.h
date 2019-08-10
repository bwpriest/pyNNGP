#ifndef FixedPriorityQueue_h
#define FixedPriorityQueue_h
// #pragma once

// #include "DistFunc.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace pyNNGP {

template <typename T>
struct operand {
  T      obj;
  double val;
};

template <typename T>
bool operator<(const operand<T> &lhs, const operand<T> &rhs) {
  if (lhs.obj == rhs.obj) { return false; }
  if (lhs.val < rhs.val) { return true; }
  if (lhs.val > rhs.val) { return false; }
  return lhs.obj < rhs.obj;
}
template <typename T>
bool operator<(const operand<T> &lhs, const double &rhs) {
  return lhs.val < rhs;
}
template <typename T>
bool operator<(const double &lhs, const operand<T> &rhs) {
  return lhs < rhs.val;
}
template <typename T>
bool operator>(const operand<T> &lhs, const operand<T> &rhs) {
  if (lhs.obj == rhs.obj) { return false; }
  if (lhs.val > rhs.val) { return true; }
  if (lhs.val < rhs.val) { return false; }
  return lhs.obj > rhs.obj;
}
template <typename T>
bool operator>(const operand<T> &lhs, const double &rhs) {
  return lhs.val > rhs;
}
template <typename T>
bool operator>(const double &lhs, const operand<T> &rhs) {
  return lhs > rhs.val;
}
template <typename T>
bool operator==(const operand<T> &lhs, const operand<T> &rhs) {
  return lhs.obj == rhs.obj;
}
template <typename T>
std::ostream &operator<<(std::ostream &os, const operand<T> &op) {
  os << op.obj << " " << op.val;
  return os;
}

template <typename T>
struct sorted_vector {
  typedef operand<T>                         op;
  typedef std::vector<op>                    vec;
  typedef typename std::vector<op>::iterator iterator;
  // CompFunc &                                 cf;
  const int m_capacity;
  iterator  begin() { return std::begin(V); }
  iterator  end() { return std::end(V); }

  // sorted_vector(const std::size_t k, const CompFunc &_cf) : cf(_cf), k(k) {
  //   V.reserve(k);
  // }
  sorted_vector(const std::size_t k) : m_capacity(k) { V.reserve(m_capacity); }

  const op &operator[](const int idx) const {
    if (idx < V.size() && idx >= 0) {
      return V[idx];
    } else {
      throw std::out_of_range("Index out of range");
    }
  }

  void insert(const op &t) {
    if (V.size() == 0) {
      V.push_back(t);
      return;
    }
    op back = V.back();
    if (V.size() < m_capacity) {
      // if (cf(t.val, back.val) {
      if (back.val < t.val) {
        V.push_back(t);
      } else {
        if (V.size() == 1) {
          V[0] = t;
        } else {
          shift(t, V.size());
        }
        V.push_back(back);
      }
    } else {
      // if (cf(t.val, bacl.val)) { shift(t, capacity); }
      if (t.val < back.val) { shift(t, m_capacity); }
    }
  }

  std::size_t size() const { return V.size(); }

  template <typename U>
  friend std::ostream &operator<<(std::ostream &, const sorted_vector<U> &);

 private:
  vec V;

  void shift(const op &t, const int size) {
    for (int j = size - 1; j > 0; --j) {
      op temp = V[j];
      // if (cf(t.val, temp.val)) {
      if (V[j - 1].val < t.val && t.val < V[j]) {
        V[j] = t;
        break;
      } else {
        V[j] = V[j - 1];
      }
      if (j == 1) { V[0] = t; }
    }
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const sorted_vector<T> &svec) {
  os << "\t";
  for (int i = 0; i < svec.size(); ++i) {
    os << svec[i] << "\n";
    if (i < svec.size() - 1) { os << "\t"; }
  }
  return os;
}

template <typename OBJ>
class FixedPriorityQueue {
  typedef operand<OBJ>       oper;
  typedef sorted_vector<OBJ> oper_svec;

 public:
  // FixedPriorityQueue(const std::size_t k, const CompFunc &_cf)
  //     : m_k(k), cf(_cf) {}
  FixedPriorityQueue(const std::size_t k) : m_svec(k) {}

  // Assignment operators
  FixedPriorityQueue(const FixedPriorityQueue &) = default;
  FixedPriorityQueue(FixedPriorityQueue &&)      = default;
  FixedPriorityQueue &operator=(FixedPriorityQueue &) = default;
  FixedPriorityQueue &operator=(const FixedPriorityQueue &) = default;

  ~FixedPriorityQueue() {}

  // CompFunc &cf;

  const oper &operator[](const int idx) const { return m_svec[idx]; }

  inline void enqueue(const OBJ &obj, const double val) {
    oper ins({obj, val});
    m_svec.insert(ins);
  }

  inline std::size_t size() const { return m_svec.size(); }
  inline std::size_t capacity() const { return m_svec.m_capacity; }

  template <typename T>
  friend std::ostream &operator<<(std::ostream &,
                                  const FixedPriorityQueue<T> &);

 private:
  // std::size_t m_k;
  oper_svec m_svec;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const FixedPriorityQueue<T> &fpq) {
  os << "\t";
  for (int i = 0; i < fpq.size(); ++i) {
    os << fpq[i] << "\n";
    if (i < fpq.size() - 1) { os << "\t"; }
  }
  return os;
}

template <typename K, typename V>
V get_with_default(const std::map<K, V> &mp, const K &key, const V &defval) {
  typename std::map<K, V>::const_iterator it = mp.find(key);
  if (it == std::end(mp)) {
    return defval;
  } else {
    return it->second;
  }
}

}  // namespace pyNNGP
#endif