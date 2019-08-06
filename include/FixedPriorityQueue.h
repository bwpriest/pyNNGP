#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
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

template <typename OBJ>
class FixedPriorityQueue {
  typedef operand<OBJ>                oper;
  typedef std::set<oper>              oper_set;
  typedef typename oper_set::iterator oper_it;

 public:
  FixedPriorityQueue(const std::size_t k) : m_k(k) {}

  // Assignment operators
  FixedPriorityQueue(const FixedPriorityQueue &) = default;
  FixedPriorityQueue(FixedPriorityQueue &&)      = default;
  FixedPriorityQueue &operator=(FixedPriorityQueue &) = default;
  FixedPriorityQueue &operator=(const FixedPriorityQueue &) = default;

  ~FixedPriorityQueue() {}

  inline void enqueue(const OBJ &obj, const double val) {
    if (val < 0) { return; }
    oper ins({obj, val});
    if (m_set.size() < m_k) {
      m_set.insert(ins);
      return;
    }
    if (*std::prev(std::end(m_set)) > val) {
      m_set.erase(std::prev(std::end(m_set)));
      m_set.insert(ins);
    }
  }

  inline std::size_t size() const { return m_set.size(); }
  inline std::size_t capacity() const { return m_set.m_k; }

  inline std::vector<oper> get_pairs() const { return get_pairs(m_set.size()); }
  inline std::vector<oper> get_pairs(int n) const {
    if (n > m_set.size()) { n = m_set.size(); }
    std::vector<oper> ret;
    ret.reserve(n);
    oper_it it(std::begin(m_set));
    for (int i(0); i < n; ++i) {
      ret.push_back(*it);
      ++it;
    }
    return ret;
  }

  inline std::map<OBJ, double> get_map() const { return get_map(m_set.size()); }
  inline std::map<OBJ, double> get_map(int n) const {
    if (n > m_set.size()) { n = m_set.size(); }
    std::map<OBJ, double> ret;
    oper_it               it(std::begin(m_set));
    for (int i(0); i < n; ++i) {
      ret[it->obj] = it->val;
      ++it;
    }
    return ret;
  }

  inline std::vector<OBJ> get_obj_vec() const {
    return get_obj_vec(m_set.size());
  }
  inline std::vector<OBJ> get_obj_vec(int n) const {
    if (n > m_set.size()) { n = m_set.size(); }
    std::vector<OBJ> ret(n);
    oper_it          it(std::begin(m_set));
    for (int i(0); i < n; ++i) {
      ret[i] = it->obj;
      ++it;
    }
    return ret;
  }

  inline std::vector<double> get_val_vec() const {
    return get_val_vec(m_set.size());
  }
  inline std::vector<double> get_val_vec(int n) const {
    if (n > m_set.size()) { n = m_set.size(); }
    std::vector<double> ret(n);
    oper_it             it(std::begin(m_set));
    for (int i(0); i < n; ++i) {
      ret[i] = it->val;
      ++it;
    }
    return ret;
  }

  inline std::vector<oper> get_pairs_from_set(std::set<OBJ> &objs) {
    std::vector<oper> ret;
    ret.reserve(objs.size());
    for (const oper &elt : m_set) {
      oper_it it(std::begin(objs));
      oper_it end_it(std::end(objs));
      while (it != end_it) {
        if (*it == elt.obj) {
          ret.push_back(elt);
          objs.erase(it);
          break;
        } else {
          ++it;
        }
      }
    }
    return ret;
  }

  inline double min_val() const { return std::prev(std::end(m_set))->val; }
  inline double min_val(int offset) const {
    return std::next(std::begin(m_set), offset - 1)->val;
  }
  inline double max_val() const { return std::begin(m_set)->val; }

  inline oper min_element() const { return *std::prev(std::end(m_set)); }
  inline oper min_element(int offset) const {
    return *(std::next(std::begin(m_set), offset - 1));
  }
  inline oper   max_element() const { return *std::begin(m_set); }
  inline double min() const { return min_element().val; }
  inline double min(int offset) const { return min_element(offset).val; }
  inline double max() const { return max_element().val; }
  inline void   erase_max_element() { m_set.erase(std::begin(m_set)); }

  //   inline auto begin() { return std::begin(m_set); }
  //   inline auto end() { return std::end(m_set); }
  //   inline auto cbegin() { return std::cbegin(m_set); }
  //   inline auto cend() { return std::cend(m_set); }

  template <typename T>
  friend std::ostream &operator<<(std::ostream &,
                                  const FixedPriorityQueue<T> &);

 private:
  std::size_t m_k;
  oper_set    m_set;
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const FixedPriorityQueue<T> &fpq) {
  for (auto elem : fpq.m_set) { os << elem << std::endl; }
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
