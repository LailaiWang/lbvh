#ifndef __GEOM_CUH__
#define __GEOM_CUH__

#include "cuda_runtime.h"
#include <type_traits>
#include <limits>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

namespace GEOM {

typedef std::integral_constant<int, 1> bar_2_t;
typedef std::integral_constant<int, 3> quad_4_t;
typedef std::integral_constant<int, 2> tri_3_t;

typedef std::integral_constant<int, 2> dim_2_t;
typedef std::integral_constant<int, 3> dim_3_t;

template<typename F, typename E>
struct ray_t;

template<>
struct ray_t<float, dim_2_t> {
  float2 O;
  float2 D;
  using value_type = float;
};
template<>
struct ray_t<double, dim_2_t>{
  double2 O;
  double2 D;
  using value_type = double;
};

template<>
struct ray_t<float, dim_3_t> {
  float4 O;
  float4 D;

  using value_type = float;
};
template<>
struct ray_t<double, dim_3_t> {
  double4 O;
  double4 D;
  using value_type = double;
};

template<typename F, typename E>
struct element_t;

template< >
struct element_t<float, bar_2_t> {
  float4 coords;
  using elem_type = bar_2_t;
  using elem_dim = dim_2_t;
  using value_type = float;
};
template< >
struct element_t<double, bar_2_t> {
  double4 coords;
  using elem_type = bar_2_t;
  using elem_dim = dim_2_t;
  using value_type = double;
};

template< >
struct element_t<float, quad_4_t> {
  float4 coords[4];
  using elem_type = quad_4_t;
  using elem_dim = dim_3_t;
  using value_type = float;
};

template< >
struct element_t<double, quad_4_t> {
  double4 coords[4];
  using elem_type = quad_4_t;
  using elem_dim = dim_3_t;
  using value_type = double;
};

}

#endif
