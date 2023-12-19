#ifndef LBVH_AABB_CUH
#define LBVH_AABB_CUH
#include "utility.cuh"
#include "geom.cuh"
#include <thrust/swap.h>
#include <cmath>

namespace lbvh
{

template<typename T>
struct aabb
{
    typename vector_of<T>::type upper;
    typename vector_of<T>::type lower;
    using value_type = T;
};

template<typename F, typename E>
struct aabb_getter;

template<typename F>
struct aabb_getter<F, GEOM::bar_2_t> {
  __device__ __host__
  lbvh::aabb<F> operator() (const GEOM::element_t<F, GEOM::bar_2_t> object) const noexcept {
    lbvh::aabb<F> ret;
    F xmin{0}, ymin{0}, xmax{0}, ymax{0};
    if constexpr (std::is_same<F, float>::value) {
      xmin = ::fminf(object.coords.x, object.coords.z);
      ymin = ::fminf(object.coords.y, object.coords.w);
      xmax = ::fmaxf(object.coords.x, object.coords.z);
      ymax = ::fmaxf(object.coords.y, object.coords.w);
      ret.upper = make_float4(xmax, ymax, 0, 0);
      ret.lower = make_float4(xmin, ymin, 0, 0);
    } else 
    if constexpr (std::is_same<F, double>::value) {
      xmin = ::fmin(object.coords.x, object.coords.z);
      ymin = ::fmin(object.coords.y, object.coords.w);
      xmax = ::fmax(object.coords.x, object.coords.z);
      ymax = ::fmax(object.coords.y, object.coords.w);
      ret.upper = make_double4(xmax, ymax, 0, 0);
      ret.lower = make_double4(xmin, ymin, 0, 0);
    }
    return ret;
  }
};

template<typename F>
struct aabb_getter<F, GEOM::quad_4_t> {
  __device__ __host__
  lbvh::aabb<F> operator() (const GEOM::element_t<F, GEOM::quad_4_t> object) const noexcept {
    lbvh::aabb<F> ret;
    F xmin{ 1e+20};
    F ymin{ 1e+20};
    F zmin{ 1e+20};
    F xmax{-1e+20};
    F ymax{-1e+20};
    F zmax{-1e+20};
    if constexpr (std::is_same<F, float>::value) {
      #pragma unroll
      for(int k=0;k<4;++k) {
        xmin = ::fminf(xmin, object.coords[k].x);
        ymin = ::fminf(ymin, object.coords[k].y);
        zmin = ::fminf(zmin, object.coords[k].z);
        xmax = ::fmaxf(xmax, object.coords[k].x);
        ymax = ::fmaxf(ymax, object.coords[k].y);
        zmax = ::fmaxf(zmax, object.coords[k].z);
      }
      ret.upper = make_float4(xmax,ymax,zmax,0);
      ret.lower = make_float4(xmin,ymin,zmin,0);
    } else 
    if constexpr (std::is_same<F, double>::value) {
      #pragma unroll
      for(int k=0;k<4;++k) {
        xmin = ::fmin(xmin, object.coords[k].x);
        ymin = ::fmin(ymin, object.coords[k].y);
        zmin = ::fmin(zmin, object.coords[k].z);
        xmax = ::fmax(xmax, object.coords[k].x);
        ymax = ::fmax(ymax, object.coords[k].y);
        zmax = ::fmax(zmax, object.coords[k].z);
      }
      ret.upper = make_float4(xmax,ymax,zmax,0);
      ret.lower = make_float4(xmin,ymin,zmin,0);
    }
    return ret;
  }
};

template<typename T, bvh_dim dim>
__device__ __host__
inline bool intersects(const aabb<T>& lhs, const aabb<T>& rhs) noexcept
{
    if(lhs.upper.x < rhs.lower.x || rhs.upper.x < lhs.lower.x) {return false;}
    if(lhs.upper.y < rhs.lower.y || rhs.upper.y < lhs.lower.y) {return false;}
    if constexpr (dim == bvh_dim::three) {
        if(lhs.upper.z < rhs.lower.z || rhs.upper.z < lhs.lower.z) {return false;}
    }
    return true;
}

template<typename T, typename Ray, typename Object,
         std::enable_if_t<std::is_same<typename Object::elem_type, GEOM::bar_2_t>::value, bool> = true
        >
__device__ __host__
inline bool ray_surf_intersects(Object& obj, Ray& ray, T& alpha, T& beta) {
    using T1 = typename Ray::value_type;
    using T2 = typename Object::value_type;
    
    static_assert(std::is_same<T1,T2>::value, "Float type not same in Ray and Object!");
    bool ans = false;

    T x0 = obj.coords.x, y0 = obj.coords.y;
    T x1 = obj.coords.z, y1 = obj.coords.w;
    // check if parallel first
    T dx = x1 - x0, dy = y1 - y0;
    T dox = ray.O.x - x0, doy = ray.O.y - y0;
    // line 1 = (x0, y0) + alpha * (dx, dy)
    // line 2 = (O.x, O.y) + alpha * (D.x + D.y)
    // equation to be solved is 
    //  |dx  -D.x|  |\alpha| = O.x - x0
    //  |dy  -D.y|  |\beta | = O.y - y0
    //  inverse of the matrix is 
    //  1/J | -D.y D.x|
    //      | -dy  dx |
    T J = - dx * ray.D.y + dy * ray.D.x;
    if (J == 0.0) {
      return false;
    }
    alpha = (-ray.D.y * dox + ray.D.x * doy)/J;
    beta  = (-dy * dox + dx * doy )/J;
    if (alpha>=0.0 && alpha <= 1.0) {
      ans = true;
    }
    return ans;
}

template<typename T, typename Object, typename Ray,
         std::enable_if_t<std::is_same<typename Object::elem_type,GEOM::quad_4_t>::value,bool> = true
        >
__device__ __host__
inline bool ray_surf_intersects() {
    bool ans = false;
    return ans;
}

template<typename T, typename Ray, typename Object,
         std::enable_if_t<std::is_same<typename Object::elem_type,GEOM::tri_3_t>::value,bool> = true
        > 
__device__ __host__
inline bool ray_surf_intersects() {
    bool ans = false;
    if constexpr (std::is_same<T, float>::value) {

    } else 
    if constexpr (std::is_same<T, double>::value) {

    }
    return ans;
}

// type Ray has a memer O as origin, a memer D as the direction
template<typename T, typename Ray, bvh_dim dim>
__device__ __host__
inline bool ray_bvh_intersects(const aabb<T>& box, const Ray& ray) noexcept
{
    // use slab test to check if the ray intersect with the BVH bounding box
    if constexpr (dim == bvh_dim::two) {
        T tx1 = (box.lower.x - ray.O.x) / ray.D.x;
        T tx2 = (box.upper.x - ray.O.x) / ray.D.x;
        T tmin = 0;
        T tmax = 0;
        if constexpr (std::is_same<T, float>::value) {
          tmin = ::fminf( tx1, tx2 );
          tmax = ::fmaxf( tx1, tx2 );
        } else 
        if constexpr (std::is_same<T, double>::value) {
          tmin = ::fmin( tx1, tx2 );
          tmax = ::fmax( tx1, tx2 );
        }
        T ty1 = (box.lower.y - ray.O.y) / ray.D.y;
        T ty2 = (box.upper.y - ray.O.y) / ray.D.y;
        if constexpr (std::is_same<T, float>::value) {
          tmin = ::fmaxf( tmin, fminf( ty1, ty2 ) );
          tmax = ::fminf( tmax, fmaxf( ty1, ty2 ) );
        } else 
        if constexpr (std::is_same<T, double>::value) {
          tmin = ::fmax( tmin, fmin( ty1, ty2 ) );
          tmax = ::fmin( tmax, fmax( ty1, ty2 ) );
        }
        return tmax >= tmin && tmax > 0;
    } else 
    if constexpr (dim == bvh_dim::two) {
        T tx1 = (box.lower.x - ray.O.x) / ray.D.x;
        T tx2 = (box.upper.x - ray.O.x) / ray.D.x;
        T tmin = 0;
        T tmax = 0;
        if constexpr (std::is_same<T, float>::value) {
          tmin = ::fminf( tx1, tx2 );
          tmax = ::fmaxf( tx1, tx2 );
        } else 
        if constexpr (std::is_same<T, double>::value) {
          tmin = ::fmin( tx1, tx2 );
          tmax = ::fmax( tx1, tx2 );
        }
        T ty1 = (box.lower.y - ray.O.y) / ray.D.y;
        T ty2 = (box.upper.y - ray.O.y) / ray.D.y;
        if constexpr (std::is_same<T, float>::value) {
          tmin = ::fmaxf( tmin, fminf( ty1, ty2 ) );
          tmax = ::fminf( tmax, fmaxf( ty1, ty2 ) );
        } else 
        if constexpr (std::is_same<T, double>::value) {
          tmin = ::fmax( tmin, fmin( ty1, ty2 ) );
          tmax = ::fmin( tmax, fmax( ty1, ty2 ) );
        }
        T tz1 = (box.lower.z - ray.O.z) / ray.D.z;
        T tz2 = (box.upper.z - ray.O.z) / ray.D.z;
        if constexpr (std::is_same<T, float>::value) {
          tmin = ::fmaxf( tmin, fminf( tz1, tz2 ) );
          tmax = ::fminf( tmax, fmaxf( tz1, tz2 ) );
        } else 
        if constexpr (std::is_same<T, float>::value) {
          tmin = ::fmax( tmin, fmin( tz1, tz2 ) );
          tmax = ::fmin( tmax, fmax( tz1, tz2 ) );
        }
        return tmax >= tmin && tmax > 0;
    }

}


template<bvh_dim dim>
__device__ __host__
inline aabb<double> merge(const aabb<double>& lhs, const aabb<double>& rhs) noexcept
{
    aabb<double> merged;
    merged.upper.x = ::fmax(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmax(lhs.upper.y, rhs.upper.y);
    if constexpr (dim == bvh_dim::three) 
        merged.upper.z = ::fmax(lhs.upper.z, rhs.upper.z);
    merged.lower.x = ::fmin(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fmin(lhs.lower.y, rhs.lower.y);
    if constexpr (dim == bvh_dim::three) 
        merged.lower.z = ::fmin(lhs.lower.z, rhs.lower.z);
    return merged;
}

template<bvh_dim dim>
__device__ __host__
inline aabb<float> merge(const aabb<float>& lhs, const aabb<float>& rhs) noexcept
{
    aabb<float> merged;
    merged.upper.x = ::fmaxf(lhs.upper.x, rhs.upper.x);
    merged.upper.y = ::fmaxf(lhs.upper.y, rhs.upper.y);
    if constexpr (dim == bvh_dim::three)
        merged.upper.z = ::fmaxf(lhs.upper.z, rhs.upper.z);
    merged.lower.x = ::fminf(lhs.lower.x, rhs.lower.x);
    merged.lower.y = ::fminf(lhs.lower.y, rhs.lower.y);
    if constexpr (dim == bvh_dim::three)
        merged.lower.z = ::fminf(lhs.lower.z, rhs.lower.z);
    return merged;
}

// metrics defined in
// Nearest Neighbor Queries (1995) ACS-SIGMOD
// - Nick Roussopoulos, Stephen Kelley FredericVincent

template<bvh_dim dim>
__device__ __host__
inline float mindist(const aabb<float>& lhs, const float4& rhs) noexcept
{
    const float dx = ::fminf(lhs.upper.x, ::fmaxf(lhs.lower.x, rhs.x)) - rhs.x;
    const float dy = ::fminf(lhs.upper.y, ::fmaxf(lhs.lower.y, rhs.y)) - rhs.y;
    if constexpr (dim == bvh_dim::three) {
        const float dz = ::fminf(lhs.upper.z, ::fmaxf(lhs.lower.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    } else {
        return dx * dx + dy * dy;
    }
    return 0.0f;
}

template<bvh_dim dim>
__device__ __host__
inline double mindist(const aabb<double>& lhs, const double4& rhs) noexcept
{
    const double dx = ::fmin(lhs.upper.x, ::fmax(lhs.lower.x, rhs.x)) - rhs.x;
    const double dy = ::fmin(lhs.upper.y, ::fmax(lhs.lower.y, rhs.y)) - rhs.y;
    if constexpr (dim == bvh_dim::three) {
        const double dz = ::fmin(lhs.upper.z, ::fmax(lhs.lower.z, rhs.z)) - rhs.z;
        return dx * dx + dy * dy + dz * dz;
    } else {
        return dx * dx + dy * dy;
    }
    return 0.0;
}

template<bvh_dim dim>
__device__ __host__
inline float minmaxdist(const aabb<float>& lhs, const float4& rhs) noexcept
{ 
    if constexpr (dim == bvh_dim::three) {
        float3 rm_sq = make_float3((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x),
                                   (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
                                   (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z));
        float3 rM_sq = make_float3((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x),
                                   (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
                                   (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z));
        if((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        if((lhs.upper.z + lhs.lower.z) * 0.5f < rhs.z)
        {
            thrust::swap(rm_sq.z, rM_sq.z);
        }
        
        const float dx = rm_sq.x + rM_sq.y + rM_sq.z;
        const float dy = rM_sq.x + rm_sq.y + rM_sq.z;
        const float dz = rM_sq.x + rM_sq.y + rm_sq.z;
        return ::fminf(dx, ::fminf(dy, dz));
    } else {
        float2 rm_sq = make_float2((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x),
                                   (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y));
                               
        float2 rM_sq = make_float2((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x),
                                   (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y));
        if((lhs.upper.x + lhs.lower.x) * 0.5f < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if((lhs.upper.y + lhs.lower.y) * 0.5f < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        
        const float dx = rm_sq.x + rM_sq.y;
        const float dy = rM_sq.x + rm_sq.y;
        return ::fminf(dx, dy);
    }
}

template<bvh_dim dim>
__device__ __host__
inline double minmaxdist(const aabb<double>& lhs, const double4& rhs) noexcept
{ 
    if constexpr (dim == bvh_dim::three) {
        double3 rm_sq = make_double3((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x),
                                     (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y),
                                     (lhs.lower.z - rhs.z) * (lhs.lower.z - rhs.z));
        double3 rM_sq = make_double3((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x),
                                     (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y),
                                     (lhs.upper.z - rhs.z) * (lhs.upper.z - rhs.z));
      
        if((lhs.upper.x + lhs.lower.x) * 0.5 < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if((lhs.upper.y + lhs.lower.y) * 0.5 < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        if((lhs.upper.z + lhs.lower.z) * 0.5 < rhs.z)
        {
            thrust::swap(rm_sq.z, rM_sq.z);
        }
      
        const double dx = rm_sq.x + rM_sq.y + rM_sq.z;
        const double dy = rM_sq.x + rm_sq.y + rM_sq.z;
        const double dz = rM_sq.x + rM_sq.y + rm_sq.z;
        return ::fmin(dx, ::fmin(dy, dz));
    } else {
        double2 rm_sq = make_double2((lhs.lower.x - rhs.x) * (lhs.lower.x - rhs.x),
                                     (lhs.lower.y - rhs.y) * (lhs.lower.y - rhs.y));
        double2 rM_sq = make_double2((lhs.upper.x - rhs.x) * (lhs.upper.x - rhs.x),
                                     (lhs.upper.y - rhs.y) * (lhs.upper.y - rhs.y));
      
        if((lhs.upper.x + lhs.lower.x) * 0.5 < rhs.x)
        {
            thrust::swap(rm_sq.x, rM_sq.x);
        }
        if((lhs.upper.y + lhs.lower.y) * 0.5 < rhs.y)
        {
            thrust::swap(rm_sq.y, rM_sq.y);
        }
        const double dx = rm_sq.x + rM_sq.y;
        const double dy = rM_sq.x + rm_sq.y;
        return ::fmin(dx, dy);
    }
}

template<typename T>
__device__ __host__
inline typename vector_of<T>::type centroid(const aabb<T>& box) noexcept
{
    typename vector_of<T>::type c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}


} // lbvh
#endif// LBVH_AABB_CUH
