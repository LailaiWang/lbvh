#ifndef LBVH_PREDICATOR_CUH
#define LBVH_PREDICATOR_CUH
#include "aabb.cuh"

namespace lbvh
{

template<typename Real, typename Ray, bvh_dim dim>
struct query_ray_bvh_intersection {
    __device__ __host__
    query_ray_bvh_intersection (const Ray& ray) : ray_{ray} {}

    __device__ __host__ 
    inline bool operator() (const aabb<Real>& box) noexcept
    {
        return ray_bvh_intersects<dim>(box, ray_);
    }
    // this stores the ray
    Ray ray_;
};

template<typename Real, typename Ray, bvh_dim dim>
__device__ __host__
query_ray_bvh_intersection<Real, Ray, dim> ray_bvh_intersection(const Ray& ray) noexcept 
{
    return query_ray_bvh_intersection<Real,Ray,dim>(ray);
}


template<typename Real, bvh_dim dim>
struct query_overlap
{
    __device__ __host__
    query_overlap(const aabb<Real>& tgt): target(tgt) {}

    query_overlap()  = default;
    ~query_overlap() = default;
    query_overlap(const query_overlap&) = default;
    query_overlap(query_overlap&&)      = default;
    query_overlap& operator=(const query_overlap&) = default;
    query_overlap& operator=(query_overlap&&)      = default;

    __device__ __host__
    inline bool operator()(const aabb<Real>& box) noexcept
    {
        return intersects<dim>(box, target);
    }

    aabb<Real> target;
};

template<typename Real, bvh_dim dim>
__device__ __host__
query_overlap<Real, dim> overlaps(const aabb<Real>& region) noexcept
{
    return query_overlap<Real, dim>(region);
}

template<typename Real, bvh_dim dim>
struct query_nearest
{
    // float4/double4
    using vector_type = typename vector_of<Real>::type;

    __device__ __host__
    query_nearest(const vector_type& tgt): target(tgt) {}

    query_nearest()  = default;
    ~query_nearest() = default;
    query_nearest(const query_nearest&) = default;
    query_nearest(query_nearest&&)      = default;
    query_nearest& operator=(const query_nearest&) = default;
    query_nearest& operator=(query_nearest&&)      = default;

    vector_type target;
};

template<bvh_dim dim>
__device__ __host__
inline query_nearest<float, dim> nearest(const float4& point) noexcept
{
    return query_nearest<float, dim>(point);
}
template<bvh_dim dim>
__device__ __host__
inline query_nearest<float, dim> nearest(const float3& point) noexcept
{
    return query_nearest<float, dim>(make_float4(point.x, point.y, point.z, 0.0f));
}
template<bvh_dim dim>
__device__ __host__
inline query_nearest<double, dim> nearest(const double4& point) noexcept
{
    return query_nearest<double, dim>(point);
}
template<bvh_dim dim>
__device__ __host__
inline query_nearest<double, dim> nearest(const double3& point) noexcept
{
    return query_nearest<double, dim>(make_double4(point.x, point.y, point.z, 0.0));
}

} // lbvh
#endif// LBVH_PREDICATOR_CUH
