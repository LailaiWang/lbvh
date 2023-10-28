#ifndef LBVH_PREDICATOR_CUH
#define LBVH_PREDICATOR_CUH
#include "aabb.cuh"

namespace lbvh
{

template<typename Real, bvhdim mydim = bvhdim::three>
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
        return intersects<mydim>(box, target);
    }

    aabb<Real> target;
};

template<typename Real, bvhdim mydim = bvhdim::three>
__device__ __host__
query_overlap<Real, mydim> overlaps(const aabb<Real>& region) noexcept
{
    return query_overlap<Real, mydim>(region);
}

template<typename Real, bvhdim mydim = bvhdim::three>
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

template<bvhdim mydim = bvhdim::three>
__device__ __host__
inline query_nearest<float, mydim> nearest(const float4& point) noexcept
{
    return query_nearest<float, mydim>(point);
}
template<bvhdim mydim = bvhdim::three>
__device__ __host__
inline query_nearest<float, mydim> nearest(const float3& point) noexcept
{
    return query_nearest<float, mydim>(make_float4(point.x, point.y, point.z, 0.0f));
}
template<bvhdim mydim = bvhdim::three>
__device__ __host__
inline query_nearest<double, mydim> nearest(const double4& point) noexcept
{
    return query_nearest<double, mydim>(point);
}
template<bvhdim mydim = bvhdim::three>
__device__ __host__
inline query_nearest<double, mydim> nearest(const double3& point) noexcept
{
    return query_nearest<double, mydim>(make_double4(point.x, point.y, point.z, 0.0));
}

} // lbvh
#endif// LBVH_PREDICATOR_CUH
