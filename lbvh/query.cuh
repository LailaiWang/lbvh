#ifndef LBVH_QUERY_CUH
#define LBVH_QUERY_CUH
#include "predicator.cuh"
#include "utility.cuh"

namespace lbvh
{
/*
template<typename Real typename Objects, bool IsConst, typename OutputIterator>
__device__
unsigned int query_device(
    const detail::basic_device_bvh<Real, Objects, IsConst>& bvh
    const query_intersection<> q, OutputIterator outiter,
    const unsigned int max_buffer_size = 0xffffffff) noexcept 
{ 
    using bvh_type = detail::basic_device_bvh<Real, Objects, IsConst>;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;
    index_type stack[64];
    index_type* stack_ptr = stack;
    
    *stack_ptr++ = 0;
    unsigned int num_found = 0;
    do {
        const index_type node = *--stack_ptr;
        const index_type L_idx = bvh.nodes[node].left_idx;
        const index_type R_idx = bvh.nodes[node].right_idx;
        
        if(intersects_ray_box(), bvh.aabbs[L_idx]) {
            const auto obj_idx = bvh.nodes[L_idx].object_idx;
            if(obj_idx != 0xffffffff) {
                // this is a leaf node
                if(intersects_ray_surf, bv) { // check if the ray intersections with the surface
                    if(num_found < max_buffer_size) {
                        *outiter++ = obj_idx;
                    }
                }
            } else {
                *stack_ptr++ = L_idx;
            }
        }
        if(intersects_ray(), bvh.aabbs[R_idx]) {
            const auto obj_idx = bvh.nodes[R_idx].object_idx;
            if(obj_idx != 0xffffffff) {
              // this is a leaf node
              if(intersects_ray_surf) {
                if(num_found < max_buffer_size) {

                }
              }
            } else {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
    return num_found;
}
*/
// query object indices that potentially overlaps with query aabb.
//
// requirements:
// - OutputIterator should be writable and its object_type should be uint32_t
//
template<typename Real, typename Objects, bool IsConst, bvh_dim dim, typename OutputIterator>
__device__
unsigned int query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const query_overlap<Real, dim> q, OutputIterator outiter,
        const unsigned int max_buffer_size = 0xFFFFFFFF) noexcept
{
    using bvh_type   = detail::basic_device_bvh<Real, Objects, IsConst>;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;
    using real_type  = typename bvh_type::real_type;

    index_type  stack[64]; // is it okay?
    index_type* stack_ptr = stack;
    *stack_ptr++ = 0; // root node is always 0

    unsigned int num_found = 0;
    do
    {
        const index_type node  = *--stack_ptr;
        const index_type L_idx = bvh.nodes[node].left_idx;
        const index_type R_idx = bvh.nodes[node].right_idx;

        if(intersects<real_type, dim>(q.target, bvh.aabbs[L_idx]))
        {
            const auto obj_idx = bvh.nodes[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if(intersects<real_type, dim>(q.target, bvh.aabbs[R_idx]))
        {
            const auto obj_idx = bvh.nodes[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    }
    while (stack < stack_ptr);
    return num_found;
}

// query object index that is the nearst to the query point.
//
// requirements:
// - DistanceCalculator must be able to calc distance between a point to an object.
//
template<typename Real, typename Objects, bool IsConst, bvh_dim dim,
         typename DistanceCalculator>
__device__
thrust::pair<unsigned int, Real> query_device(
        const detail::basic_device_bvh<Real, Objects, IsConst>& bvh,
        const query_nearest<Real, dim>& q, DistanceCalculator calc_dist) noexcept
{
    using bvh_type   = detail::basic_device_bvh<Real, Objects, IsConst>;
    using real_type  = typename bvh_type::real_type;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    // pair of {node_idx, mindist}
    thrust::pair<index_type, real_type>  stack[64];
    thrust::pair<index_type, real_type>* stack_ptr = stack;
    *stack_ptr++ = thrust::make_pair(0, mindist<dim>(bvh.aabbs[0], q.target));

    unsigned int nearest = 0xFFFFFFFF;
    real_type dist_to_nearest_object = infinity<real_type>();
    do
    {
        const auto node  = *--stack_ptr;
        if(node.second > dist_to_nearest_object)
        {
            // if aabb mindist > already_found_mindist, it cannot have a nearest
            continue;
        }

        const index_type L_idx = bvh.nodes[node.first].left_idx;
        const index_type R_idx = bvh.nodes[node.first].right_idx;

        const aabb_type& L_box = bvh.aabbs[L_idx];
        const aabb_type& R_box = bvh.aabbs[R_idx];

        const real_type L_mindist = mindist<dim>(L_box, q.target);
        const real_type R_mindist = mindist<dim>(R_box, q.target);

        const real_type L_minmaxdist = minmaxdist<dim>(L_box, q.target);
        const real_type R_minmaxdist = minmaxdist<dim>(R_box, q.target);

        // there should be an object that locates within minmaxdist.

        if(L_mindist <= R_minmaxdist) // L is worth considering
        {
            const auto obj_idx = bvh.nodes[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF) // leaf node
            {
                const real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
                if(dist <= dist_to_nearest_object)
                {
                    dist_to_nearest_object = dist;
                    nearest = obj_idx;
                }
            }
            else
            {
                *stack_ptr++ = thrust::make_pair(L_idx, L_mindist);
            }
        }
        if(R_mindist <= L_minmaxdist) // R is worth considering
        {
            const auto obj_idx = bvh.nodes[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF) // leaf node
            {
                const real_type dist = calc_dist(q.target, bvh.objects[obj_idx]);
                if(dist <= dist_to_nearest_object)
                {
                    dist_to_nearest_object = dist;
                    nearest = obj_idx;
                }
            }
            else
            {
                *stack_ptr++ = thrust::make_pair(R_idx, R_mindist);
            }
        }
        assert(stack_ptr < stack + 64);
    }
    while (stack < stack_ptr);
    return thrust::make_pair(nearest, dist_to_nearest_object);
}

template<typename Real, typename Objects, typename AABBGetter, bvh_dim dim, 
         typename MortonCodeCalculator, typename OutputIterator>
unsigned int query_host(
    const bvh<Real, Objects, AABBGetter, dim, MortonCodeCalculator>& tree,
    const query_overlap<Real, dim> q, OutputIterator outiter,
    const unsigned int max_buffer_size = 0xFFFFFFFF)
{
    using bvh_type   = ::lbvh::bvh<Real, Objects, AABBGetter, dim, MortonCodeCalculator>;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    if(!tree.query_host_enabled())
    {
        throw std::runtime_error("lbvh::bvh query_host is not enabled");
    }

    std::vector<std::size_t> stack;
    stack.reserve(64);
    stack.push_back(0);

    unsigned int num_found = 0;
    do
    {
        const index_type node  = stack.back(); stack.pop_back();
        const index_type L_idx = tree.nodes_host()[node].left_idx;
        const index_type R_idx = tree.nodes_host()[node].right_idx;

        if(intersects<Real, dim>(q.target, tree.aabbs_host()[L_idx]))
        {
            const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                stack.push_back(L_idx);
            }
        }
        if(intersects<Real, dim>(q.target, tree.aabbs_host()[R_idx]))
        {
            const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF)
            {
                if(num_found < max_buffer_size)
                {
                    *outiter++ = obj_idx;
                }
                ++num_found;
            }
            else // the node is not a leaf.
            {
                stack.push_back(R_idx);
            }
        }
    }
    while (!stack.empty());
    return num_found;
}

template<typename Real, typename Objects, typename AABBGetter, bvh_dim dim,
         typename MortonCodeCalculator, typename DistanceCalculator>
std::pair<unsigned int, Real> query_host(
        const bvh<Real, Objects, AABBGetter, dim, MortonCodeCalculator>& tree,
        const query_nearest<Real, dim>& q, DistanceCalculator calc_dist) noexcept
{
    using bvh_type   = ::lbvh::bvh<Real, Objects, AABBGetter, dim, MortonCodeCalculator>;
    using real_type  = typename bvh_type::real_type;
    using index_type = typename bvh_type::index_type;
    using aabb_type  = typename bvh_type::aabb_type;
    using node_type  = typename bvh_type::node_type;

    if(!tree.query_host_enabled())
    {
        throw std::runtime_error("lbvh::bvh query_host is not enabled");
    }

    // pair of {node_idx, mindist}
    std::vector<std::pair<index_type, real_type>> stack = {
        {0, mindist(tree.aabbs_host()[0], q.target)}
    };
    stack.reserve(64);

    unsigned int nearest = 0xFFFFFFFF;
    real_type current_nearest_dist = infinity<real_type>();
    do
    {
        const auto node = stack.back(); stack.pop_back();
        if(node.second > current_nearest_dist)
        {
            // if aabb mindist > already_found_mindist, it cannot have a nearest
            continue;
        }

        const index_type L_idx = tree.nodes_host()[node.first].left_idx;
        const index_type R_idx = tree.nodes_host()[node.first].right_idx;

        const aabb_type& L_box = tree.aabbs_host()[L_idx];
        const aabb_type& R_box = tree.aabbs_host()[R_idx];

        const real_type L_mindist = mindist<dim>(L_box, q.target);
        const real_type R_mindist = mindist<dim>(R_box, q.target);

        const real_type L_minmaxdist = minmaxdist<dim>(L_box, q.target);
        const real_type R_minmaxdist = minmaxdist<dim>(R_box, q.target);

       // there should be an object that locates within minmaxdist.

        if(L_mindist <= R_minmaxdist) // L is worth considering
        {
            const auto obj_idx = tree.nodes_host()[L_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF) // leaf node
            {
                const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                if(dist <= current_nearest_dist)
                {
                    current_nearest_dist = dist;
                    nearest = obj_idx;
                }
            }
            else
            {
                stack.emplace_back(L_idx, L_mindist);
            }
        }
        if(R_mindist <= L_minmaxdist) // R is worth considering
        {
            const auto obj_idx = tree.nodes_host()[R_idx].object_idx;
            if(obj_idx != 0xFFFFFFFF) // leaf node
            {
                const real_type dist = calc_dist(q.target, tree.objects_host()[obj_idx]);
                if(dist <= current_nearest_dist)
                {
                    current_nearest_dist = dist;
                    nearest = obj_idx;
                }
            }
            else
            {
                stack.emplace_back(R_idx, R_mindist);
            }
        }
    }
    while (!stack.empty());
    return std::make_pair(nearest, current_nearest_dist);
}
} // lbvh
#endif// LBVH_QUERY_CUH
