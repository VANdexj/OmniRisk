"""TensorRT-safe policy invocation helpers."""


def forward_with_contiguous_inputs(policy, depth, mask, velocity, obs):
    """Keep contiguous input buffers alive until the policy call returns."""
    inputs = tuple(value.contiguous() for value in (depth, mask, velocity, obs))
    if inputs[0].data_ptr() == inputs[1].data_ptr():
        raise RuntimeError('depth and mask input buffers must not alias')
    return policy(*inputs)
