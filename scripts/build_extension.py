from torch.utils.cpp_extension import load

pvcnn_backend = load(
    name="_pvcnn_backend",
    sources=[
        "modules/functional/src/voxelization/vox.cpp",
        "modules/functional/src/ball_query/ball_query.cpp",
        "modules/functional/src/interpolate/neighbor_interpolate.cpp",
    ],
    extra_cflags=["-O3", "-std=c++17", "-I/usr/include"],
    verbose=True,
)

