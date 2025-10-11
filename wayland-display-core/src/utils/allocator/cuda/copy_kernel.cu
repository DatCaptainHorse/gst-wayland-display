extern "C" __global__ void copy_array_to_linear(
    cudaTextureObject_t src_tex,
    unsigned char* dst,
    int width,
    int height,
    int dst_pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uchar4 pixel = tex2D<uchar4>(src_tex, x + 0.5f, y + 0.5f);
        int dst_idx = y * dst_pitch + x * 4;
        dst[dst_idx + 0] = pixel.x;
        dst[dst_idx + 1] = pixel.y;
        dst[dst_idx + 2] = pixel.z;
        dst[dst_idx + 3] = pixel.w;
    }
}