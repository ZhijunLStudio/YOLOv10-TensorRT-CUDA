#include"../utils/kernel_function.h"
#include<math.h>

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

__device__ 
void affine_project_device_kernel(utils::AffineMat* matrix, int x, int y, float* proj_x, float* proj_y)
{
	*proj_x = matrix->v0 * x + matrix->v1 * y + matrix->v2;
	*proj_y = matrix->v3 * x + matrix->v4 * y + matrix->v5;
}

__global__ 
void resize_rgb_padding_device_kernel(float* src, int src_width, int src_height, int src_area, int src_volume,
	float* dst, int dst_width, int dst_height, int dst_area, int dst_volume,
	int batch_size, float padding_value, utils::AffineMat matrix)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < dst_area && dy < batch_size)
	{
		int dst_y = dx / dst_width; 
		int dst_x = dx % dst_width; 
		float src_x = 0;
		float src_y = 0;
		affine_project_device_kernel(&matrix, dst_x, dst_y, &src_x, &src_y);
		float c0 = padding_value, c1 = padding_value, c2 = padding_value;
		if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
		{
		}
		else
		{
			int y_low = floorf(src_y);
			int x_low = floorf(src_x);
			int y_high = y_low + 1;   
			int x_high = x_low + 1;   
			float const_values[] = { padding_value, padding_value, padding_value };
			float ly = src_y - y_low;
			float lx = src_x - x_low;
			float hy = 1 - ly;
			float hx = 1 - lx;
			float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; 
			float* v1 = const_values;
			float* v2 = const_values;
			float* v3 = const_values;
			float* v4 = const_values;

			if (y_low >= 0)
			{
				if (x_low >= 0)
					v1 = src + dy * src_volume + y_low * src_width * 3 + x_low * 3;

				if (x_high < src_width)
					v2 = src + dy * src_volume + y_low * src_width * 3 + x_high * 3;
			}

			if (y_high < src_height)
			{
				if (x_low >= 0)
					v3 = src + dy * src_volume + y_high * src_width * 3 + x_low * 3;

				if (x_high < src_width)
					v4 = src + dy * src_volume + y_high * src_width * 3 + x_high * 3;
			}
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}
		float* pdst = dst + dy * dst_volume + dst_y * dst_width * 3 + dst_x * 3;
		pdst[0] = c0;
		pdst[1] = c1;
		pdst[2] = c2;
	}
}



__global__ 
void bgr2rgb_device_kernel(float* src, float* dst,
	int batch_size, int img_height, int img_width, int img_area, int img_volume)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < img_volume && dy < batch_size)
	{
		int ch = dx % 3;
		assert(ch < 3);

		switch (ch)
		{
		case 0:
			dst[dy * img_volume + dx] = src[dy * img_volume + dx + 2];
			return;
		case 1:
			dst[dy * img_volume + dx] = src[dy * img_volume + dx];
			return;
		case 2:
			dst[dy * img_volume + dx] = src[dy * img_volume + dx - 2];
			return;
		}
	}
}

static __device__  
float norm_device(float val, float s, float mean, float std)
{
	return ((val / s) - mean) / std;
}

__global__ 
void norm_device_kernel(float* src, float* dst,
	int batch_size, int img_height, int img_width, int img_area, int img_volume,
	float scale,
	float mean0, float mean1, float mean2,
	float std0, float std1, float std2)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < img_volume && dy < batch_size)
	{
		int ch = dx % 3;
		assert(ch < 3);

		switch (ch)
		{
		case 0:
			dst[dy * img_volume + dx] = norm_device(src[dy * img_volume + dx], scale, mean0, std0);
			break;
		case 1:
			dst[dy * img_volume + dx] = norm_device(src[dy * img_volume + dx], scale, mean1, std1);
			break;
		case 2:
			dst[dy * img_volume + dx] = norm_device(src[dy * img_volume + dx], scale, mean2, std2);
			break;
		}
	}
}

__global__ void hwc2chw_device_kernel(float* src, float* dst,
	int batch_size, int img_height, int img_width, int img_area, int img_volume)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < img_volume && dy < batch_size)
	{

		int ch = dx / img_area;
		assert(ch < 3);
		int sub_idx = dx % img_area;
		int row = sub_idx / img_width;
		int col = sub_idx % img_width;

		int dx_ = row * (img_width * 3) + col * 3 + ch;
		dst[dy * img_volume + dx] = src[dy * img_volume + dx_];
	}
}

void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight,
	float paddingValue, AffineMat matrix)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	int src_volume = 3 * srcHeight * srcWidth;
	int src_area = srcHeight * srcWidth;

	int dst_volume = 3 * dstHeight * dstWidth;
	int dst_area = dstHeight * dstWidth;

	resize_rgb_padding_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, srcWidth, srcHeight, src_area, src_volume,
		dst, dstWidth, dstHeight, dst_area, dst_volume,
		batchSize, paddingValue, matrix);
}



void bgr2rgbDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	int img_volume = 3 * srcHeight * srcWidth;
	int img_area = srcHeight * srcWidth;
	int img_height = srcHeight;
	int img_width = srcWidth;
	bgr2rgb_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, dst, batchSize, img_height, img_width, img_area, img_volume);
}

void normDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	int img_volume = 3 * srcHeight * srcWidth;
	int img_area = srcHeight * srcWidth;
	int img_height = srcHeight;
	int img_width = srcWidth;
	norm_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, dst, batchSize, img_height, img_width, img_area, img_volume,
		param.scale, means[0], means[1], means[2], stds[0], stds[1], stds[2]);
}

void hwc2chwDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);


	int img_volume = 3 * srcHeight * srcWidth;
	int img_area = srcHeight * srcWidth;
	int img_height = srcHeight;
	int img_width = srcWidth;
	hwc2chw_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, dst, batchSize, img_height, img_width, img_area, img_volume);
}
