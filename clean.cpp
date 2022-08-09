#include <iostream>
#include <CL/sycl.hpp>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include "filter.h"
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
using namespace std;
using namespace sycl;
#include <vector>;
// the following three has to be decided by us regarding input size
typedef struct Complex {
	double real;
	double imag;
} Complex;

typedef struct double3 {
	double x;
	double y;
	double z;
} double3;
typedef struct Source {
	double l;
	double m;
	double intensity;
} Source;
static auto exception_handler = [](sycl::exception_list e_list) {
	for (std::exception_ptr const& e : e_list) {
		try {
			std::rethrow_exception(e);
		}
		catch (std::exception const& e) {
#if _DEBUG
			std::cout << "Failure" << std::endl;
#endif
			std::terminate();
		}
	}
};
int perform_clean(queue& q, double* dirty, double* psf, double gain, double thresh, int iters, double* local_max_x,
	double* local_max_y, double* local_max_z, double* model_l, double* model_m, double* model_inten) {
	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->image_size);
	int num_blocks = (int)ceil((double)1024 / max_threads_per_block);
	int cycle_number = 0;
	double flux = 0.0;
	bool exit_early = false;
	double loop_gain = 0.1;
	double weak_source_percent = 0.01;
	double noise_detection_factor = 2.0;
	range<1> num_rows{ 1024 };
	//Find max row reduct
	for (int i = 0; i < 60; i++) {
		auto e = q.parallel_for(num_rows, [=](auto j) {
			double max_x = make_double(0.0);
			double max_y = abs(dirty[j * 1024]);
			double max_z = dirty[j * 1024]);
		double current;
		for (int col_index = 1; col_index < 1024; ++col_index)
		{
			current = dirty[j * 1024 + col_index];
			max_y += abs(current);
			if (abs(current) > abs(max_z))
			{
				max_x = (double)col_index;
				max_z = current;
			}
		}

		local_max_x[j] = max_x;
		local_max_y[j] = max_y;
		local_max_z[j] = max_z;
		});
		e.wait();

		//Find max col reduct
		double max_x1 = local_max_x[0];
		double max_y1 = local_max_y[0];
		double max_z1 = local_max_z[0];
		double running_avg = local_max_y[0];
		max_y1 = 0.0;
		auto e = q.parallel_for(num_rows, [=](auto k) {
			double current_x = local_max_x[k + 1];
			double current_y = local_max_y[k + 1];
			double current_z = local_max_z[k + 1];
			running_avg += current_y;
			current_y = k + 1;
			if (abs(current_z) > abs(max_z1)) {
				max_x1 = current_x;
				max_y1 = current_y;
				max_z1 = current_z;
			}

			});
		e.wait();
		running_avg /= (image_size * image_size);
		const int half_psf = 1024 / 2;
		bool extracting_noise = max_z1 < noise_detection_factor* running_avg* loop_gain;
		bool weak_source = max_z1 < model_intensity* weak_source_percent;
		bool exit_early = extracting_noise || weak_source;
		if (exit_early) {
			return;
		}
		model_l[d_source_counter] = max_x1;
		model_m[d_source_counter] = max_y1;
		model_intensity[d_source_counter] = max_z1;
		d_flux = flux + max_z1;
		++d_source_counter;
		for (int i = 0; i < 1024; i++) {
			auto e = q.parallel_for(num_rows, [=](auto k) {
				image_coord_x = model_l[d_source] - half_psf + i;
				image_coord_y = model_m[d_cource_counter - 1] - half_psf + k;
				double psf_weight = psf[k * 1024 + i];
				dirty[image_coord_y * 1024 + image_coord.x] -= psf_weight * model_intensity[d_source_counter - 1];
				});
		}

		compress_sources();

	}

}
bool load_image_from_file(double* image, unsigned int size, char* input_file)
{
	FILE* file = fopen(input_file, "r");

	if (file == NULL)
	{
		printf(">>> ERROR: Unable to load image from file...\n\n");
		return false;
	}

	for (int row = 0; row < size; ++row)
	{
		for (int col = 0; col < size; ++col)
		{
			int image_index = row * size + col;
			fscanf(file, "%lf ", &(image[image_index]));

		}
	}

	fclose(file);
	return true;
}
int main() {
#if FPGA_EMULATOR
	// DPC++ extension: FPGA emulator selector on systems without FPGA card.
	ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
	// DPC++ extension: FPGA selector on systems with FPGA card.
	ext::intel::fpga_selector d_selector;
#else
	// The default device selector will select the most performant device.
	default_selector d_selector;
#endif
	size_t image_size = 1024;
	size_t psf_size = 1024;
	size_t size_square = 1024 * 1024;
	size_t number_cycles = 60;
	try {
		queue q(d_selector, exception_handler);
		double* dirty = malloc_shared<double>(size_square, q);
		double* psf = malloc_shared<double>(size_square, q);
		double* model_l = malloc_shared<double>(number_cycles, q);
		double* model_m = malloc_shared<double>(number_cycles, q);
		double* model_intensity = malloc_shared<double>(number_cycles, q);
		double* max_local_x = malloc_shared<double>(image_size, q);
		double* max_local_y = malloc_shared<double>(image_size, q);
		double* max_local_z = malloc_shared<double>(image_size, q);
		bool loaded_dirty = load_image_from_file(dirty, image_size, 'dirty.csv');
		bool loaded_psf = load_image_from_file(psf, psf_size, 'psf.csv');
	}
	catch (std::exception const& e) {
		std::cout << "An exception is caught for FIR.\n";
		std::terminate();
	}
	return 0;
}