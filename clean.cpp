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
int perform_clean(queue &q,double* dirty, double* psf, double gain, double thresh, int iters) {
	int max_threads_per_block = min(config->gpu_max_threads_per_block, config->image_size);
	int num_blocks = (int)ceil((double)1024 / max_threads_per_block);
	int cycle_number = 0;
	double flux = 0.0;
	bool exit_early = false;
	range<1> num_rows{ 1024 };
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
		find_max_col_reduction();
		subtract_psf();
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
	size_t size_square = 1024*1024;
	size_t number_cycles = 60;
	try {
		queue q(d_selector, exception_handler);
		double* dirty = malloc_shared<double>(size_square, q);
		double* psf = malloc_shared<double>(size_square, q);
		Source* model = malloc_shared<Source>(number_cycles, q);
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