#include <iostream>
#include <CL/sycl.hpp>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#include <filesystem>
#include <cstdlib>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
using namespace std;
using namespace sycl;
#include <vector>

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
int perform_clean(queue& q, double *dirty, double *psf, double gain, int iters, double *local_max_x,
	double *local_max_y, double *local_max_z, double *model_l, double *model_m, double *model_intensity,int *d_source_c,
	double *max_xyz,double *running_avg) {
	int image_size = 1024;
	int cycle_number = 0;
	double flux = 0.0;
	bool exit_early = false;
	int num_cy = 0;
	double loop_gain = 0.1;
	double weak_source_percent = 0.01;
	double noise_detection_factor = 2.0;
	range<1> num_rows{ 1024 };
	
	//Find max row reduct
	for (int i = 0; i < 6; i++) {
		auto h = q.parallel_for(num_rows, [=](auto j) {
			double max_x = double(0);
			double max_y = abs(dirty[j * 1024]);
			double max_z = dirty[j * 1024];
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
		h.wait();

		//Find max col reduct
		max_xyz[0] = local_max_x[0];
		max_xyz[1] = local_max_y[0];
		max_xyz[2] = local_max_z[0];
		running_avg[0] = local_max_y[0];
		max_xyz[1] = 0.0;
		auto g = q.parallel_for(num_rows, [=](auto k) {
			double current_x = local_max_x[k + 1];
			double current_y = local_max_y[k + 1];
			double current_z = local_max_z[k + 1];
			running_avg[0] += current_y;
			current_y = k + 1;
			if (abs(current_z) > abs(max_xyz[2])) {
				max_xyz[0] = current_x;
				max_xyz[1] = current_y;
				max_xyz[2] = current_z;
			}

			});
		g.wait();

		// substract psf values for input
		running_avg[0] /= (image_size * image_size);
		const int half_psf = 1024 / 2;
		double *avg = &running_avg[0];
		double *zc = &max_xyz[2];
		double* zero_index = &model_intensity[0];
		bool extracting_noise = *zc < noise_detection_factor * *avg * loop_gain;
		bool weak_source = *zc < *zero_index * weak_source_percent;
		
		if (extracting_noise || weak_source) {
			return num_cy;
		}
		
		model_l[d_source_c[0]] = max_xyz[0];
		model_m[d_source_c[0]] = max_xyz[1];
		model_intensity[d_source_c[0]] = max_xyz[2];
		
		
		d_source_c[0] += 1;
		for (int i = 0; i < 1024; i++) {
			auto e = q.parallel_for(num_rows, [=](auto k){
				int image_coord_x = model_l[d_source_c[0]] - half_psf + i;
				int image_coord_y = model_m[d_source_c[0] - 1] - half_psf + k;
				double psf_weight = psf[k * 1024 + i];
				dirty[image_coord_y * 1024 + image_coord_x] -= psf_weight * model_intensity[d_source_c[0] - 1];
				std::cout << "Removing outstanders" << "\n";
				std::cout << "Updated value on row: ";
				std::cout << image_coord_y << "\n";
				std::cout << dirty[image_coord_y * 1024 + image_coord_x] << "\n";
				});
			e.wait();
		}

		
		auto f = q.parallel_for(num_rows, [=](auto m)  {
			double last_source_x = model_l[d_source_c[0] - 1];
			double last_source_y = model_m[d_source_c[0] - 1];
			double last_source_z = model_intensity[d_source_c[0] - 1];
			for (int w = d_source_c[0] - 2; w >= 0; w--) {
				if ((int)last_source_x == (int)model_l[w] && (int)last_source_y == (int)model_m[w])
				{
					model_intensity[w] += last_source_z;
					d_source_c[0]--;
					break;

				}
			}
			});
		f.wait();
		num_cy++;

	}
	return num_cy;

}
bool load_image_from_file(double* image, unsigned int size, char* input_file)
{
	FILE* file = fopen(input_file, "r");

	if (file == NULL)
	{
		perror("Failed: ");
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
void save_image_to_file(double* image, unsigned int size, char* real_file)
{
	FILE* image_file = fopen(real_file, "w");

	if (image_file == NULL)
	{
		printf(">>> ERROR: Unable to save image to file, moving on...\n\n");
		return;
	}

	for (int row = 0; row < size; ++row)
	{
		for (int col = 0; col < size; ++col)
		{
			unsigned int image_index = row * size + col;
			fprintf(image_file, "%.15f ", image[image_index]);
		}

		fprintf(image_file, "\n");
	}
 
	fclose(image_file);
}
void save_sources_to_file(double* source_x, double* source_y, double* source_z, int number_of_sources, char* output_file)
{
	FILE* file = fopen(output_file, "w");

	if (file == NULL)
	{
		printf(">>> ERROR: Unable to save sources to file, moving on...\n\n");
		return;
	}

	fprintf(file, "%d\n", number_of_sources);
	for (int index = 0; index < number_of_sources; ++index)
	{

		fprintf(file, "%.15f %.15f %.15f\n", source_x[index], source_y[index], source_z[index]);

	}

	fclose(file);
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
	size_t size_square = 1024 *1024;
	size_t number_cycles = 60;
	size_t single_element = 1;
	size_t three_d = 3;
	try {
		queue q(d_selector, exception_handler);
		double gain = 0.1;
		int iters = 60;
		double* dirty = malloc_shared<double>(size_square, q);
		double* psf = malloc_shared<double>(size_square, q);
		double* model_l = malloc_shared<double>(number_cycles, q);
		double* model_m = malloc_shared<double>(number_cycles, q);
		double* model_intensity = malloc_shared<double>(number_cycles, q);
		double* local_max_x = malloc_shared<double>(image_size, q);
		double* local_max_y = malloc_shared<double>(image_size, q);
		double* local_max_z = malloc_shared<double>(image_size, q);
		double* max_xyz = malloc_shared<double>(three_d, q);
		int* d_source_c = malloc_shared<int>(single_element, q);
		double* running_avg = malloc_shared<double>(single_element, q);
		char* dirty_image = "dirty1.csv";
		

		char* psf_image = new char[7];
		strcpy(psf_image, "psf.csv");
		char* output_img=new char[7];
		strcpy(output_img, "img.csv");
		char* output_src = new char[10];
		strcpy(output_src, "source.csv");

		std::filesystem::path cwd = std::filesystem::current_path();
		std::cout << cwd.string() << "\n";
		
		
		bool loaded_dirty = load_image_from_file(dirty, 1024, dirty_image);
		cout << loaded_dirty << "\n";
		
		
		
		bool loaded_psf = load_image_from_file(psf, 1024, psf_image);
		int number_of_cycle=perform_clean(q, dirty, psf, gain, iters, local_max_x,
			local_max_y, local_max_z, model_l, model_m, model_intensity, d_source_c,max_xyz,running_avg);
		save_image_to_file(dirty,1024, output_img);
		for (int i = 0; i < 65; i++) {
			cout << output_img[i] << "\n";
		}
		save_sources_to_file(model_l,model_m,model_intensity,number_of_cycle,output_src);
		
	}
	catch (std::exception const& e) {
		std::cout << "An exception is caught for FIR.\n";
		std::terminate();
	}
	return 0;
}
