#include <iostream>
#include <CL/sycl.hpp>
#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
using namespace std;
using namespace sycl;
#include <vector>
#include <chrono>
using namespace std::chrono;

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
int perform_clean(queue& q, float* dirty, float* psf, float gain, int iters, float* local_max_x,
	float* local_max_y, float* local_max_z, float* model_l, float* model_m, float* model_intensity, int* d_source_c,
	float* max_xyz, float* running_avg, float* operation_count) {
	int image_size = 8;
	int cycle_number = 0;
	float flux = 0.0;
	bool exit_early = false;
	int num_cy = 0;
	float loop_gain = 0.1;
	float weak_source_percent = 0.01;
	float noise_detection_factor = 2.0;
	range<1> num_rows{ 8 };

	//Find max row reduct
	for (int i = 0; i < 6; i++) {
		auto h = q.parallel_for(num_rows, [=](auto j) {
			float max_x = float(0);
			float max_y = fabs(dirty[j * 8]);
			float max_z = dirty[j * 8];

			float current;
			for (int col_index = 1; col_index < 8; ++col_index)
			{
				current = dirty[j * 8 + col_index];

				max_y += fabs(current);
				if (fabs(current) > fabs(max_z))
				{
					max_x = (float)col_index;
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
			float current_x = local_max_x[k + 1];
			float current_y = local_max_y[k + 1];
			float current_z = local_max_z[k + 1];
			running_avg[0] += current_y;
			current_y = k + 1;
			if (fabs(current_z) > fabs(max_xyz[2])) {
				max_xyz[0] = current_x;
				max_xyz[1] = current_y;
				max_xyz[2] = current_z;
			}

			});
		g.wait();

		// substract psf values for input
		running_avg[0] /= (image_size * image_size);
		const int half_psf = 8 / 2;
		float* avg = &running_avg[0];
		float* zc = &max_xyz[2];
		float* zero_index = &model_intensity[0];
		bool extracting_noise = *zc < noise_detection_factor** avg* loop_gain;
		bool weak_source = *zc < *zero_index* weak_source_percent;

		if (extracting_noise || weak_source) {
			return num_cy;
		}

		model_l[d_source_c[0]] = max_xyz[0];
		model_m[d_source_c[0]] = max_xyz[1];
		model_intensity[d_source_c[0]] = max_xyz[2];


		d_source_c[0] += 1;
		for (int i = 0; i < 8; i++) {
			auto e = q.parallel_for(num_rows, [=](auto k) {
				int image_coord_x = model_l[d_source_c[0]] - half_psf + i;
				int image_coord_y = model_m[d_source_c[0] - 1] - half_psf + k;
				float psf_weight = psf[k * 8 + i];
				dirty[image_coord_y * 8 + image_coord_x] -= psf_weight * model_intensity[d_source_c[0] - 1];
				operation_count[0] += 1.0;
				operation_count[1] = dirty[image_coord_y * 8 + image_coord_x];
				operation_count[2] = psf_weight * model_intensity[d_source_c[0] - 1];
				});
			std::cout << "Operation: ";
			std::cout << operation_count[0] << "\n";
			std::cout << operation_count[1] << "\n";
			std::cout << operation_count[2] << "\n";

		}


		auto f = q.parallel_for(num_rows, [=](auto m) {
			float last_source_x = model_l[d_source_c[0] - 1];
			float last_source_y = model_m[d_source_c[0] - 1];
			float last_source_z = model_intensity[d_source_c[0] - 1];
			for (int w = d_source_c[0] - 2; w >= 0; w--) {
				if ((int)last_source_x == (int)model_l[w] && (int)last_source_y == (int)model_m[w])
				{
					model_intensity[w] += last_source_z;
					d_source_c[0]--;
					break;

				}
			}
			});
		
		num_cy++;

	}
	return num_cy;

}
bool load_image_from_file(float* image, unsigned int size, char* input_file)
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
			fscanf(file, "%f ", &(image[image_index]));

		}
	}

	fclose(file);
	return true;
}
void save_image_to_file(float* image, unsigned int size, char* real_file)
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
			fprintf(image_file, "%f ", image[image_index]);
		}

		fprintf(image_file, "\n");
	}

	fclose(image_file);
}
void save_sources_to_file(float* source_x, float* source_y, float* source_z, int number_of_sources, char* output_file)
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

		fprintf(file, "%f %f %f\n", source_x[index], source_y[index], source_z[index]);

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
	size_t image_size = 8;
	size_t psf_size = 8;
	size_t size_square = 8 * 8;
	size_t number_cycles = 60;
	size_t single_element = 1;
	size_t three_d = 3;

	try {
		queue q(d_selector, exception_handler);
		float gain = 0.1;
		int iters = 60;
		float* dirty = malloc_shared<float>(size_square, q);
		float* psf = malloc_shared<float>(size_square, q);
		float* model_l = malloc_shared<float>(number_cycles, q);
		float* model_m = malloc_shared<float>(number_cycles, q);
		float* model_intensity = malloc_shared<float>(number_cycles, q);
		for (int i = 0; i < 8; i++) {
			model_intensity[i] = 0.0;
		}
		float* local_max_x = malloc_shared<float>(image_size, q);
		float* local_max_y = malloc_shared<float>(image_size, q);
		float* local_max_z = malloc_shared<float>(image_size, q);
		float* max_xyz = malloc_shared<float>(three_d, q);
		int* d_source_c = malloc_shared<int>(single_element, q);
		d_source_c[0] = 0;
		float* operation_count = malloc_shared<float>(three_d, q);
		operation_count[0] = 0.0;
		float* running_avg = malloc_shared<float>(single_element, q);
		char* dirty_image = new char[9];
		strcpy(dirty_image, "dirty.csv");
		char* psf_image = new char[7];
		strcpy(psf_image, "psf.csv");
		char* output_img = new char[7];
		strcpy(output_img, "img.csv");
		char* output_src = new char[10];
		strcpy(output_src, "source.csv");


		bool loaded_dirty = load_image_from_file(dirty, 8, dirty_image);
		bool loaded_psf = load_image_from_file(psf, 8, psf_image);

		// start timer
		auto start = high_resolution_clock::now();

		int number_of_cycle = perform_clean(q, dirty, psf, gain, iters, local_max_x,
			local_max_y, local_max_z, model_l, model_m, model_intensity, d_source_c, max_xyz, running_avg, operation_count);

		// stop timer
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);

		// To get the value of duration use the count()
		// member function on the duration object
		cout << "The time taken is:" << std::endl;
		cout << duration.count() << std::endl;

		std::cout << number_of_cycle << "\n";
		save_image_to_file(dirty, 8, output_img);
		save_sources_to_file(model_l, model_m, model_intensity, number_of_cycle, output_src);
	}
	catch (std::exception const& e) {
		std::cout << "An exception is caught for clean.\n";
		std::terminate();
	}
	return 0;
}