# UoA-2022-Radio-Astronomy-p4p
Repository for project "radio astronomy on FPGAs using high-level approaches
In order to run the clean deconvolution: 

•    Step 1: Register an Intel Devcloud account. Instructions can be found on Intel’s own documentation sites. This can be found with the provided link (requires Intel account sign-in) https://devcloud.intel.com/oneapi/get_started/baseToolkitSamples/ 

•    Step 2: Log in to your Devcloud account with the terminal on Linux systems. Use the command “ssh devcloud”. 

•    Step 3: Clone the oneAPI sample GitHub repository

•    Step 4: Navigate to the following directory

•    Step 5: Upload the following files to the directory in step 4 via the scp command: 

        •  Img.csv
        •  psf.csv
        •  dirty.csv
        •  source.csv
        
•    Step 6: Navigate to the src directory, and replace the content in vector-add-usm with the content found in clean.cpp

•    Step 7: Navigate back to the vector-add directory. Compile the project via the command “qsub build.sh” to compile the project on GPU, “qsub build_fpga_emu.sh” to compile on FPGA emulator, or “qsub build_fpga.sh” to compile project on FPGA hardware.
o    One can edit the build_fpga.sh file to change the targeted FPGA device type

•    Step 8: Check the error log for any failure reports. Execute the compiled project with the command “qsub run.sh” if one compiled on GPU, “qsub run_fpga_emu.sh” for FPGA emulators, or “qsub run_fpga_hw.sh” for hardware FPGA.

•    Step 9: Check the output file for execution timings. Check img.csv for the output data. One can compare the img.csv with the residual_image_1024.csv of the CUDA CLEAN. The instruction on setting up CUDA CLEAN can be found here: https://github.com/ska-telescope/CUDA_Deconvolution
