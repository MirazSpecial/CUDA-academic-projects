# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/miraz/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/211.7442.42/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/miraz/.local/share/JetBrains/Toolbox/apps/CLion/ch-0/211.7442.42/bin/cmake/linux/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/miraz/semestr_4/JNP2/strumienie_i_2D

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/strumienie_i_2D.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/strumienie_i_2D.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/strumienie_i_2D.dir/flags.make

CMakeFiles/strumienie_i_2D.dir/main.cu.o: CMakeFiles/strumienie_i_2D.dir/flags.make
CMakeFiles/strumienie_i_2D.dir/main.cu.o: ../main.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/strumienie_i_2D.dir/main.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/miraz/semestr_4/JNP2/strumienie_i_2D/main.cu -o CMakeFiles/strumienie_i_2D.dir/main.cu.o

CMakeFiles/strumienie_i_2D.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/strumienie_i_2D.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/strumienie_i_2D.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/strumienie_i_2D.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target strumienie_i_2D
strumienie_i_2D_OBJECTS = \
"CMakeFiles/strumienie_i_2D.dir/main.cu.o"

# External object files for target strumienie_i_2D
strumienie_i_2D_EXTERNAL_OBJECTS =

CMakeFiles/strumienie_i_2D.dir/cmake_device_link.o: CMakeFiles/strumienie_i_2D.dir/main.cu.o
CMakeFiles/strumienie_i_2D.dir/cmake_device_link.o: CMakeFiles/strumienie_i_2D.dir/build.make
CMakeFiles/strumienie_i_2D.dir/cmake_device_link.o: CMakeFiles/strumienie_i_2D.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/strumienie_i_2D.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/strumienie_i_2D.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/strumienie_i_2D.dir/build: CMakeFiles/strumienie_i_2D.dir/cmake_device_link.o

.PHONY : CMakeFiles/strumienie_i_2D.dir/build

# Object files for target strumienie_i_2D
strumienie_i_2D_OBJECTS = \
"CMakeFiles/strumienie_i_2D.dir/main.cu.o"

# External object files for target strumienie_i_2D
strumienie_i_2D_EXTERNAL_OBJECTS =

strumienie_i_2D: CMakeFiles/strumienie_i_2D.dir/main.cu.o
strumienie_i_2D: CMakeFiles/strumienie_i_2D.dir/build.make
strumienie_i_2D: CMakeFiles/strumienie_i_2D.dir/cmake_device_link.o
strumienie_i_2D: CMakeFiles/strumienie_i_2D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable strumienie_i_2D"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/strumienie_i_2D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/strumienie_i_2D.dir/build: strumienie_i_2D

.PHONY : CMakeFiles/strumienie_i_2D.dir/build

CMakeFiles/strumienie_i_2D.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/strumienie_i_2D.dir/cmake_clean.cmake
.PHONY : CMakeFiles/strumienie_i_2D.dir/clean

CMakeFiles/strumienie_i_2D.dir/depend:
	cd /home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/miraz/semestr_4/JNP2/strumienie_i_2D /home/miraz/semestr_4/JNP2/strumienie_i_2D /home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug /home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug /home/miraz/semestr_4/JNP2/strumienie_i_2D/cmake-build-debug/CMakeFiles/strumienie_i_2D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/strumienie_i_2D.dir/depend
