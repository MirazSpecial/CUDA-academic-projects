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
CMAKE_SOURCE_DIR = /home/miraz/semestr_4/JNP2/Redukcja_shuffle

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/Redukcja_shuffle.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Redukcja_shuffle.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Redukcja_shuffle.dir/flags.make

CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o: CMakeFiles/Redukcja_shuffle.dir/flags.make
CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o: ../Redukcja_shuffle.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o"
	/opt/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -dc /home/miraz/semestr_4/JNP2/Redukcja_shuffle/Redukcja_shuffle.cu -o CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o

CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target Redukcja_shuffle
Redukcja_shuffle_OBJECTS = \
"CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o"

# External object files for target Redukcja_shuffle
Redukcja_shuffle_EXTERNAL_OBJECTS =

CMakeFiles/Redukcja_shuffle.dir/cmake_device_link.o: CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o
CMakeFiles/Redukcja_shuffle.dir/cmake_device_link.o: CMakeFiles/Redukcja_shuffle.dir/build.make
CMakeFiles/Redukcja_shuffle.dir/cmake_device_link.o: CMakeFiles/Redukcja_shuffle.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA device code CMakeFiles/Redukcja_shuffle.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Redukcja_shuffle.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Redukcja_shuffle.dir/build: CMakeFiles/Redukcja_shuffle.dir/cmake_device_link.o

.PHONY : CMakeFiles/Redukcja_shuffle.dir/build

# Object files for target Redukcja_shuffle
Redukcja_shuffle_OBJECTS = \
"CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o"

# External object files for target Redukcja_shuffle
Redukcja_shuffle_EXTERNAL_OBJECTS =

Redukcja_shuffle: CMakeFiles/Redukcja_shuffle.dir/Redukcja_shuffle.cu.o
Redukcja_shuffle: CMakeFiles/Redukcja_shuffle.dir/build.make
Redukcja_shuffle: CMakeFiles/Redukcja_shuffle.dir/cmake_device_link.o
Redukcja_shuffle: CMakeFiles/Redukcja_shuffle.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable Redukcja_shuffle"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Redukcja_shuffle.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Redukcja_shuffle.dir/build: Redukcja_shuffle

.PHONY : CMakeFiles/Redukcja_shuffle.dir/build

CMakeFiles/Redukcja_shuffle.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Redukcja_shuffle.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Redukcja_shuffle.dir/clean

CMakeFiles/Redukcja_shuffle.dir/depend:
	cd /home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/miraz/semestr_4/JNP2/Redukcja_shuffle /home/miraz/semestr_4/JNP2/Redukcja_shuffle /home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug /home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug /home/miraz/semestr_4/JNP2/Redukcja_shuffle/cmake-build-debug/CMakeFiles/Redukcja_shuffle.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Redukcja_shuffle.dir/depend
