# 1 Use CMake-gui to modify CMakeLists.txt parameters
src directory: D:\WJ_git\motcpp  
build directory: D:\WJ_git\motcpp\build  
Configure.  
Modify OpenCV_DIR: D:/WJ_software/opencv/opencv-4.12.0/install  
Modify yaml-cpp_DIR: D:/WJ_git/yaml-cpp/install/lib/cmake/yaml-cpp  
Modify Eigen3_DIR (NB Cannot use Eigen3.3.90, must be >=Eigen3.4.1. See # 3): D:/WJ_git/eigen/install/share/eigen3/cmake  
MOTCPP_ENABLE_ONNX: untick  
MOTCPP_BUILD_TESTS: untick  
CMAKE_INSTALL_PREFIX: D:/WJ_git/motcpp/install  
Generate.  

# 2 build using mingw32-make
cd build
mingw32-make  


# 3 Eigen3
To see the Eigen3 version, see D:\WJ_git\eigen\Eigen\src\Core\util\Macros.h  
git clone from https://gitlab.com/libeigen/eigen  
Switch to branch 3.4  
https://github.com/PX4/eigen master branch is only 3.3.90.  

# 4 Build Eigen3 from source
Use CMake-gui.  
src directory: D:/WJ_git/eigen  
build directory: D:/WJ_git/eigen/build  
Configure.  
Modify CMAKE_INSTALL_PREFIX: D:/WJ_git/eigen/install  
Generate.  
mingw32-make  
To rename INSTALL file to INSTALL_INSTRUCTION  
mingw32-make install  

# 4 Run 
cd D:\WJ_git\motcpp\build\tools  
motcpp_eval.exe  
It should returns:  
    Usage: D:\WJ_git\motcpp\build\tools\motcpp_eval.exe <mot_root> <output_dir> [tracking_method] [det_emb_root] [model_name] [reid_name] [reid_weights]  
    Example: D:\WJ_git\motcpp\build\tools\motcpp_eval.exe ../assets/MOT17-mini/train ./results bytetrack  
    Example with pre-generated dets/embs: D:\WJ_git\motcpp\build\tools\motcpp_eval.exe <mot_root> <output_dir> bytetrack ../assets/yolox_x_ablation yolox_x_ablation osnet_x1_0_dukemtmcreid  
    Example for DeepOCSort: D:\WJ_git\motcpp\build\tools\motcpp_eval.exe <mot_root> <output_dir> deepocsort <det_emb_root> <model_name> <reid_name> <reid_weights.onnx>  
    Example for StrongSORT: D:\WJ_git\motcpp\build\tools\motcpp_eval.exe <mot_root> <output_dir> strongsort <det_emb_root> <model_name> <reid_name> <reid_weights.onnx>  

# 4 Possible issues
## 4.1 The procedure entry point .... could not be located in the dynamic link library ...\motcpp_eval.exe.
Reason: Version mismatch between built and runtime dll loaded. Use DependenciesGUI to check.  
Solution: set PATH=D:\WJ_software\opencv\opencv-4.12.0\install\x64\mingw\bin;D:\WJ_software\mingw64\bin;%PATH% so that opencv and mingw dll used are the right one.  

## 4.2 Git: fatal: unable to access "https://github.com/kindaichinisan/motcpp.git/": SSL certificate problem: unable to get local issuer certificate
Download certificates certbundle.zip from https://gitlab.ai-stack.dso/eugene/ai-stack-certificates/-/tree/main/dso?ref_type=heads  
git config --global --replace-all http.sslCAPath "C:\certificates\"  
git config --global --replace-all http.sslCAInfo "C:\certificates\certbundle.crt"  

## 4.3 A single .cpp file generates too many template instantiations
mingw32-make [ 3%] Building CXX object CMakeFiles/motcpp.dir/src/config.cpp.obj [ 6%] Building CXX object CMakeFiles/motcpp.dir/src/tracker.cpp.obj [ 10%] Building CXX object CMakeFiles/motcpp.dir/src/utils/ops.cpp.obj [ 13%] Building CXX object CMakeFiles/motcpp.dir/src/utils/iou.cpp.obj [ 16%] Building CXX object CMakeFiles/motcpp.dir/src/utils/association.cpp.obj [ 20%] Building CXX object CMakeFiles/motcpp.dir/src/utils/matching.cpp.obj [ 23%] Building CXX object CMakeFiles/motcpp.dir/src/motion/kalman_filter.cpp.obj D:/WJ_software/mingw64/bin/../lib/gcc/x86_64-w64-mingw32/15.1.0/../../../../x86_64-w64-mingw32/bin/as.exe: CMakeFiles\motcpp.dir\src\motion\kalman_filter.cpp.obj: too many sections (33105) C:\Users\hwenjun\AppData\Local\Temp\ccOfTMgi.s: Assembler messages: C:\Users\hwenjun\AppData\Local\Temp\ccOfTMgi.s: Fatal error: can't write 197 bytes to section .text of CMakeFiles\motcpp.dir\src\motion\kalman_filter.cpp.obj: 'file too big' D:/WJ_software/mingw64/bin/../lib/gcc/x86_64-w64-mingw32/15.1.0/../../../../x86_64-w64-mingw32/bin/as.exe: CMakeFiles\motcpp.dir\src\motion\kalman_filter.cpp.obj: too many sections (33105) C:\Users\hwenjun\AppData\Local\Temp\ccOfTMgi.s: Fatal error: CMakeFiles\motcpp.dir\src\motion\kalman_filter.cpp.obj: file too big mingw32-make[2]: *** [CMakeFiles\motcpp.dir\build.make:166: CMakeFiles/motcpp.dir/src/motion/kalman_filter.cpp.obj] Error 1 mingw32-make[1]: *** [CMakeFiles\Makefile2:117: CMakeFiles/motcpp.dir/all] Error 2 mingw32-make: *** [makefile:135: all] Error 2  

## 5 Build program
g++ -std=c++17 test_MOT17_data.cpp -o test_MOT17_data -ID:\WJ_git\motcpp\include -LD:\WJ_git\motcpp\build -lmotcpp -ID:\WJ_software\opencv\opencv-4.12.0\install\include -LD:\WJ_software\opencv\opencv-4.12.0\install\x64\mingw\lib -lopencv_core4120 -lopencv_imgproc4120 -lopencv_highgui4120 -lopencv_video4120 -lopencv_videoio4120 -lopencv_imgcodecs4120 -ID:\WJ_git\eigen\install\include\eigen3 -LD:\WJ_git\eigen\install\lib  

## 6 Build motcpp lib for debugging to allow code stepping
Use CMake-gui.  
CMAKE_BUILD_TYPE=Debug  
Doing this alone will have the following issues in ## 4.3 A single .cpp file generates too many template instantiations  
Tick Advanced. CMAKE_CXX_FLAGS=-Wa,-mbig-obj  

## 7 To automate build
Ctrl + Shift + P > Tasks: Configure Task > C/C++: g++.exe build active file. tasks.json will appear.  
Modify tasks.json.  
"label": Change to a meaningful name so that can use as preLaunchTask in launch.json  
"args": Change all inside to be something like  
    "args": [  
        "-g", //for debug info  
        "-std=c++17",  
        "${workspaceFolder}\\wj_scripts\\test_MOT17_data.cpp",  
        "-o",  
        "${workspaceFolder}\\wj_scripts\\test_MOT17_data.exe",  
        "-ID:\\WJ_git\\motcpp\\include",  
        "-LD:\\WJ_git\\motcpp\\build",  
        "-lmotcpp",  
        "-ID:\\WJ_software\\opencv\\opencv-4.12.0\\install\\include",  
        "-LD:\\WJ_software\\opencv\\opencv-4.12.0\\install\\x64\\mingw\\lib",  
        "-lopencv_core4120",  
        "-lopencv_imgproc4120",  
        "-lopencv_highgui4120",  
        "-lopencv_video4120",  
        "-lopencv_videoio4120",  
        "-lopencv_imgcodecs4120",  
        "-ID:\\WJ_git\\eigen\\install\\include\\eigen3",  
        "-LD:\\WJ_git\\eigen\\install\\lib"  
    ]  
Ctrl + Shif + B > select correct build task  

## 8 To automate debug
Ctrl + Shift + D: create a launch.json file > C/C++: gdb launch. A launch.json will be created.  
Modify launch.json.  
"program": "${workspaceFolder}\\wj_scripts\\test_MOT17_data.exe",  
"miDebuggerPath": "D:\\WJ_software\\mingw64\\bin\\gdb.exe",  

## 9 Dont use #include <bits/stdc++.h>
#include <bits/stdc++.h> is not standard C++, it is a convenience header provided by GCC/MinGW. It does not exist on every system. Using it will fail on VS2022 build.  

## 10 Able to build motcpp in VS2022
add add_subdirectory(wj_scripts) to main CMakeLists.txt.  
Add CMakeLists.txt in wj_scripts folder.  
Use Cmake-gui to configure and generate.  
Use VS2022 to build in Debug and Release.  
Can step through in VS2022.  

## 11 Ensure VSC dont switch to cmake in Visual Studio (cl.exe)
Ctrl + Shift + P > settings.json (Workspace setting)  
Type the following inside.  
{
    "cmake.generator": "MinGW Makefiles",
    "cmake/cmakePath": "D:\\WJ_software\\mingw64\\bin\\cmake.exe"
}