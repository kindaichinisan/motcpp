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
Solution: set PATH="D:\WJ_software\opencv\opencv-4.12.0\install\x64\mingw\bin;D:\WJ_software\mingw64\bin";%PATH% so that opencv and mingw dll used are the right now.  

## 4.2 Git: fatal: unable to access "https://github.com/kindaichinisan/motcpp.git/": SSL certificate problem: unable to get local issuer certificate
git config --global --replace-all http.sslCAPath "C:\certificates\"  
git config --global --replace-all http.sslCAInfo "C:\certificates\certbundle.crt"  
