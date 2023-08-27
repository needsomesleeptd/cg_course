cmake_minimum_required(VERSION 3.26.X)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(lab_03 VERSION 0.1 LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    #target_link_libraries(rendererLib CUDA::cudart)
    #target_link_libraries(rendererLib CUDA::cuda_driver)
else()
    message(STATUS "No CUDA compiler found")
endif()



MACRO(HEADER_DIRECTORIES return_list)
    FILE(GLOB_RECURSE new_list *.h)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()
HEADER_DIRECTORIES(header_dir_list)


set(PROJECT_SOURCES
        main.cpp
        mainwindow.cpp
        mainwindow.h
        mainwindow.ui
        math_primitives/vector/vector.cpp math_primitives/vector/vector.h math_primitives/vector/vector_exceptions.h
        math_primitives/ray/ray.cu math_primitives/ray/ray.h
        visitor/TransformVisitor.h visitor/TransformVisitor.cpp visitor/visitor.h
        object/primitives/transform.h
        object/visibleObject/shapes/sphere/sphere.cpp object/visibleObject/shapes/sphere/sphere.h
        object/visibleObject/shapes/baseShape.h
        object/object.h
        object/invisibleObject/cameraStructure.cpp  object/invisibleObject/cameraStructure.h
        object/invisibleObject/baseCamera.h
        object/invisibleObject/camera.h
        object/invisibleObject/camera.cpp
        scene/scene.cu scene/scene.h
        object/visibleObject/composite.cpp object/visibleObject/composite.h
        utils/glmWrapper.h utils/glmWrapper.cpp
        manager/scene/sceneManager.cpp manager/scene/sceneManager.h
        manager/scene/sceneManagerCreator.cpp manager/scene/sceneManagerCreator.h
        manager/draw/drawManager.cpp manager/draw/drawManager.h
        manager/draw/drawManagerCreator.cpp manager/draw/drawManagerCreator.h

        object/invisibleObject/lightSource/LightSource.cpp object/invisibleObject/lightSource/LightSource.h object/invisibleObject/lightSource/baseLightSource.h

        color/color.cu color/color.h image_adapter/ImageAdapter.cu image_adapter/ImageAdapter.h material/material.cpp material/material.h utils/cudaUtils.h object/visibleObject/CudaShape.cpp object/visibleObject/CudaShape.h
        renderer/baseRenderer.h
        renderer/Renderer.cuh
        renderer/Renderer.cu
        )





find_package(CUDA REQUIRED)

set(CUDA_TOOLKIT_ROOT_DIR " usr/local/cuda-12.2")
message("CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message("cuda version: ${CUDA_VERSION_STRING}")

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11  -DNDEBUG")
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")


set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(QT_DIR "/opt/Qt5.12.0")
set(QT_SUB_DIR "5.12.0")
find_package(Qt5 COMPONENTS Widgets Gui OpenGL HINTS ${QT_DIR}/${QT_SUB_DIR}/gcc_64/lib/cmake/Qt5)


set(CMAKE_CXX_STANDARD 17)


CUDA_ADD_EXECUTABLE(lab_03 ${PROJECT_SOURCES})
target_compile_options(lab_03 PUBLIC -fPIC)

list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp -dc)

#set_source_files_properties(renderer/baseRenderer.h renderer/Renderer.cuh renderer/Renderer.cu PROPERTIES LANGUAGE CUDA)

set(RendererLibName "rendererLib")
ADD_LIBRARY(${RendererLibName} SHARED
        renderer/baseRenderer.h
        renderer/Renderer.cuh
        renderer/Renderer.cu
        )

set_target_properties(lab_03 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${RendererLibName} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_property(TARGET lab_03 PROPERTY CUDA_SEPARABLE_COMPILATION ON)



target_include_directories(
        ${RendererLibName}
        PUBLIC
        ${header_dir_list}
)

#list(APPEND CUDA_LIBRARIES rendererLib)





message(CUDA_LIBS==${CUDA_LIBRARIES})
target_link_libraries(
        ${PROJECT_NAME}

        Qt5::Widgets
        ${CUDA_LIBRARIES}
        ${CUDA_cudadevrt_LIBRARY}

)



target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${header_dir_list}
)

if ( TARGET Qt5::Core )
    get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
    string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
    set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
    set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )

endif()