cmake_minimum_required(VERSION 3.24.3)

project(lab_03 VERSION 0.1 LANGUAGES CXX CUDA)




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

         color/color.cu color/color.h image_adapter/ImageAdapter.cu image_adapter/ImageAdapter.h material/material.cpp material/material.h utils/cudaUtils.h)




find_package(CUDA REQUIRED)

message("CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message("cuda version: ${CUDA_VERSION_STRING}")

include_directories(
        ${CUDA_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -gencode arch=compute_61,code=sm_61 -DNDEBUG")
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

target_link_libraries(lab_03
        Qt5::Widgets
        ${CUDA_LIBRARIES}
        ${CUDA_cudadevrt_LIBRARY}
        )

HEADER_DIRECTORIES(header_dir_list)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${header_dir_list}
)

list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fopenmp)

CUDA_ADD_LIBRARY(rendererLib SHARED
       renderer/baseRenderer.h
        renderer/Renderer.h
        renderer/Renderer.cu
        OPTIONS ${CUDA_NVCC_FLAGS}
        )
target_include_directories(
        rendererLib
        PUBLIC
        ${header_dir_list}
)

target_link_libraries(rendererLib
        Qt5::Widgets
        )


TARGET_LINK_LIBRARIES(lab_03 rendererLib)