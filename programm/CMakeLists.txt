cmake_minimum_required(VERSION 3.26.X)

set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")
project(lab_03 VERSION 0.1 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE TRUE)

#set(CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}${GENCODE}")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=sm_60)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#message(CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS}")



find_package(QT NAMES Qt5  REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

find_package(glm REQUIRED)
target_link_libraries(glm::glm)


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
        main.cu
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
        scene/scene.cpp scene/scene.h
        object/visibleObject/composite.cpp object/visibleObject/composite.h
        utils/glmWrapper.h utils/glmWrapper.cpp
        manager/scene/sceneManager.cpp manager/scene/sceneManager.h
        manager/scene/sceneManagerCreator.cpp manager/scene/sceneManagerCreator.h
        manager/draw/drawManager.cpp manager/draw/drawManager.h
        manager/draw/drawManagerCreator.cpp manager/draw/drawManagerCreator.h

        renderer/Renderer.cu renderer/Renderer.h object/invisibleObject/lightSource/LightSource.cpp object/invisibleObject/lightSource/LightSource.h object/invisibleObject/lightSource/baseLightSource.h

        renderer/baseRenderer.h color/color.cpp color/color.h image_adapter/ImageAdapter.cpp image_adapter/ImageAdapter.h material/material.cpp material/material.h)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(lab_03
            MANUAL_FINALIZATION
            ${PROJECT_SOURCES}
            )
    # Define target properties for Android with Qt 6 as:
    #    set_property(TARGET lab_03 APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
    #                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
    # For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(lab_03 SHARED
                ${PROJECT_SOURCES}
                )
        # Define properties for Android with Qt 5 after find_package() calls as:
        #    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(lab_03
                ${PROJECT_SOURCES}
                )
    endif()
endif()
HEADER_DIRECTORIES(header_dir_list)

target_include_directories(
        ${PROJECT_NAME}
        PUBLIC
        ${header_dir_list}
)


find_library(LIBCONFIGPP_LOCATION libconfig++)
find_library(LIBCONFIG_LOCATION libconfig)
target_link_libraries(lab_03 PRIVATE Qt${QT_VERSION_MAJOR}::Widgets -L/opt/homebrew/Cellar/libconfig/1.7.3/lib -lconfig++)

target_compile_options(lab_03 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
        --relocatable-device-code=true
        --expt-relaxed-constexpr
        >)

set_target_properties(lab_03  PROPERTIES
        MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
        MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
        MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
        MACOSX_BUNDLE TRUE
        WIN32_EXECUTABLE TRUE
        )

install(TARGETS lab_03
        BUNDLE DESTINATION .
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(lab_03)
endif()