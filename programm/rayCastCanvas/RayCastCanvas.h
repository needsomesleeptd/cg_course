//
// Created by Андрей on 05/11/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RAY_CASTRING_CANVAS_RAYCASTCANVAS_H_
#define LAB_03_CG_COURSE_PROGRAMM_RAY_CASTRING_CANVAS_RAYCASTCANVAS_H_

#include <functional>
#include <vector>

#include <QtMath>
#include <QOpenGLWindow>
#include <QOpenGLWidget>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_3_Core>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QTime>
#include <QElapsedTimer>

#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "Renderer.h"
#include "LightSource.h"
#include "Input.h"
#include "drawManager.h"
#include "sceneManager.h"
#include "sceneManagerCreator.h"
#include "drawManagerCreator.h"

#include "vertex.h"
#include "cone.h"
#include "box.h"
#include "cyllinder.h"

class RayCastCanvas : public QOpenGLWidget, protected QOpenGLExtraFunctions
{
 Q_OBJECT
 public:
	explicit RayCastCanvas(QWidget* parent = nullptr);
	~RayCastCanvas();

 public slots:
	void update();
    void updateFPS();
 signals:
	void isUpdated();
 protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);
	void clearScene();

	void updatePrimitives();
	void mouseReleaseEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;

	void modifySpheres(int index, std::shared_ptr<Sphere> sphere);
	void modifyCones(int index, std::shared_ptr<Cone> cone);
	void modifyBoxes(int index, std::shared_ptr<Box> box);
	void modifyCyllinders(int index, std::shared_ptr<Cyllinder> cyllinder,bool binding);

	void addSphere(const std::shared_ptr<Sphere>& sphere);
	void addCone(const std::shared_ptr<Cone>& cone);
	void addBox(const std::shared_ptr<Box>& box);
	void addCyllinder(const std::shared_ptr<Cyllinder>& cyllinder);

	Material defaultMaterial = Material(0.1, 0.1, 0.1, ColorRGB(0.3, 0.5, 0.7));
	VecD3 defaultInitPos = VecD3(0.0);
	std::shared_ptr<Sphere> defaultSphere = std::make_shared<Sphere>(defaultInitPos, 1.0, defaultMaterial);
	std::shared_ptr<Cone>
		defaultCone = std::make_shared<Cone>(0.9, 1, defaultInitPos, VecD3(0.0, -1.0, 0.0), defaultMaterial);
	std::shared_ptr<Box>
		defaultBox = std::make_shared<Box>(defaultInitPos, glm::mat3(1.0f), VecD3(1.0, 1.0, 1.0), defaultMaterial);

	std::shared_ptr<Cyllinder>
		defaultCyllinder = std::make_shared<Cyllinder>(
		VecD3(1.0, 0.0, 1.0),
		VecD3(-1.0, 0.0, -1.0),
		1.0f,
		defaultMaterial);

	std::vector<int> shapeTypes;
	int frameCount;
	float fps;
	QElapsedTimer timer;


 public:
	void measureTime(int objType);
	const int add_sphere_idx = 0;
	const int add_cone_idx = 1;
	const int add_cylinder_idx = 3;
	const int add_box_idx = 2;

 public:
	void movePrimitive(int idx_prim, VecD3 delta);
	void addPrimitive(int idx_prim);
	float getFPS();
	void genScene(int objCount, int objType);

	std::shared_ptr<DrawManager> _drawManager;
	std::shared_ptr<SceneManager> _sceneManager;

	QOpenGLShaderProgram* m_program;
	QString m_active_mode;

	QOpenGLBuffer m_vertex;
	QOpenGLVertexArrayObject m_object;



	int spheres_count;
	int cylinders_count;
	int boxes_count;
	int cones_count;



};
#endif //LAB_03_CG_COURSE_PROGRAMM_RAY_CASTRING_CANVAS_RAYCASTCANVAS_H_
