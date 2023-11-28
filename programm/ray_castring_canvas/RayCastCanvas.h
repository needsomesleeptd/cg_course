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

class RayCastCanvas : public QOpenGLWidget, protected QOpenGLExtraFunctions
{
 Q_OBJECT
 public:
	explicit RayCastCanvas(QWidget* parent = nullptr);
	~RayCastCanvas();

 public slots:
	void update();

 protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);


	void mouseReleaseEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;

	void modifySpheres(int index, std::shared_ptr<Sphere> sphere);
	void modifyCones(int index, std::shared_ptr<Cone> cone);
	void modifyBoxes(int index, std::shared_ptr<Box> box);

	void addSphere(const std::shared_ptr<Sphere>& sphere);
	void addCone(const std::shared_ptr<Cone>& cone);
	void addBox(const std::shared_ptr<Box>& box);

	Material defaultMaterial = Material(0.1, 0.1, 0.1, ColorRGB(0.3, 0.5, 0.7));
	VecD3 defaultInitPos = VecD3(0.0);
	std::shared_ptr<Sphere> defaultSphere = std::make_shared<Sphere>(defaultInitPos, 1.0, defaultMaterial);
	std::shared_ptr<Cone>
		defaultCone = std::make_shared<Cone>(0.9, 1, defaultInitPos, VecD3(0.0, 1.0, 0.0), defaultMaterial);
	std::shared_ptr<Box>
		defaultBox = std::make_shared<Box>(defaultInitPos, glm::mat3(1.0f), VecD3(1.0, 1.0, 1.0), defaultMaterial);
	std::vector<int> shapeTypes;

 public:

	const int add_sphere_idx = 0;
	const int add_cone_idx = 1;
	const int add_cylinder_idx = 3;
	const int add_box_idx = 2;

 public:
	void movePrimitive(int idx_prim, VecD3 delta);
	void addPrimitive(int idx_prim);
	const GLfloat m_fov = 60.0f;                                          /*!< Vertical field of view. */
	QColor m_background;                          /*!< Viewport background colour. */
	QOpenGLFunctions_4_3_Core* functions;
	GLuint ssbo = 0;
	int pos;
	GLfloat* data;
	const GLfloat m_gamma = 2.2f; /*!< Gamma correction parameter. */

	std::shared_ptr<DrawManager> _drawManager;
	std::shared_ptr<SceneManager> _sceneManager;

	QOpenGLShaderProgram* m_program;
	QString m_active_mode;

	QOpenGLBuffer m_vertex;
	QOpenGLBuffer spheres;
	QOpenGLVertexArrayObject m_object;

	GLuint scaled_width();
	GLuint scaled_height();

	int spheres_count;
	int cylinders_count;
	int boxes_count;
	int cones_count;

	QPointF pixel_pos_to_view_pos(const QPointF& p);

};
#endif //LAB_03_CG_COURSE_PROGRAMM_RAY_CASTRING_CANVAS_RAYCASTCANVAS_H_
