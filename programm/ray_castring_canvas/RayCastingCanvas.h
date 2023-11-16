//
// Created by Андрей on 05/11/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_RAY_CASTRING_CANVAS_RAYCASTINGCANVAS_H_
#define LAB_03_CG_COURSE_PROGRAMM_RAY_CASTRING_CANVAS_RAYCASTINGCANVAS_H_

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

class RayCastCanvas : public QOpenGLWidget, protected QOpenGLExtraFunctions
{
 Q_OBJECT
 public:
	explicit RayCastCanvas(QWidget* parent = nullptr);
	~RayCastCanvas();

	void setStepLength(const GLfloat step_length)
	{
		m_stepLength = step_length;
		update();
	}

	void setVolume(const QString& volume)
	{
		//m_raycasting_volume->load_volume(volume);
		update();
	}

	void setThreshold(const double threshold)
	{
		//auto range = m_raycasting_volume ? getRange() : std::pair<double, double>{0.0, 1.0};
		//m_threshold = threshold / (range.second - range.first);
		update();
	}

	void setMode(const QString& mode)
	{
		m_active_mode = mode;
		update();
	}

	void setBackground(const QColor& colour)
	{
		m_background = colour;
		update();
	}

	QColor getBackground(void)
	{
		return m_background;
	}

 signals:
	// NOPE

 public slots:
	void update();

 protected:
	void initializeGL();
	void paintGL();
	void resizeGL(int width, int height);
	void initShaders();

	void mouseReleaseEvent(QMouseEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;

 private:

	QMatrix4x4 m_viewMatrix;
	QMatrix4x4 m_modelViewProjectionMatrix;
	QMatrix3x3 m_normalMatrix;

	const GLfloat m_fov = 60.0f;                                          /*!< Vertical field of view. */
	const GLfloat m_focalLength = 1.0 / qTan(M_PI / 180.0 * m_fov / 2.0); /*!< Focal length. */
	GLfloat m_aspectRatio;                                                /*!< width / height */

	QVector2D m_viewportSize;
	QVector3D m_rayOrigin; /*!< Camera position in model space coordinates. */

	QVector3D m_lightPosition{ 3.0, 0.0, 3.0 };    /*!< In camera coordinates. */
	QVector3D m_diffuseMaterial{ 1.0, 1.0, 1.0 };  /*!< Material colour. */
	GLfloat m_stepLength;                         /*!< Step length for ray march. */
	GLfloat m_threshold;                          /*!< Isosurface intensity threshold. */
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
	QOpenGLVertexArrayObject m_object;

	GLuint scaled_width();
	GLuint scaled_height();



	QPointF pixel_pos_to_view_pos(const QPointF& p);

};
#endif //LAB_03_CG_COURSE_PROGRAMM_RAY_CASTRING_CANVAS_RAYCASTINGCANVAS_H_
