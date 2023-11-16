//
// Created by Андрей on 05/11/2023.
//

#include <QOpenGLFunctions_4_3_Core>
#include "RayCastingCanvas.h"

static const Vertex sg_vertexes[] = {
	Vertex( QVector3D( -1.0f,  -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f) ),
	Vertex( QVector3D(1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f) ),
	Vertex( QVector3D( -1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f) ),

	Vertex( QVector3D( 1.0f,  1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f) ),
	Vertex( QVector3D(-1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f) ),
	Vertex( QVector3D( 1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 1.0f) )

};

/*!
 * \brief Convert a QColor to a QVector3D.
 * \return A QVector3D holding a RGB representation of the colour.
 */
QVector3D to_vector3d(const QColor& colour)
{
	return QVector3D(colour.redF(), colour.greenF(), colour.blueF());
}

/*!
 * \brief Constructor for the canvas.
 * \param parent Parent widget.
 */
RayCastCanvas::RayCastCanvas(QWidget* parent)
	: QOpenGLWidget{ parent }
{
	data = new GLfloat[12];
	data[0] = -1.;  data[1] = -1.;  data[2] = 0.;
	data[3] = 1.;   data[4] = -1.;  data[5] = 0;
	data[6] = 1.;   data[7] = 1.;   data[8] = 0;
	data[9] = -1.;  data[10] = 1.;  data[11] = 0;
}

/*!
 * \brief Destructor.
 */
RayCastCanvas::~RayCastCanvas()
{
	delete data;
	/*for (auto& [key, val] : m_shaders) {
		delete val;
	}
	delete m_raycasting_volume;*/
}

/*!
 * \brief Initialise OpenGL-related state.
 */
void RayCastCanvas::initializeGL()
{
	m_program = new QOpenGLShaderProgram();
	initializeOpenGLFunctions();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	_sceneManager = SceneManagerCreator().createManager();
	_drawManager = DrawManagerCreator().createManager();
	{
		// Create Shader (Do not release until VAO is created)
		m_program = new QOpenGLShaderProgram();

		if (!m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shaders/ex.vert"))
			close();
		if (!m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shaders/ex.frag"))
			close();
		if (!m_program->link())
		{
			qDebug() << "unable to link";
			close();
		}
		m_program->bind();

		// Create Buffer (Do not release until VAO is created)
		m_vertex.create();
		m_vertex.bind();
		m_vertex.setUsagePattern(QOpenGLBuffer::DynamicCopy);
		m_vertex.allocate(sg_vertexes, sizeof(sg_vertexes));

		// Create Vertex Array Object
		m_object.create();
		m_object.bind();
		m_program->enableAttributeArray(0);
		m_program->enableAttributeArray(1);
		m_program->setAttributeBuffer(0, GL_FLOAT, Vertex::positionOffset(), Vertex::PositionTupleSize, Vertex::stride());
		m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());

		// Release (unbind) all
		m_object.release();
		m_vertex.release();
		m_program->release();
	}


}

/*!
 * \brief Callback to handle canvas resizing.
 * \param w New width.
 * \param h New height.
 */
void RayCastCanvas::resizeGL(int w, int h)
{
	//(void)w;
	//(void)h;
	glViewport(0, 0, w, h);
	//m_aspectRatio = (float) scaled_width() / scaled_height();
	//glViewport(0, 0, scaled_width(), scaled_height());
	//m_raycasting_volume->create_noise();
}

/*!
 * \brief Paint a frame on the canvas.
 */
void RayCastCanvas::paintGL()
{
	qDebug() << "started_painting\n";
	glClear(GL_COLOR_BUFFER_BIT);

	// Render using our shader
	m_program->bind();
	{
		m_object.bind();
		glDrawArrays(GL_TRIANGLES, 0, sizeof(sg_vertexes) / sizeof(sg_vertexes[0]));
		m_object.release();
	}
	m_program->release();



	/*m_program->setUniformValue("camera.position", QVector3D(cam_pos.x, cam_pos.y, cam_pos.z));
	m_program->setUniformValue("camera.view", QVector3D(cam_dir.x, cam_dir.y, cam_dir.z));

	m_program->setUniformValue("camera.up", QVector3D(cam_up.x, cam_up.y, cam_up.z));
	m_program->setUniformValue("camera.side", QVector3D(cam_r.x, cam_r.y, cam_r.z));
	m_program->setUniformValue("scale", QVector2D(width(), height()));
	m_program->setUniformValue("light.position", QVector3D(light_pos.x, light_pos.y, light_pos.z));
	m_program->setUniformValue("light.intensivity", QVector3D(light_int, light_int, light_int));

	m_program->setUniformValue("light_pos", QVector3D(light_pos.x, light_pos.y, light_pos.z));
	m_program->release();*/
	qDebug() << QString("Finished Painting");
/*
	 *    // Create Buffer (Do not release until VAO is created)
    m_vertex.create();
    m_vertex.bind();
    m_vertex.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vertex.allocate(sg_vertexes, sizeof(sg_vertexes));

    // Create Vertex Array Object
    m_object.create();
    m_object.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, Vertex::positionOffset(), Vertex::PositionTupleSize, Vertex::stride());
    m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());

    // Release (unbind) all
    m_object.release();
    m_vertex.release();
    m_program->release();
*/
	//qDebug() << "Paint GL " << cam_pos.x << ' ' << cam_pos.y << ' ' << cam_pos.z;
	//qDebug() << "Paint GL " << camera.view.x() << ' ' << camera.view.y() << ' ' << camera.view.z();
	//qDebug() << "Paint GL " << camera.up.x() << ' ' << camera.up.y() << ' ' << camera.up.z();
	//qDebug() << "Paint GL " << camera.side.x() << ' ' << camera.side.y() << ' ' << camera.side.z();
}

/*!
 * \brief Width scaled by the pixel ratio (for HiDPI devices).
 */
GLuint RayCastCanvas::scaled_width()
{
	return devicePixelRatio() * width();
}

/*!
 * \brief Height scaled by the pixel ratio (for HiDPI devices).
 */
GLuint RayCastCanvas::scaled_height()
{
	return devicePixelRatio() * height();
}

/*!
 * \brief Perform isosurface raycasting.
 */
void RayCastCanvas::raycasting(const QString& shader)
{
	/*m_shaders[shader]->bind();
	{
		m_shaders[shader]->setUniformValue("ViewMatrix", m_viewMatrix);
		m_shaders[shader]->setUniformValue("ModelViewProjectionMatrix", m_modelViewProjectionMatrix);
		m_shaders[shader]->setUniformValue("NormalMatrix", m_normalMatrix);
		m_shaders[shader]->setUniformValue("aspect_ratio", m_aspectRatio);
		m_shaders[shader]->setUniformValue("focal_length", m_focalLength);
		m_shaders[shader]->setUniformValue("viewport_size", m_viewportSize);
		m_shaders[shader]->setUniformValue("ray_origin", m_rayOrigin);
		m_shaders[shader]->setUniformValue("top", m_raycasting_volume->top());
		m_shaders[shader]->setUniformValue("bottom", m_raycasting_volume->bottom());
		m_shaders[shader]->setUniformValue("background_colour", to_vector3d(m_background));
		m_shaders[shader]->setUniformValue("light_position", m_lightPosition);
		m_shaders[shader]->setUniformValue("material_colour", m_diffuseMaterial);
		m_shaders[shader]->setUniformValue("step_length", m_stepLength);
		m_shaders[shader]->setUniformValue("threshold", m_threshold);
		m_shaders[shader]->setUniformValue("gamma", m_gamma);
		m_shaders[shader]->setUniformValue("volume", 0);
		m_shaders[shader]->setUniformValue("jitter", 1);

		glClearColor(m_background.redF(), m_background.greenF(), m_background.blueF(), m_background.alphaF());
		glClear(GL_COLOR_BUFFER_BIT);

		m_raycasting_volume->paint();
	}
	m_shaders[shader]->release();*/
}

/*!
 * \brief Convert a mouse position into normalised canvas coordinates.
 * \param p Mouse position.
 * \return Normalised coordinates for the mouse position.
 */
QPointF RayCastCanvas::pixel_pos_to_view_pos(const QPointF& p)
{
	return QPointF(2.0 * float(p.x()) / width() - 1.0,
		1.0 - 2.0 * float(p.y()) / height());
}

/*!
 * \brief Callback for mouse movement.
 */
void RayCastCanvas::mouseMoveEvent(QMouseEvent* event)
{
	/*if (event->buttons() & Qt::LeftButton) {
		m_trackBall.move(pixel_pos_to_view_pos(event->pos()), m_scene_trackBall.rotation().conjugated());
	} else {
		m_trackBall.release(pixel_pos_to_view_pos(event->pos()), m_scene_trackBall.rotation().conjugated());
	}
	update();*/
}

/*!
 * \brief Callback for mouse press.
 */
void RayCastCanvas::mousePressEvent(QMouseEvent* event)
{
	/*if (event->buttons() & Qt::LeftButton) {
		m_trackBall.push(pixel_pos_to_view_pos(event->pos()), m_scene_trackBall.rotation().conjugated());
	}*/
	update();
}

/*!
 * \brief Callback for mouse release.
 */
void RayCastCanvas::mouseReleaseEvent(QMouseEvent* event)
{
	/*if (event->button() == Qt::LeftButton) {
		m_trackBall.release(pixel_pos_to_view_pos(event->pos()), m_scene_trackBall.rotation().conjugated());
	}*/
	update();
}

/*!
 * \brief Callback for mouse wheel.
 */
void RayCastCanvas::wheelEvent(QWheelEvent* event)
{
	/*m_distExp += event->delta();
	if (m_distExp < -1800)
		m_distExp = -1800;
	if (m_distExp > 600)
		m_distExp = 600;*/
	update();
}

void RayCastCanvas::add_shader(const QString& name, const QString& vertex, const QString& fragment)
{
	/*m_shaders[name] = new QOpenGLShaderProgram(this);
	m_shaders[name]->addShaderFromSourceFile(QOpenGLShader::Vertex, vertex);
	m_shaders[name]->addShaderFromSourceFile(QOpenGLShader::Fragment, fragment);
	m_shaders[name]->link();*/
}
void RayCastCanvas::initShaders()
{
	if (!m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "../shaders/ex.vert"))
		close();

	if (!m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "../shaders/ex.frag"))
		close();

	if (m_program->link())
		close();
}
