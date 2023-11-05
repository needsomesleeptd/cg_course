//
// Created by Андрей on 05/11/2023.
//

#include <QOpenGLFunctions_4_3_Core>
#include "RayCastingCanvas.h"
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

}

/*!
 * \brief Destructor.
 */
RayCastCanvas::~RayCastCanvas()
{
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
	::glClearColor(1.0, 0.0, 0.0, 1.0f);
	initializeOpenGLFunctions();
	initShaders();
	qDebug() << QString("Log programm");
	qDebug() << m_program.log();

	if (!m_program.bind())
	{
		qWarning("Error bind programm");
	}

	std::shared_ptr<BaseLightSource> lightsource = LightSourceFactory(VecD3(-10, 0, 0), 1).create();
	lightsource->setColor(ColorRGB(1, 1, 1));
	_sceneManager->getScene()->setLightSource(lightsource);
	std::shared_ptr<Camera> camera = CameraFactory({ 0, 0, -5 }, { 0, 0, 1 }).create();
	_drawManager->setCamera(camera);
	_sceneManager->getScene()->addCamera(camera);

	ColorRGB red(1.0f, 0, 0);
	ColorRGB green(0, 1.0f, 0.0);
	ColorRGB blue(0, 0.0, 1.0f);
	Material materialRed(0.1, 1.0, 0.3, red);
	std::vector<Sphere> all_spheres;
	std::shared_ptr<Sphere> sphereRed = std::make_shared<Sphere>(VecD3({ 1.0, 1.0, 0 }), 1.0, materialRed);
	all_spheres.push_back(*sphereRed);

	m_program.setUniformValue("vector_size", (int)all_spheres.size());

	functions = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_4_3_Core>();
	functions->glGenBuffers(1, &ssbo);
	functions->glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	functions->glBufferData(GL_SHADER_STORAGE_BUFFER,
		all_spheres.size() * sizeof(Sphere),
		all_spheres.data(),
		GL_DYNAMIC_COPY);
	functions->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo);
	m_program.release();

}

/*!
 * \brief Callback to handle canvas resizing.
 * \param w New width.
 * \param h New height.
 */
void RayCastCanvas::resizeGL(int w, int h)
{
	(void)w;
	(void)h;
	m_viewportSize = { (float)scaled_width(), (float)scaled_height() };
	//m_aspectRatio = (float) scaled_width() / scaled_height();
	//glViewport(0, 0, scaled_width(), scaled_height());
	//m_raycasting_volume->create_noise();
}

/*!
 * \brief Paint a frame on the canvas.
 */
void RayCastCanvas::paintGL()
{

	glClear(GL_COLOR_BUFFER_BIT);
	if (!m_program.bind())
	{
		return;
	}
	m_program.enableAttributeArray(pos);
	m_program.setAttributeArray(pos, data, 3);

	glDrawArrays(GL_QUADS, 0, 4);

	std::shared_ptr<Camera> camera = _sceneManager->getScene()->getCamera();
	VecD3 cam_pos = camera->getViewPoint();
	VecD3 cam_dir = camera->getViewDirection();
	VecD3 cam_up = camera->getUpVector();
	VecD3 cam_r = cam_dir * cam_up;

	std::shared_ptr<BaseLightSource> lightSource = _sceneManager->getScene()->getLightSource();
	VecD3 light_pos = lightSource->getPosition();
	float light_int = lightSource->getIntensivity(); //TODO::fix later;

	m_program.disableAttributeArray(pos);


	m_program.setUniformValue("camera.position", QVector3D(cam_pos.x, cam_pos.y, cam_pos.z));
	m_program.setUniformValue("camera.view", QVector3D(cam_dir.x, cam_dir.y, cam_dir.z));

	m_program.setUniformValue("camera.up", QVector3D(cam_up.x, cam_up.y, cam_up.z));
	m_program.setUniformValue("camera.side", QVector3D(cam_r.x, cam_r.y, cam_r.z));
	m_program.setUniformValue("scale", QVector2D(width(), height()));
	m_program.setUniformValue("light.position",QVector3D(light_pos.x,light_pos.y,light_pos.z));
	m_program.setUniformValue("light.intensivity",QVector3D(light_int,light_int,light_int));

	m_program.setUniformValue("light_pos",QVector3D(light_pos.x,light_pos.y,light_pos.z));
	m_program.release();

	/*qDebug() << "Paint GL " << camera.pos.x() << ' ' << camera.pos.y() << ' ' << camera.pos.z();
	qDebug() << "Paint GL " << camera.view.x() << ' ' << camera.view.y() << ' ' << camera.view.z();
	qDebug() << "Paint GL " << camera.up.x() << ' ' << camera.up.y() << ' ' << camera.up.z();
	qDebug() << "Paint GL " << camera.side.x() << ' ' << camera.side.y() << ' ' << camera.side.z();*/
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
	if (!m_program.addShaderFromSourceFile(QOpenGLShader::Vertex, "../shaders/ex.vert"))
		close();

	if (!m_program.addShaderFromSourceFile(QOpenGLShader::Fragment, "../shaders/ex.frag"))
		close();

	if (m_program.link())
		close();
}
