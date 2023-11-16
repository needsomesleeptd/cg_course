//
// Created by Андрей on 05/11/2023.
//

#include <QOpenGLFunctions_4_3_Core>
#include <QPaintDeviceWindow>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include "RayCastingCanvas.h"


QVector3D to_q_vec(const VecD3& vec_src)
{
	QVector3D res = QVector3D(vec_src.x,vec_src.y,vec_src.z);
	return res;
}

static const Vertex sg_vertexes[] = {
	Vertex(QVector3D(-1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(-1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),

	Vertex(QVector3D(1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(-1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 1.0f))

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
	data[0] = -1.;
	data[1] = -1.;
	data[2] = 0.;
	data[3] = 1.;
	data[4] = -1.;
	data[5] = 0;
	data[6] = 1.;
	data[7] = 1.;
	data[8] = 0;
	data[9] = -1.;
	data[10] = 1.;
	data[11] = 0;
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
	setFocusPolicy(Qt::StrongFocus);
	//qDebug() << "started initialization";
	connect(this, SIGNAL(frameSwapped()), this, SLOT(update()));
	_sceneManager = SceneManagerCreator().createManager();
	_drawManager = DrawManagerCreator().createManager();


	std::shared_ptr<Camera> camera = CameraFactory({ 0, 0, -2 }, { 0, 0, 1 }).create();
	//camera->setImageParams(_scene->height(), _scene->width());
	//_drawManager->setCamera(camera);
	_sceneManager->getScene()->addCamera(camera);

	std::shared_ptr<BaseLightSource> lightsource = LightSourceFactory(VecD3(-1, 0, 0), 1).create();
	lightsource->setColor(ColorRGB(1, 1, 1));
	_sceneManager->getScene()->setLightSource(lightsource);

	m_program = new QOpenGLShaderProgram();
	initializeOpenGLFunctions();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	{
		// Create Shader (Do not release until VAO is created)]



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
		m_vertex.setUsagePattern(QOpenGLBuffer::DynamicDraw);
		m_vertex.allocate(sg_vertexes, sizeof(sg_vertexes));

		// Create Vertex Array Object
		m_object.create();
		m_object.bind();
		m_program->enableAttributeArray(0);
		m_program->enableAttributeArray(1);
		m_program->setAttributeBuffer(0,
			GL_FLOAT,
			Vertex::positionOffset(),
			Vertex::PositionTupleSize,
			Vertex::stride());
		m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());

		// Release (unbind) all
		m_object.release();
		m_vertex.release();
		m_program->release();
	}

}

void RayCastCanvas::resizeGL(int w, int h)
{
	//(void)w;
	//(void)h;
	glViewport(0, 0, w, h);
	//m_aspectRatio = (float) scaled_width() / scaled_height();
	//glViewport(0, 0, scaled_width(), scaled_height());
	//m_raycasting_volume->create_noise();
}

void RayCastCanvas::paintGL()
{
	//qDebug() << "started_painting\n";
	glClear(GL_COLOR_BUFFER_BIT);

	std::shared_ptr<Camera> camera = _sceneManager->getScene()->getCamera();
	std::shared_ptr<BaseLightSource> light = _sceneManager->getScene()->getLightSource();

	VecD3 cam_pos = camera->getViewPoint();
	VecD3 cam_dir = camera->getViewDirection();
	VecD3 cam_up = camera->getUpVector();
	VecD3 cam_r = cam_dir * cam_up;
	QMatrix4x4 temp;
	temp.fill(1);
	// Render using our shader
	m_program->bind();
	//camera
	m_program->setUniformValue("camera.position", QVector3D(cam_pos.x, cam_pos.y, cam_pos.z));
	m_program->setUniformValue("camera.view", QVector3D(cam_dir.x, cam_dir.y, cam_dir.z));

	m_program->setUniformValue("camera.up", QVector3D(cam_up.x, cam_up.y, cam_up.z));
	m_program->setUniformValue("camera.side", QVector3D(cam_r.x, cam_r.y, cam_r.z));
	m_program->setUniformValue("camera.inverseProjectionMatrix", temp);

	//lights

	m_program->setUniformValue("light.position", to_q_vec(light->getPosition()));
	m_program->setUniformValue("light.intensivity", QVector3D(1,1,1));






	{

		m_object.bind();
		glDrawArrays(GL_TRIANGLES, 0, sizeof(sg_vertexes) / sizeof(sg_vertexes[0]));

		//m_program->setUniformValue("scale", QVector2D(width(), height()));
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
	//qDebug() << QString("Finished Painting");
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

QPointF RayCastCanvas::pixel_pos_to_view_pos(const QPointF& p)
{
	return QPointF(2.0 * float(p.x()) / width() - 1.0,
		1.0 - 2.0 * float(p.y()) / height());
}

void RayCastCanvas::update()
{
	// Update input
	Input::update();
	// Camera Transformation
	//qDebug() << "starting update";
	if (Input::buttonPressed(Qt::RightButton))
	{
		const float speed = 0.5f;
		std::shared_ptr<Camera> camera = _sceneManager->getScene()->getCamera();
		VecD3 right = camera->getViewDirection() * camera->getUpVector();
		static const float transSpeed = 0.5f;
		static const float rotSpeed = 0.5f;

		float delta_x = Input::mouseDelta().x() * 0.002f;
		float delta_y = Input::mouseDelta().y() * 0.002f;
		glm::vec2 delta = { delta_x, delta_y };
		// Handle rotations
		//camera.  (-rotSpeed * Input::mouseDelta().x(), Camera3D::LocalUp);
		//m_camera.rotate(-rotSpeed * Input::mouseDelta().y(), m_camera.right());

		// Handle translations
		VecD3 translation = {0.0f,0.0f,0.0f};
		if (Input::keyPressed(Qt::Key_W))
		{
			translation += camera->getViewDirection();
		}
		if (Input::keyPressed(Qt::Key_S))
		{
			translation -= camera->getViewDirection();
		}
		if (Input::keyPressed(Qt::Key_A))
		{
			translation -= right;
		}
		if (Input::keyPressed(Qt::Key_D))
		{
			translation += right;
		}
		if (Input::keyPressed(Qt::Key_Q))
		{
			translation -= camera->getUpVector();
		}
		if (Input::keyPressed(Qt::Key_E))
		{
			translation += camera->getUpVector();
		}
		qDebug() << "translation is" << translation.x << translation.y << translation.z;
		camera->_cameraStructure->move(translation * speed);
		if (delta.x != 0.0f || delta.y != 0.0f)
		{
			float pitchDelta = delta.y * rotSpeed;
			float yawDelta = delta.x * rotSpeed;

			glm::quat q = -glm::normalize(glm::cross(glm::angleAxis(-pitchDelta, camera->_cameraStructure->getRight()),
				glm::angleAxis(-yawDelta, glm::vec3(0.f, 1.0f, 0.0f))));

			camera->_cameraStructure->_forward = glm::rotate(q, camera->_cameraStructure->getViewDirection());
			//moved = true;
		}
	}

	// Update instance information
	//m_transform.rotate(1.0f, QVector3D(0.4f, 0.3f, 0.3f));

	// Schedule a redraw
	//QOpenGLWindow::update();
	QOpenGLWidget::update();
	//paintGL();
}

void RayCastCanvas::keyPressEvent(QKeyEvent* event)
{
	/*if (event->isAutoRepeat())
	{
		event->ignore();
	}
	else
	{*/
		Input::registerKeyPress(event->key());
	//}
	qDebug() << "key press event";
	event->accept();
}

void RayCastCanvas::keyReleaseEvent(QKeyEvent* event)
{
	if (event->isAutoRepeat())
	{
		event->ignore();
	}
	else
	{
		Input::registerKeyRelease(event->key());
	}

}

void RayCastCanvas::mousePressEvent(QMouseEvent* event)
{
	Input::registerMousePress(event->button());
}

void RayCastCanvas::mouseReleaseEvent(QMouseEvent* event)
{
	Input::registerMouseRelease(event->button());
}