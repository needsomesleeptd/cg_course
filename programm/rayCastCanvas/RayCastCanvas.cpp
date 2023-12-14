//
// Created by Андрей on 05/11/2023.
//

#include <QOpenGLFunctions_4_3_Core>
#include <QPaintDeviceWindow>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include "RayCastCanvas.h"
#include <QThread>
#include <QCoreApplication>

#include "vertex.h"

void delay(int waitSec)
{
	QTime dieTime= QTime::currentTime().addSecs(waitSec);
	while (QTime::currentTime() < dieTime)
		QCoreApplication::processEvents(QEventLoop::AllEvents, waitSec * 1000); // cause in mc
}

template<class T>
void setUniformArrayValue(QOpenGLShaderProgram* program,
	const QString& arrayName,
	const QString& varName,
	int index,
	const T& value)
{
	std::string name = QString("%1[%2].%3")
		.arg(arrayName)
		.arg(index)
		.arg(varName)
		.toStdString();
	program->setUniformValue(name.c_str(), value);
}

QVector3D to_q_vec(const VecD3& vec_src)
{
	QVector3D res = QVector3D(vec_src.x, vec_src.y, vec_src.z);
	return res;
};

static const Vertex sg_vertexes[] = {
	Vertex(QVector3D(-1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(-1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),

	Vertex(QVector3D(1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(-1.0f, 1.0f, 0.0f), QVector3D(0.0f, 0.0f, 0.0f)),
	Vertex(QVector3D(1.0f, -1.0f, 0.0f), QVector3D(0.0f, 0.0f, 1.0f))

};

QVector3D to_vector3d(const QColor& colour)
{
	return QVector3D(colour.redF(), colour.greenF(), colour.blueF());
}

RayCastCanvas::RayCastCanvas(QWidget* parent)
	: QOpenGLWidget{ parent }
{

}

RayCastCanvas::~RayCastCanvas()
{

}

void RayCastCanvas::initializeGL()
{
	initializeOpenGLFunctions();
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	connect(this, SIGNAL(frameSwapped()), this, SLOT(update()));


	spheresCount = 0;
	cylindersCount = 0;
	boxesCount = 0;
	conesCount = 0;

	setFocusPolicy(Qt::StrongFocus);
	_sceneManager = SceneManagerCreator().createManager();


	std::shared_ptr<Camera> camera = CameraFactory({ 0, 0, -2.0f }, { 0.0f, 0.0f, -1.0f }).create();

	_sceneManager->getScene()->addCamera(camera);
	_sceneManager->setCamera(0);

	std::shared_ptr<BaseLightSource> lightsource = LightSourceFactory(VecD3(0, 0, 0), 1.0).create();
	lightsource->setColor(ColorRGB(1, 1, 1));
	_sceneManager->getScene()->setLightSource(lightsource);

	camera->setImageParams(QWidget::height(), QWidget::width());
	m_program = new QOpenGLShaderProgram();



	{




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

		m_vertex.setUsagePattern(QOpenGLBuffer::StaticDraw);

		m_vertex.allocate(sg_vertexes, sizeof(sg_vertexes));





		// Create Vertex Array Object
		m_object.create();
		m_object.bind();
		m_program->enableAttributeArray(0);
		m_program->enableAttributeArray(1);
		m_program->enableAttributeArray(2);
		m_program->setAttributeBuffer(0,
			GL_FLOAT,
			Vertex::positionOffset(),
			Vertex::PositionTupleSize,
			Vertex::stride());
		m_program->setAttributeBuffer(1, GL_FLOAT, Vertex::colorOffset(), Vertex::ColorTupleSize, Vertex::stride());

		// Release (unbind) all
		m_object.release();
		m_vertex.release();


		//ended with transferring buffers

		m_program->setUniformValue("prLens.size_spheres", spheresCount);
		m_program->setUniformValue("prLens.size_cylinders", cylindersCount);
		m_program->setUniformValue("prLens.size_boxes", boxesCount);
		m_program->setUniformValue("prLens.size_cones", conesCount);

		m_program->release();
		//genScene(10, 1);
	}
	//onMeasureTimeClicked();
}

void RayCastCanvas::resizeGL(int w, int h)
{

	glViewport(0, 0, w, h);

}

void RayCastCanvas::paintGL()
{

	if (frameCount == 0)
		timer.start();

	//qDebug() << "started_painting\n";
	glClear(GL_COLOR_BUFFER_BIT);

	std::shared_ptr<Camera> camera = _sceneManager->getScene()->getCamera();
	std::shared_ptr<BaseLightSource> light = _sceneManager->getScene()->getLightSource();

	VecD3 cam_pos = camera->getViewPoint();
	VecD3 cam_dir = camera->getViewDirection();
	VecD3 cam_up = camera->getUpVector();

	VecD3 cam_r = camera->_cameraStructure->getRight();

	ColorRGB lSColor = light->getColor();

	QMatrix4x4 inverseProject(&camera->getInverseProjectionMatrix()[0][0]);
	QMatrix4x4 inverseView(&camera->getInverseViewMatrix()[0][0]);

	updatePrimitives();
	if (!m_program->bind())
	{
		qDebug() << "update loop binding failed";
	}

	//paths camera
	const char* cameraPosition = "camera.position";
	const char* cameraView = "camera.view";
	const char* cameraUp = "camera.up";
	const char* cameraRight = "camera.right";
	const char* invProject = "camera.inverseProjectionMatrix";
	const char* invView = "camera.inverseViewMatrix";
	//paths lights
	const char* lightPos = "lightSource.position";
	const char* lightInt ="lightSource.intensivity";

	//utils
	const char* scale = "scale";
	const char* view = "view";
	const char *projection = "projection";


	//camera
	m_program->setUniformValue(cameraPosition, QVector3D(cam_pos.x, cam_pos.y, cam_pos.z));
	m_program->setUniformValue(cameraView, QVector3D(cam_dir.x, cam_dir.y, cam_dir.z));

	m_program->setUniformValue(cameraUp, QVector3D(cam_up.x, cam_up.y, cam_up.z));
	m_program->setUniformValue(cameraRight, QVector3D(cam_r.x, cam_r.y, cam_r.z));
	m_program->setUniformValue(invProject , inverseProject);
	m_program->setUniformValue(invView, inverseView);
	//lights

	m_program->setUniformValue(lightPos, to_q_vec(light->getPosition()));
	m_program->setUniformValue(lightInt, QVector3D(lSColor.R, lSColor.G, lSColor.B));

	//scale
	m_program->setUniformValue(scale, QVector2D(QWidget::width(), QWidget::height()));

	//model + view + projection
	m_program->setUniformValue(view, QMatrix4x4(&camera->_cameraStructure->_mView[0][0]));
	m_program->setUniformValue(projection, QMatrix4x4(&camera->_cameraStructure->_mProjection[0][0]));
	//shapes_sizes
	m_program->setUniformValue("prLens.size_spheres", spheresCount);
	m_program->setUniformValue("prLens.size_cylinders", cylindersCount);
	m_program->setUniformValue("prLens.size_boxes", boxesCount);
	m_program->setUniformValue("prLens.size_cones", conesCount);

	{

		m_object.bind();
		glDrawArrays(GL_TRIANGLES, 0, sizeof(sg_vertexes) / sizeof(sg_vertexes[0]));

		m_object.release();
	}
	m_program->release();
	frameCount++;
	updateFPS();
	emit isUpdated();

}



void RayCastCanvas::update()
{
	// Update input
	Input::update();
	// Camera Transformation
	//qDebug() << "starting update";

	if (Input::buttonPressed(Qt::RightButton))
	{

		std::shared_ptr<Camera> camera = _sceneManager->getScene()->getCamera();
		VecD3 right = camera->getViewDirection() * camera->getUpVector();
		static const float transSpeed = 0.05f;
		static const float rotSpeed = 0.5f;

		float delta_x = Input::mouseDelta().x() * 0.002f;
		float delta_y = Input::mouseDelta().y() * 0.002f;
		glm::vec2 delta = { delta_x, delta_y };


		// Handle translations
		VecD3 translation = { 0.0f, 0.0f, 0.0f };
		if (Input::keyPressed(Qt::Key_W))
		{
			translation -= camera->getViewDirection();
		}
		if (Input::keyPressed(Qt::Key_S))
		{
			translation += camera->getViewDirection();
		}
		if (Input::keyPressed(Qt::Key_A))
		{
			translation += camera->_cameraStructure->getRight();
		}
		if (Input::keyPressed(Qt::Key_D))
		{
			translation -= camera->_cameraStructure->getRight();
		}
		if (Input::keyPressed(Qt::Key_Q))
		{
			translation += camera->getUpVector();
		}
		if (Input::keyPressed(Qt::Key_E))
		{
			translation -= camera->getUpVector();
		}
		//qDebug() << "translation is" << translation.x << translation.y << translation.z;
		camera->_cameraStructure->move(translation * transSpeed);
		/*qDebug() << "currentPosition is " << camera->_cameraStructure->_coordinates.x
		         << camera->_cameraStructure->_coordinates.y << camera->_cameraStructure->_coordinates.z;*/

		if (delta.x != 0.0f || delta.y != 0.0f)
		{
			float pitchDelta = delta.y * rotSpeed;
			float yawDelta = delta.x * rotSpeed;

			if (pitchDelta > 89.0f)
			{
				pitchDelta = 89.0f;
			}
			if (pitchDelta < -89.0f)
			{
				pitchDelta = -89.0f;
			}

			glm::quat q = glm::normalize(glm::cross(glm::angleAxis(pitchDelta, camera->_cameraStructure->_right),
				glm::angleAxis(-yawDelta, glm::vec3(0.f, 1.0f, 0.0f))));
			//qDebug() << "rotating by yaw " << yawDelta << "pitch " << pitchDelta << "\n";

			camera->_cameraStructure->_forward = normalise(glm::rotate(q, camera->_cameraStructure->_forward));
			camera->_cameraStructure->_right = normalise(glm::rotate(q, camera->_cameraStructure->_right));
		}
		camera->_cameraStructure->updateView();
		camera->_cameraStructure->updateProjection();
	}

	QOpenGLWidget::update();

}

void RayCastCanvas::keyPressEvent(QKeyEvent* event)
{
	if (event->isAutoRepeat())
	{
		event->ignore();
	}
	else
	{
		Input::registerKeyPress(event->key());
	}
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

void RayCastCanvas::modifySpheres(int index, std::shared_ptr<Sphere> sphere)
{

	QString sphere_pos = "spheres";
	QString center_str = "center";
	QString radius_str = "radius";
	QString mat_color_str = "material.color";
	QString mat_coefs_str = "material.lightKoefs";

	const char* sphereSizes = "prLens.size_spheres";

	Material material = sphere->getMaterial();
	QVector3D lightCoeffs = { material._k_a, material._k_d, material._k_s };
	QVector3D color = { material._color.R, material._color.G, material._color.B };
	{
		if (!m_program->bind())
		{
			qDebug() << "sphere binding failed";
		}
		setUniformArrayValue<QVector3D>(m_program, sphere_pos, center_str, index, to_q_vec(sphere->getCenter()));
		setUniformArrayValue<float>(m_program, sphere_pos, radius_str, index, (float)sphere->getRadius());
		setUniformArrayValue<QVector3D>(m_program, sphere_pos, mat_color_str, index, color);
		setUniformArrayValue<QVector3D>(m_program, sphere_pos, mat_coefs_str, index, lightCoeffs);
		m_program->setUniformValue(sphereSizes, spheresCount);
		m_program->release();
	}

}

void RayCastCanvas::addPrimitive(int idx_prim)
{

	if (idx_prim == sphereIdx)
		addSphere(defaultSphere);
	if (idx_prim == coneIdx)
		addCone(defaultCone);
	if (idx_prim == boxIdx)
		addBox(defaultBox);
	if (idx_prim == cylinderIdx)
		addCyllinder(defaultCyllinder);

}

void RayCastCanvas::movePrimitive(int idx_prim, VecD3 delta)
{

	std::shared_ptr<BaseShape>
		shape = std::dynamic_pointer_cast<BaseShape>(_sceneManager->getScene()->getModels()[idx_prim]);
	shape->move(delta);

}
void RayCastCanvas::addSphere(const std::shared_ptr<Sphere>& sphere)
{
	Sphere newSphere = *sphere.get();
	std::shared_ptr<Sphere> newSpherePtr = std::make_shared<Sphere>(newSphere);
	_sceneManager->getScene()->addModel(newSpherePtr);
	shapeTypes.push_back(sphereIdx);

	++spheresCount;
	int shape_count = spheresCount - 1;

	modifySpheres(shape_count, newSpherePtr);
	{
		m_program->bind();
		m_program->setUniformValue("prLens.size_spheres", spheresCount);
		m_program->release();
	}

}
void RayCastCanvas::addCone(const std::shared_ptr<Cone>& cone)
{
	Cone newCone = *cone;
	std::shared_ptr<Cone> newConePtr = std::make_shared<Cone>(newCone);

	_sceneManager->getScene()->addModel(newConePtr);
	shapeTypes.push_back(coneIdx);
	++conesCount;
	int shape_count = conesCount - 1;
	modifyCones(shape_count, newConePtr);


	const char* size_cones_path = "prLens.size_cones";
	{
		m_program->bind();
		m_program->setUniformValue(size_cones_path, conesCount);
		m_program->release();
	}

}
void RayCastCanvas::modifyCones(int index, std::shared_ptr<Cone> cone)
{

	int shape_count = index;


	QString coneName = "cones";
	QString angle = "cosa";
	QString height = "h";
	QString tip_position = "c";
	QString axis = "v";
	QString mat_color_str = "material.color";
	QString mat_coefs_str = "material.lightKoefs";

	Material material = cone->getMaterial();
	QVector3D lightCoeffs = { material._k_a, material._k_d, material._k_s };
	QVector3D color = { material._color.R, material._color.G, material._color.B };
	const char* size_cones_path = "prLens.size_cones";
	{
		m_program->bind();


		setUniformArrayValue<float>(m_program, coneName, angle, index, cone->_cosa);
		setUniformArrayValue<float>(m_program, coneName, height, index, cone->_h);

		setUniformArrayValue<QVector3D>(m_program, coneName, axis, index, to_q_vec(cone->_v));
		setUniformArrayValue<QVector3D>(m_program, coneName, tip_position, index, to_q_vec(cone->_c));

		setUniformArrayValue<QVector3D>(m_program, coneName, mat_color_str, index, color);
		setUniformArrayValue<QVector3D>(m_program, coneName, mat_coefs_str, index, lightCoeffs);
		m_program->setUniformValue(size_cones_path, conesCount);

		m_program->release();
	}

}
void RayCastCanvas::addBox(const std::shared_ptr<Box>& box)
{
	Box newBox = *box;
	std::shared_ptr<Box> newBoxPtr = std::make_shared<Box>(newBox);
	_sceneManager->getScene()->addModel(newBoxPtr);
	++boxesCount;
	shapeTypes.push_back(boxIdx);
	int shape_count = boxesCount - 1;

	modifyBoxes(shape_count, newBoxPtr);
	m_program->bind();
	m_program->setUniformValue("prLens.size_boxes", boxesCount);
	m_program->release();
}
void RayCastCanvas::modifyBoxes(int index, std::shared_ptr<Box> box)
{

	QString box_name = "boxes";
	QString position = "position";
	QString rotation = "rotation";
	QString halfSize = "halfSize";
	QString mat_color_str = "material.color";
	QString mat_coefs_str = "material.lightKoefs";

	Material material = box->getMaterial();
	QVector3D lightCoeffs = { material._k_a, material._k_d, material._k_s };
	QVector3D color = { material._color.R, material._color.G, material._color.B };

	m_program->bind();

	setUniformArrayValue<QVector3D>(m_program, box_name, position, index, to_q_vec(box->_position));
	setUniformArrayValue<QMatrix3x3>(m_program, box_name, rotation, index, QMatrix3x3(&box->_rotation[0][0]));

	setUniformArrayValue<QVector3D>(m_program, box_name, halfSize, index, to_q_vec(box->_halfSize));

	setUniformArrayValue<QVector3D>(m_program, box_name, mat_color_str, index, color);
	setUniformArrayValue<QVector3D>(m_program, box_name, mat_coefs_str, index, lightCoeffs);

	m_program->release();
}
void RayCastCanvas::addCyllinder(const std::shared_ptr<Cyllinder>& cyllinder)
{
	Cyllinder newCylinder = *cyllinder;
	std::shared_ptr<Cyllinder> newCylPtr = std::make_shared<Cyllinder>(newCylinder);

	_sceneManager->getScene()->addModel(newCylPtr);
	shapeTypes.push_back(cylinderIdx);
	cylindersCount++;
	int shape_count = cylindersCount - 1;

	modifyCyllinders(shape_count, newCylPtr, false);

	m_program->bind();
	m_program->setUniformValue("prLens.size_cylinders", cylindersCount);
	m_program->release();

}
void RayCastCanvas::modifyCyllinders(int index, std::shared_ptr<Cyllinder> cyllinder, bool binding)
{

	QString cyl_name = "cylinders";
	QString extr_a = "extr_a";
	QString extr_b = "extr_b";
	QString radius = "ra";
	QString mat_color_str = "material.color";
	QString mat_coefs_str = "material.lightKoefs";

	Material material = cyllinder->getMaterial();
	QVector3D lightCoeffs = { material._k_a, material._k_d, material._k_s };
	QVector3D color = { material._color.R, material._color.G, material._color.B };
	if (!binding)
		m_program->bind();

	setUniformArrayValue<QVector3D>(m_program, cyl_name, extr_a, index, to_q_vec(cyllinder->_extr_a));
	setUniformArrayValue<QVector3D>(m_program, cyl_name, extr_b, index, to_q_vec(cyllinder->_extr_b));
	setUniformArrayValue<float>(m_program, cyl_name, radius, index, cyllinder->_ra);

	setUniformArrayValue<QVector3D>(m_program, cyl_name, mat_color_str, index, color);
	setUniformArrayValue<QVector3D>(m_program, cyl_name, mat_coefs_str, index, lightCoeffs);
	if (!binding)
		m_program->release();

}
void RayCastCanvas::updatePrimitives()
{

	int shape_count =  shapeTypes.size();
	int cur_spheres_index = 0;
	int cur_cylinders_index = 0;
	int cur_boxes_index = 0;
	int cur_cones_index = 0;
	//qDebug() << "shapeTypes size" << shape_count << "models size" <<_sceneManager->getScene()->getModels().size();
	for (int i = 0; i < shape_count; i++)
	{
		std::shared_ptr<BaseShape>
			shape = std::dynamic_pointer_cast<BaseShape>(_sceneManager->getScene()->getModels()[i]);
		int shapeType = shapeTypes[i];
		if (shapeType == sphereIdx)
		{
			std::shared_ptr<Sphere> sphere = std::dynamic_pointer_cast<Sphere>(shape);
			modifySpheres(cur_spheres_index, sphere);
			++cur_spheres_index;
		}
		else if (shapeType == coneIdx)
		{
			std::shared_ptr<Cone> cone = std::dynamic_pointer_cast<Cone>(shape);
			modifyCones(cur_cones_index, cone);
			++cur_cones_index;
		}
		else if (shapeType == boxIdx)
		{
			std::shared_ptr<Box> box = std::dynamic_pointer_cast<Box>(shape);
			modifyBoxes(cur_boxes_index, box);
			++cur_boxes_index;
		}
		else if (shapeType == cylinderIdx)
		{
			std::shared_ptr<Cyllinder> cylinder = std::dynamic_pointer_cast<Cyllinder>(shape);
			modifyCyllinders(cur_cylinders_index, cylinder, false);
			++cur_cylinders_index;
		}
	}
}
void RayCastCanvas::updateFPS()
{
	fps = (double)frameCount / (timer.elapsed() / 1000.0);
	if (timer.elapsed()  > 1000) //because time is in msecs
	{
		frameCount = 0;
		//timer.restart();
	}
}
float RayCastCanvas::getFPS()
{
	updateFPS();
	return fps;
}
void RayCastCanvas::genScene(int objCount, int objType)
{


	int delta_z = 0;
	int mod = 10;
	int delta_x = 0;
	int step = 2;
	for (int i = 0; i < objCount; i++)
	{
		delta_z = (i % mod) * step;
		delta_x = i  / (mod) * step;




		VecD3 position = { delta_x, 0, delta_z };
		addPrimitive(objType);
		movePrimitive(i, position);

	}
}
void RayCastCanvas::clearScene()
{
	disconnect(this, SIGNAL(frameSwapped()), this, SLOT(update()));
	//delay(1);

	spheresCount = 0;
	cylindersCount = 0;
	conesCount = 0;
	boxesCount = 0;
	_sceneManager->getScene()->getModels().clear();
	shapeTypes.clear();
	connect(this, SIGNAL(frameSwapped()), this, SLOT(update()));
}
void RayCastCanvas::measureTime(int objType)
{
	//VecD3 CameraTestPosition = VecD3({0.0,3.0,1.0});
	//_sceneManager->getCamera()->_cameraStructure->move(CameraTestPosition);
	//QSignalBlocker blocker(this);
	int countTimes = 1;
	float fpsCount = 0.0;
	int minObj = 90;
	int maxObj = 110;
	int objStep = 10;
	int countShapeTypes = 4;
	int waitSec = 30;

	for (int j = minObj; j < maxObj; j+=objStep)
	{


			frameCount = 0;
			genScene(j, objType);
			//qDebug() << k;
			QElapsedTimer timer;
			timer.start();
			delay(waitSec);

			fpsCount = (double)frameCount / (timer.elapsed() / 1000.0);
			frameCount = 0;
			//qDebug() << conesCount << spheresCount << boxesCount << cylindersCount;
			qDebug() << "|" << fpsCount << "|" << j << "|" << "\n";
			clearScene();




	}
	clearScene();

	qDebug() << "finished";
}
std::shared_ptr<BaseShape> RayCastCanvas::getPrim(int index)
{
	return std::dynamic_pointer_cast<BaseShape>(_sceneManager->getScene()->getModels()[index]);
}
std::shared_ptr<LightSource> RayCastCanvas::getLight()
{
	return std::dynamic_pointer_cast<LightSource>(_sceneManager->getScene()->getLightSource());
}
int RayCastCanvas::getModelsCount()
{
	return _sceneManager->getScene()->getModels().size();
}


