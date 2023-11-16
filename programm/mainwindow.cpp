#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "sphere.h"
#include "plane.h"
#include "camera.h"
#include "Renderer.h"
#include "LightSource.h"
#include <stdlib.h>
#include "cone.h"

float GenerateRandom()
{

	return rand() / (RAND_MAX + 1.);
}

MainWindow::MainWindow(QWidget* parent)
	: QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	//_sceneManager = SceneManagerCreator().createManager();

	//_sceneManager->setScene()
	//_drawManager = DrawManagerCreator().createManager();

	//setupScene();




	/*std::string config = "/home/andrew/OOP/OOP/OOP/lab_03Another/data/config.txt";
	std::shared_ptr<ConfigCreator> config_creator = std::make_shared<ConfigCreator>();
	config_creator->createConfig()->getConfigInfo();
	_loadManager = LoadManagerCreator().createManager();

	_drawManager = DrawManagerCreator().createManager();
	_sceneManager = SceneManagerCreator().createManager();
	_transformManager = TransformManagerCreator().createManager();

	_facade = std::make_shared<Facade>(Facade());*/

	/*connect(ui->pushButton_load_model, &QPushButton::clicked, this, &MainWindow::onImportModelButtonClicked);
	connect(ui->pushButton_del_model_cur, &QPushButton::clicked, this, &MainWindow::onRemoveModelButtonClicked);
	connect(ui->pushButton_del_model_all, &QPushButton::clicked, this, &MainWindow::onRemoveAllModelsButtonClicked);

	connect(ui->pushButton_add_camera, &QPushButton::clicked, this, &MainWindow::onAddCameraButtonClicked);
	connect(ui->pushButton_del_camera_cur, &QPushButton::clicked, this, &MainWindow::onRemoveCameraButtonClicked);



	connect(ui->pushButton_move, &QPushButton::clicked, this, &MainWindow::onMoveButtonClicked);
	connect(ui->pushButton_move_all, &QPushButton::clicked, this, &MainWindow::onMoveAllButtonClicked);

	connect(ui->pushButton_scale, &QPushButton::clicked, this, &MainWindow::onScaleButtonClicked);
	connect(ui->pushButton_scale_all, &QPushButton::clicked, this, &MainWindow::onScaleAllButtonClicked);

	connect(ui->pushButton_spin, &QPushButton::clicked, this, &MainWindow::onRotateButtonClicked);
	connect(ui->pushButton_spin_all, &QPushButton::clicked, this, &MainWindow::onRotateAllButtonClicked);

	connect(ui->comboBox_cameras, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::changeCam);*/
}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::setupScene()
{
	//connect(this, SIGNAL(frameSwapped()), this, SLOT(updateScene()));
	/*_scene = new QGraphicsScene(this);

	ui->graphicsView->setScene(_scene);
	ui->graphicsView->setAlignment(Qt::AlignTop | Qt::AlignLeft);
	ui->graphicsView->setStyleSheet("QGraphicsView {background-red: white}");*/

	//auto cont = ui->graphicsView->contentsRect();
	//_scene->setSceneRect(0, 0, cont.width(), cont.height());
	ColorRGB red(1.0f, 0, 0);
	ColorRGB green(0, 1.0f, 0.0);
	ColorRGB blue(0, 0.0, 1.0f);
	Material materialRed(0.1, 1.0, 0.3, red);
	Material materialGreen(0.1, 0.3, 0.5, green);
	Material materialBlue(0.3, 0.1, 0.2, blue);


	//Generating random spheres;
	/*int spheresCount = 5;
	for (int i = 0; i < spheresCount; i++)
	{
		ColorRGB randomColor(GenerateRandom(),GenerateRandom(),GenerateRandom());
		Material randomMaterial(GenerateRandom(),GenerateRandom(),GenerateRandom(),randomColor);
		std::shared_ptr<Sphere> sphereRandom = std::make_shared<Sphere>(VecD3({-i,-i,0}),1.0,randomMaterial);
		_sceneManager->getScene()->addModel(sphereRandom);
	}*/
	/*std::shared_ptr<Cone> cone = std::make_shared<Cone>(VecD3({ 0.0, 0.0, 1.0 }), VecD3({ 1.0, 0.0,  0.0}),1.0, materialRed);
	std::shared_ptr<Sphere> sphereRed = std::make_shared<Sphere>(VecD3({ 1.0, 1.0, 0 }), 1.0, materialRed);
	std::shared_ptr<Sphere> sphereGreen = std::make_shared<Sphere>(VecD3({ 1.0, -5.0, 0 }), 1.0, materialGreen);

	std::shared_ptr<Plane>
		planeRed = std::make_shared<Plane>(VecD3{ 0.0, 0.0, -10.0 }, VecD3{ 0.0, 1.0, 0.3 }, materialBlue);
	_sceneManager->getScene()->addModel(sphereGreen);
	_sceneManager->getScene()->addModel(sphereRed);
	_sceneManager->getScene()->addModel(cone);
	//_sceneManager->getScene()->addModel(planeRed);

	std::shared_ptr<BaseLightSource> lightsource = LightSourceFactory(VecD3(-10, 0, 0), 1).create();
	lightsource->setColor(ColorRGB(1, 1, 1));
	_sceneManager->getScene()->setLightSource(lightsource);

	std::shared_ptr<BaseRenderer> renderer = std::make_shared<Renderer>(_scene);
	_drawManager->setRenderer(renderer);

	std::shared_ptr<Camera> camera = CameraFactory({ 0, 0, -5 }, { 0, 0, 1 }).create();
	//camera->setImageParams(_scene->height(), _scene->width());
	_drawManager->setCamera(camera);*/
	//_sceneManager->getScene()->addCamera(camera);
	//updateScene();

}
void MainWindow::updateScene()
{
	/*Input::update();
	std::shared_ptr<Camera> camera = _sceneManager->getScene()->getCamera();
	// Camera Transformation
	if (Input::buttonPressed(Qt::RightButton))
	{
		static const float transSpeed = 0.5f;
		static const float rotSpeed = 0.5f;

		// Handle rotations
		m_camera.rotate(-rotSpeed * Input::mouseDelta().x(), Camera3D::LocalUp);
		m_camera.rotate(-rotSpeed * Input::mouseDelta().y(), m_camera.right());

		// Handle translations
		QVector3D translation;
		if (Input::keyPressed(Qt::Key_W))
		{
			translation += m_camera.forward();
		}
		if (Input::keyPressed(Qt::Key_S))
		{
			translation -= m_camera.forward();
		}
		if (Input::keyPressed(Qt::Key_A))
		{
			translation -= m_camera.right();
		}
		if (Input::keyPressed(Qt::Key_D))
		{
			translation += m_camera.right();
		}
		if (Input::keyPressed(Qt::Key_Q))
		{
			translation -= m_camera.up();
		}
		if (Input::keyPressed(Qt::Key_E))
		{
			translation += m_camera.up();
		}
		m_camera.translate(transSpeed * translation);
	}*/

	// Update instance information
	//m_transform.rotate(1.0f, QVector3D(0.4f, 0.3f, 0.3f));

	// Schedule a redraw
	//_drawManager->drawScene(_sceneManager->getScene());

}
