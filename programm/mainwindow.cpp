#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget* parent)
	: QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	setupScene();

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
	_scene = new QGraphicsScene(this);

	ui->graphicsView->setScene(_scene);
	ui->graphicsView->setAlignment(Qt::AlignTop | Qt::AlignLeft);
	ui->graphicsView->setStyleSheet("QGraphicsView {background-color: white}");

	auto cont = ui->graphicsView->contentsRect();
	_scene->setSceneRect(0, 0, cont.width(), cont.height());

	/*std::shared_ptr<AbstractDrawerFactory> factory(new DrawerQtFactory(_scene));
	_drawer = factory->graphicCreate();*/
}

