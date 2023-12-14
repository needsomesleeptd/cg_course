#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "sphere.h"
#include "camera.h"
#include "LightSource.h"
#include <stdlib.h>
#include "cone.h"
#include "color.h"



MainWindow::MainWindow(QWidget* parent)
	: QMainWindow(parent), ui(new Ui::MainWindow)
{
	ui->setupUi(this);

	ui->addPrimitivesBox->addItem("Добавить сферу");
	ui->addPrimitivesBox->addItem("Добавить конус");
	ui->addPrimitivesBox->addItem("Добавить куб");
	ui->addPrimitivesBox->addItem("Добавить цилиндр");

	connect(ui->translate, SIGNAL(clicked()), this, SLOT(onTranslateButtonClicked()));

	connect(ui->add, SIGNAL(clicked()), this, SLOT(onAddButtonClicked()));

	connect(ui->choose_primitives_box, SIGNAL(currentIndexChanged(int)), this, SLOT(currentShapeChanged(int)));

	connect(ui->rotate, SIGNAL(clicked()), this, SLOT(onRotateButtonClicked()));


	//material work
	connect(ui->obj_R, &QDoubleSpinBox::textChanged, this, &MainWindow::materialUpdate);
	connect(ui->obj_G, &QDoubleSpinBox::textChanged, this, &MainWindow::materialUpdate);
	connect(ui->obj_B, &QDoubleSpinBox::textChanged, this, &MainWindow::materialUpdate);

	connect(ui->obj_k_a, &QDoubleSpinBox::textChanged, this, &MainWindow::materialUpdate);
	connect(ui->obj_k_d, &QDoubleSpinBox::textChanged, this, &MainWindow::materialUpdate);
	connect(ui->obj_k_s, &QDoubleSpinBox::textChanged, this, &MainWindow::materialUpdate);


	//Working with Lights

	connect(ui->light_x, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_y, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);
	connect(ui->light_z, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightPositionChangeButtonClicked);

	connect(ui->light_r, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);
	connect(ui->light_g, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);
	connect(ui->light_b, &QDoubleSpinBox::textChanged, this, &MainWindow::onLightColorChangeButtonClicked);
	//FPS
	connect(ui->graphicsView, SIGNAL(isUpdated()),this, SLOT(MainWindowFPSDisplay()));

	//Measure
	connect(ui->time_measure, SIGNAL(triggered()), this, SLOT(onMeasureTimeClicked()));
}




void MainWindow::onLightColorChangeButtonClicked()
{
	ColorRGB newColor = ColorRGB(ui->light_r->value(), ui->light_g->value(), ui->light_b->value());
	ui->graphicsView->getLight()->setColor(newColor);
}

void MainWindow::addToSelectionPrimitives(int idx)
{
	if (idx == ui->graphicsView->add_sphere_idx)
	{
		int spheres_count = ui->graphicsView->spheres_count;
		ui->choose_primitives_box->addItem(("Сфера" + std::to_string(spheres_count)).c_str());
	}
	if (idx == ui->graphicsView->add_cone_idx)
	{
		int cones_count = ui->graphicsView->cones_count;
		ui->choose_primitives_box->addItem(("Конус" + std::to_string(cones_count)).c_str());
	}

	if (idx == ui->graphicsView->add_box_idx)
	{
		int boxes_count = ui->graphicsView->boxes_count;
		ui->choose_primitives_box->addItem(("Куб" + std::to_string(boxes_count)).c_str());
	}

	if (idx == ui->graphicsView->add_cylinder_idx)
	{
		int cylinders_count = ui->graphicsView->cylinders_count;
		ui->choose_primitives_box->addItem(("Цилиндр" + std::to_string(cylinders_count)).c_str());
	}

}

MainWindow::~MainWindow()
{
	delete ui;
}

void MainWindow::setupScene()
{

}
void MainWindow::updateScene()
{

}
void MainWindow::onLightPositionChangeButtonClicked()
{
	ui->graphicsView->getLight()->setPosition(VecD3(ui->light_x->value(),
		ui->light_y->value(),
		ui->light_z->value()));
}
void MainWindow::onTranslateButtonClicked()
{
	int modelsCount = ui->graphicsView->getModelsCount();
	if (modelsCount == 0)
	{
		QMessageBox::critical(nullptr, "Ошибка", "Не добавлено ни одного примитива.");

	}
	else
	{
		int idx = ui->choose_primitives_box->currentIndex();
		VecD3 moveParams = VecD3(ui->obj_x->value(), ui->obj_y->value(), ui->obj_z->value());
		qDebug() << "started moving on" << idx << moveParams.x << moveParams.y << moveParams.z;
		ui->graphicsView->movePrimitive(idx, moveParams);
	}
}
void MainWindow::onAddButtonClicked()
{
	int idx = ui->addPrimitivesBox->currentIndex();
	ui->graphicsView->addPrimitive(idx);
	addToSelectionPrimitives(idx);


	//int models_count = ui->graphicsView->_sceneManager->getScene()->getModels().size();
	//ui->choose_primitives_box->setCurrentIndex(models_count - 1);
}
void MainWindow::materialUpdate()
{
	int modelsCount = ui->graphicsView->getModelsCount();
	if (modelsCount == 0)
	{
		QMessageBox::critical(nullptr, "Ошибка", "Не добавлено ни одного примитива.");

	}
	else
	{
		int idx_model = ui->choose_primitives_box->currentIndex();
		std::shared_ptr<BaseShape> shape = ui->graphicsView->getPrim(idx_model);
		ColorRGB color(ui->obj_R->value(), ui->obj_G->value(), ui->obj_B->value());

		Material material = Material(ui->obj_k_a->value(), ui->obj_k_d->value(), ui->obj_k_s->value(), color);

		shape->setMaterial(material);
	}

}
void MainWindow::currentShapeChanged(int shape_idx)
{
	qDebug() << "curr_shape_idx" << shape_idx;
	if (ui->choose_primitives_box->count() > 0)
	{
		std::shared_ptr<BaseShape> shape =
			ui->graphicsView->getPrim(shape_idx);
		qDebug() << "curr_shape_idx" << shape_idx;
		VecD3 pos = shape->getCenter();
		Material material = shape->getMaterial();
		ColorRGB color = material._color;
		ui->obj_x->setValue(pos.x);
		ui->obj_y->setValue(pos.y);
		ui->obj_z->setValue(pos.z);

		ui->obj_k_a->setValue(material._k_a);
		ui->obj_k_d->setValue(material._k_d);
		ui->obj_k_s->setValue(material._k_s);

		ui->obj_R->setValue(color.R);
		ui->obj_G->setValue(color.G);
		ui->obj_B->setValue(color.B);
	}

}
void MainWindow::onRotateButtonClicked()
{
	int modelsCount = ui->graphicsView->getModelsCount();
	if (modelsCount == 0)
	{
		QMessageBox::critical(nullptr, "Ошибка", "Не добавлено ни одного примитива.");

	}
	else
	{
		int idx_model = ui->choose_primitives_box->currentIndex();
		std::shared_ptr<BaseShape> shape =
			ui->graphicsView->getPrim(idx_model);
		VecD3 rotation_params = { ui->obj_rot_x->value(), ui->obj_rot_y->value(), ui->obj_rot_z->value() };
		TransformParams transformParams;
		transformParams.setRotateParams(rotation_params);
		shape->transform(transformParams);
	}
}
void MainWindow::MainWindowFPSDisplay()
{
	float fps = ui->graphicsView->getFPS();
	ui->FPS_OUTPUT->setText(QString(QString::fromStdString(std::to_string(fps))));
}
void MainWindow::onMeasureTimeClicked()
{
	//ui->graphicsView->measureTime(ui->addPrimitivesBox->currentIndex());
}
