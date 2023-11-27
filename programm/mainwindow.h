#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QGraphicsScene>
#include <QFileDialog>
#include <QMessageBox>
#include <QKeyEvent>
#include <memory>

#include "drawManager.h"
#include "drawManagerCreator.h"
#include "sceneManager.h"
#include "sceneManagerCreator.h"

QT_BEGIN_NAMESPACE
namespace Ui
{
	class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
 Q_OBJECT

 public:
	MainWindow(QWidget* parent = nullptr);
	~MainWindow();

 public slots:
		void addToSelectionPrimitives(int idx);

 protected:
	//void resizeEvent(QResizeEvent *event) override;
	void setupScene();
	void updateScene();
	void IsCamsExist();
	void CanDeleteCam();
	void IsModelsExist();

 private:
	Ui::MainWindow* ui;

	QGraphicsScene* _scene;
	std::shared_ptr<DrawManager> _drawManager;
	std::shared_ptr<SceneManager> _sceneManager;
	/*std::shared_ptr<Facade> _facade;
	std::shared_ptr<AbstractDrawer> _drawer;

	std::shared_ptr<LoadManager> _loadManager;
	std::shared_ptr<DrawManager> _drawManager;
	std::shared_ptr<TransformManager> _transformManager;
	std::shared_ptr<SceneManager> _sceneManager;*/



	void onLightPositionChangeButtonClicked();
	void onLightColorChangeButtonClicked();



	void onImportModelButtonClicked();
	void onRemoveModelButtonClicked();
	void onRemoveAllModelsButtonClicked();

	void onMoveButtonClicked();
	void onMoveAllButtonClicked();

	void onScaleButtonClicked();
	void onScaleAllButtonClicked();

	void onRotateButtonClicked();
	void onRotateAllButtonClicked();

	void onCameraMoveUbuttonClicked();
	void onCameraMoveLbuttonClicked();
	void onCameraMoveBbuttonClicked();
	void onCameraMoveRbuttonClicked();
	void onCameraMoveULbuttonClicked();
	void onCameraMoveURbuttonClicked();
	void onCameraMoveBLbuttonClicked();
	void onCameraMoveBRbuttonClicked();

	void onCameraRotateUbuttonClicked();
	void onCameraRotateLbuttonClicked();
	void onCameraRotateBbuttonClicked();
	void onCameraRotateRbuttonClicked();
	void onCameraRotateULbuttonClicked();
	void onCameraRotateURbuttonClicked();
	void onCameraRotateBLbuttonClicked();
	void onCameraRotateBRbuttonClicked();

	void onRemoveCameraButtonClicked();
	void onAddCameraButtonClicked();
	void changeCam();

};

#endif // MAINWINDOW_H
