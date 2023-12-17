//
// Created by Андрей on 03/11/2023.
//

#ifndef LAB_03_CG_COURSE_PROGRAMM_INPUT_INPUT_H_
#define LAB_03_CG_COURSE_PROGRAMM_INPUT_INPUT_H_

#include <Qt>
#include <QPoint>

class Input
{
 public:

	enum InputState
	{
		InputInvalid,
		InputRegistered,
		InputUnregistered,
		InputTriggered,
		InputPressed,
		InputReleased
	};

	static InputState keyState(Qt::Key key);
	static bool keyTriggered(Qt::Key key);
	static bool keyPressed(Qt::Key key);
	static bool keyReleased(Qt::Key key);
	static InputState buttonState(Qt::MouseButton button);
	static bool buttonTriggered(Qt::MouseButton button);
	static bool buttonPressed(Qt::MouseButton button);
	static bool buttonReleased(Qt::MouseButton button);
	static QPoint mousePosition();
	static QPoint mouseDelta();

 public:

	// State updating
	static void update();
	static void registerKeyPress(int key);
	static void registerKeyRelease(int key);
	static void registerMousePress(Qt::MouseButton button);
	static void registerMouseRelease(Qt::MouseButton button);
	static void reset();
	friend class MainWindow;
};

inline bool Input::keyTriggered(Qt::Key key)
{
	return keyState(key) == InputTriggered;
}

inline bool Input::keyPressed(Qt::Key key)
{
	return keyState(key) == InputPressed;
}

inline bool Input::keyReleased(Qt::Key key)
{
	return keyState(key) == InputReleased;
}

inline bool Input::buttonTriggered(Qt::MouseButton button)
{
	return buttonState(button) == InputTriggered;
}

inline bool Input::buttonPressed(Qt::MouseButton button)
{
	return buttonState(button) == InputPressed;
}

inline bool Input::buttonReleased(Qt::MouseButton button)
{
	return buttonState(button) == InputReleased;
}

#endif //LAB_03_CG_COURSE_PROGRAMM_INPUT_INPUT_H_
